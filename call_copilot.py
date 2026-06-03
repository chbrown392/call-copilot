"""
Call Copilot v2
================
A real-time AI-powered call assistant for Windows.
Captures both sides of a call (microphone + system audio loopback),
transcribes speech locally using faster-whisper, and sends live
transcripts to Claude for intelligent, context-aware assistance.

Speaker labels:
  Cyan  (#00FFFF) = You       (microphone input)
  Purple (#BF5FFF) = Caller   (loopback / system audio)

Audio setup requires VB-Audio Virtual Cable or equivalent loopback driver.
"""

import sys
import threading
import queue
import time
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
import anthropic

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QTextEdit, QLabel, QPushButton,
    QFrame, QSizePolicy, QSystemTrayIcon, QMenu
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QFont, QColor, QTextCharFormat, QTextCursor, QIcon, QAction

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Audio device indices — run `python -m sounddevice` to list available devices
MIC_DEVICE_INDEX      = 1       # Your microphone device
LOOPBACK_DEVICE_INDEX = 27      # VB-Audio Virtual Cable (loopback)

MIC_SAMPLE_RATE      = 44100    # Native sample rate for mic device
LOOPBACK_SAMPLE_RATE = 48000    # Native sample rate for loopback device
TARGET_SAMPLE_RATE   = 16000    # Required by faster-whisper

SILENCE_THRESHOLD = 0.005       # RMS threshold below which audio is treated as silence
CHUNK_SECONDS     = 3           # Seconds of audio to buffer before transcribing
CHANNELS          = 1           # Mono audio

# faster-whisper model — options: tiny, base, small, medium, large-v2, large-v3
WHISPER_MODEL = "base"
WHISPER_DEVICE = "cpu"          # "cpu" or "cuda" (GPU if available)
WHISPER_COMPUTE = "int8"        # "int8" (CPU) or "float16" (GPU)

# Claude model
CLAUDE_MODEL = "claude-sonnet-4-5"

# UI Colors
COLOR_YOU    = "#00FFFF"        # Cyan — your microphone
COLOR_CALLER = "#BF5FFF"        # Purple — caller / loopback
COLOR_AI     = "#7FFF7F"        # Green — Claude responses
COLOR_BG     = "#0D0D0D"        # Near-black background
COLOR_TEXT   = "#E0E0E0"        # Default text

# ─── AUDIO CAPTURE ────────────────────────────────────────────────────────────

class AudioStream(QObject):
    """
    Captures audio from a single device, resamples to TARGET_SAMPLE_RATE,
    and queues chunks for transcription.
    """
    chunk_ready = pyqtSignal(np.ndarray, str)  # (audio_data, speaker_label)

    def __init__(self, device_index: int, native_rate: int, label: str):
        super().__init__()
        self.device_index = device_index
        self.native_rate  = native_rate
        self.label        = label
        self.buffer       = []
        self.running      = False
        self.stream       = None
        self._buffer_lock = threading.Lock()

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"[AudioStream:{self.label}] Status: {status}")

        audio = indata[:, 0].copy().astype(np.float32)

        # Check for silence
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < SILENCE_THRESHOLD:
            return

        # Resample to target rate if necessary
        if self.native_rate != TARGET_SAMPLE_RATE:
            gcd = np.gcd(TARGET_SAMPLE_RATE, self.native_rate)
            up   = TARGET_SAMPLE_RATE // gcd
            down = self.native_rate   // gcd
            audio = resample_poly(audio, up, down).astype(np.float32)

        with self._buffer_lock:
            self.buffer.append(audio)
            total_samples = sum(len(c) for c in self.buffer)
            if total_samples >= TARGET_SAMPLE_RATE * CHUNK_SECONDS:
                chunk = np.concatenate(self.buffer)
                self.buffer = []
                self.chunk_ready.emit(chunk, self.label)

    def start(self):
        self.running = True
        self.stream  = sd.InputStream(
            device       = self.device_index,
            samplerate   = self.native_rate,
            channels     = CHANNELS,
            dtype        = "float32",
            callback     = self._callback,
            blocksize    = int(self.native_rate * 0.1),  # 100ms blocks
        )
        self.stream.start()

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()


# ─── TRANSCRIPTION WORKER ─────────────────────────────────────────────────────

class TranscriptionWorker(QThread):
    """
    Processes audio chunks from a queue using faster-whisper.
    Emits transcribed text with speaker label.
    """
    transcript_ready = pyqtSignal(str, str)  # (text, speaker_label)

    def __init__(self):
        super().__init__()
        self.queue   = queue.Queue()
        self.running = True
        self.model   = None

    def load_model(self):
        print(f"[Whisper] Loading model: {WHISPER_MODEL}")
        self.model = WhisperModel(
            WHISPER_MODEL,
            device         = WHISPER_DEVICE,
            compute_type   = WHISPER_COMPUTE
        )
        print("[Whisper] Model loaded.")

    def enqueue(self, audio: np.ndarray, label: str):
        self.queue.put((audio, label))

    def run(self):
        self.load_model()
        while self.running:
            try:
                audio, label = self.queue.get(timeout=0.5)
                segments, _ = self.model.transcribe(
                    audio,
                    beam_size   = 5,
                    language    = "en",
                    vad_filter  = True,
                )
                text = " ".join(seg.text.strip() for seg in segments).strip()
                if text:
                    self.transcript_ready.emit(text, label)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Whisper] Transcription error: {e}")

    def stop(self):
        self.running = False
        self.wait()


# ─── CLAUDE WORKER ────────────────────────────────────────────────────────────

class ClaudeWorker(QThread):
    """
    Sends conversation history to Claude and streams back responses.
    """
    response_ready  = pyqtSignal(str)
    response_chunk  = pyqtSignal(str)
    response_done   = pyqtSignal()

    SYSTEM_PROMPT = """You are Call Copilot, a real-time assistant helping the user during a live call.
You receive a transcript of the conversation as it happens.
Your job is to:
- Surface key information, facts, or context the user might need
- Flag action items, commitments, or follow-ups
- Suggest helpful responses or talking points when relevant
- Keep responses concise — the user is on a live call

Speaker labels: YOU = the user's microphone. CALLER = the other person.
Respond in plain text. Be brief and direct."""

    def __init__(self):
        super().__init__()
        self.queue   = queue.Queue()
        self.running = True
        self.client  = anthropic.Anthropic()

    def enqueue(self, history: list):
        self.queue.put(history)

    def run(self):
        while self.running:
            try:
                history = self.queue.get(timeout=0.5)
                self._stream_response(history)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Claude] Error: {e}")

    def _stream_response(self, history: list):
        try:
            with self.client.messages.stream(
                model      = CLAUDE_MODEL,
                max_tokens = 500,
                system     = self.SYSTEM_PROMPT,
                messages   = history,
            ) as stream:
                for text in stream.text_stream:
                    self.response_chunk.emit(text)
            self.response_done.emit()
        except Exception as e:
            self.response_ready.emit(f"[Error communicating with Claude: {e}]")
            self.response_done.emit()

    def stop(self):
        self.running = False
        self.wait()


# ─── MAIN WINDOW ──────────────────────────────────────────────────────────────

class CallCopilotWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Call Copilot")
        self.setMinimumSize(900, 640)
        self.setStyleSheet(f"background-color: {COLOR_BG}; color: {COLOR_TEXT};")

        # State
        self.is_active        = False
        self.conversation     = []          # Full transcript for display
        self.claude_history   = []          # Claude message history
        self.current_ai_block = ""

        # Workers
        self.transcription_worker = TranscriptionWorker()
        self.claude_worker        = ClaudeWorker()
        self.mic_stream           = AudioStream(MIC_DEVICE_INDEX,      MIC_SAMPLE_RATE,      "YOU")
        self.loopback_stream      = AudioStream(LOOPBACK_DEVICE_INDEX, LOOPBACK_SAMPLE_RATE, "CALLER")

        # Connect signals
        self.mic_stream.chunk_ready.connect(self._on_audio_chunk)
        self.loopback_stream.chunk_ready.connect(self._on_audio_chunk)
        self.transcription_worker.transcript_ready.connect(self._on_transcript)
        self.claude_worker.response_chunk.connect(self._on_claude_chunk)
        self.claude_worker.response_done.connect(self._on_claude_done)

        self._build_ui()
        self._start_workers()

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header bar
        header = QFrame()
        header.setFixedHeight(52)
        header.setStyleSheet(f"background:#111; border-bottom:1px solid #222;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(16, 0, 16, 0)

        title = QLabel("⚡ Call Copilot")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{COLOR_YOU};")
        h_layout.addWidget(title)
        h_layout.addStretch()

        self.status_label = QLabel("● Idle")
        self.status_label.setFont(QFont("Segoe UI", 10))
        self.status_label.setStyleSheet("color:#555;")
        h_layout.addWidget(self.status_label)

        h_layout.addSpacing(16)

        self.toggle_btn = QPushButton("▶  Start Capture")
        self.toggle_btn.setFixedSize(140, 32)
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background: #0F6E56; color: white; border: none;
                border-radius: 6px; font-size: 12px; font-weight: 600;
            }
            QPushButton:hover { background: #0D5A47; }
        """)
        self.toggle_btn.clicked.connect(self._toggle_capture)
        h_layout.addWidget(self.toggle_btn)

        root.addWidget(header)

        # Main content — split view
        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)

        # Left: transcript panel
        left = QFrame()
        left.setStyleSheet(f"background:{COLOR_BG}; border-right:1px solid #1E1E1E;")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        transcript_header = QLabel("  TRANSCRIPT")
        transcript_header.setFixedHeight(32)
        transcript_header.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        transcript_header.setStyleSheet("background:#111; color:#555; padding:8px 12px; letter-spacing:2px;")
        left_layout.addWidget(transcript_header)

        self.transcript_view = QTextEdit()
        self.transcript_view.setReadOnly(True)
        self.transcript_view.setFont(QFont("Consolas", 11))
        self.transcript_view.setStyleSheet(f"""
            QTextEdit {{
                background:{COLOR_BG}; color:{COLOR_TEXT};
                border:none; padding:12px;
                selection-background-color:#333;
            }}
        """)
        left_layout.addWidget(self.transcript_view)
        content.addWidget(left, stretch=3)

        # Right: AI panel
        right = QFrame()
        right.setStyleSheet(f"background:#0A0A0A;")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        ai_header = QLabel("  CLAUDE")
        ai_header.setFixedHeight(32)
        ai_header.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        ai_header.setStyleSheet("background:#111; color:#555; padding:8px 12px; letter-spacing:2px;")
        right_layout.addWidget(ai_header)

        self.ai_view = QTextEdit()
        self.ai_view.setReadOnly(True)
        self.ai_view.setFont(QFont("Segoe UI", 11))
        self.ai_view.setStyleSheet(f"""
            QTextEdit {{
                background:#0A0A0A; color:{COLOR_TEXT};
                border:none; padding:12px;
            }}
        """)
        right_layout.addWidget(self.ai_view)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(28)
        clear_btn.setStyleSheet("background:#1A1A1A; color:#666; border:none; font-size:11px;")
        clear_btn.clicked.connect(self._clear_all)
        right_layout.addWidget(clear_btn)

        content.addWidget(right, stretch=2)

        content_widget = QWidget()
        content_widget.setLayout(content)
        root.addWidget(content_widget)

        # Legend bar
        legend = QFrame()
        legend.setFixedHeight(28)
        legend.setStyleSheet("background:#0A0A0A; border-top:1px solid #1E1E1E;")
        leg_layout = QHBoxLayout(legend)
        leg_layout.setContentsMargins(16, 0, 16, 0)

        def dot(color, label):
            w = QLabel(f"● {label}")
            w.setFont(QFont("Segoe UI", 9))
            w.setStyleSheet(f"color:{color};")
            return w

        leg_layout.addWidget(dot(COLOR_YOU,    "YOU (Mic)"))
        leg_layout.addSpacing(16)
        leg_layout.addWidget(dot(COLOR_CALLER, "CALLER (Loopback)"))
        leg_layout.addSpacing(16)
        leg_layout.addWidget(dot(COLOR_AI,     "Claude"))
        leg_layout.addStretch()

        root.addWidget(legend)

    # ── Worker Management ────────────────────────────────────────────────────

    def _start_workers(self):
        self.transcription_worker.start()
        self.claude_worker.start()

    def _toggle_capture(self):
        if not self.is_active:
            self._start_capture()
        else:
            self._stop_capture()

    def _start_capture(self):
        self.is_active = True
        self.mic_stream.start()
        self.loopback_stream.start()
        self.status_label.setText("● Live")
        self.status_label.setStyleSheet(f"color:{COLOR_YOU};")
        self.toggle_btn.setText("■  Stop Capture")
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background:#7A1A1A; color:white; border:none;
                border-radius:6px; font-size:12px; font-weight:600;
            }
            QPushButton:hover { background:#991F1F; }
        """)

    def _stop_capture(self):
        self.is_active = False
        self.mic_stream.stop()
        self.loopback_stream.stop()
        self.status_label.setText("● Idle")
        self.status_label.setStyleSheet("color:#555;")
        self.toggle_btn.setText("▶  Start Capture")
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background:#0F6E56; color:white; border:none;
                border-radius:6px; font-size:12px; font-weight:600;
            }
            QPushButton:hover { background:#0D5A47; }
        """)

    # ── Signal Handlers ──────────────────────────────────────────────────────

    def _on_audio_chunk(self, audio: np.ndarray, label: str):
        self.transcription_worker.enqueue(audio, label)

    def _on_transcript(self, text: str, label: str):
        color = COLOR_YOU if label == "YOU" else COLOR_CALLER
        self._append_transcript(f"[{label}] {text}", color)

        # Build Claude message
        self.claude_history.append({
            "role": "user",
            "content": f"[{label}]: {text}"
        })

        # Send to Claude every few turns to avoid over-firing
        if len(self.claude_history) % 2 == 0 or label == "CALLER":
            self.claude_worker.enqueue(list(self.claude_history))

    def _on_claude_chunk(self, chunk: str):
        self.current_ai_block += chunk
        cursor = self.ai_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(COLOR_AI))
        cursor.insertText(chunk, fmt)
        self.ai_view.setTextCursor(cursor)
        self.ai_view.ensureCursorVisible()

    def _on_claude_done(self):
        if self.current_ai_block:
            self.claude_history.append({
                "role": "assistant",
                "content": self.current_ai_block
            })
            self.current_ai_block = ""
        self._append_ai("\n")

    def _append_transcript(self, text: str, color: str):
        cursor = self.transcript_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.insertText(text + "\n", fmt)
        self.transcript_view.setTextCursor(cursor)
        self.transcript_view.ensureCursorVisible()

    def _append_ai(self, text: str):
        cursor = self.ai_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.ai_view.setTextCursor(cursor)
        self.ai_view.ensureCursorVisible()

    def _clear_all(self):
        self.transcript_view.clear()
        self.ai_view.clear()
        self.claude_history.clear()
        self.conversation.clear()

    def closeEvent(self, event):
        self._stop_capture()
        self.transcription_worker.stop()
        self.claude_worker.stop()
        event.accept()


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Call Copilot")
    app.setStyle("Fusion")

    window = CallCopilotWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
