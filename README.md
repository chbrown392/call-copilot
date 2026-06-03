# Call Copilot

A real-time AI-powered call assistant for Windows. Captures both sides of a call simultaneously — your microphone and the caller's audio via system loopback — transcribes speech locally using faster-whisper, and streams live context and suggestions from Claude directly into a HUD overlay while you talk.

![Python](https://img.shields.io/badge/Python-3.10+-1A1A2E?style=flat-square&logo=python)
![PyQt6](https://img.shields.io/badge/UI-PyQt6-0F6E56?style=flat-square)
![Claude](https://img.shields.io/badge/AI-Claude%20API-BF5FFF?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows-185FA5?style=flat-square)

---

## What It Does

- **Dual-stream capture** — records your mic and caller audio simultaneously, in parallel
- **Local transcription** — faster-whisper runs entirely on-device, nothing leaves your machine until it hits Claude
- **Live AI assistance** — Claude receives the live transcript and streams back context, talking points, and action items in real time
- **Color-coded HUD** — cyan for YOU, purple for CALLER, green for Claude
- **Zero latency UI** — PyQt6 split-pane window that stays out of the way while you talk

---

## Screenshots

```
┌─────────────────────────────────────────────────────────────┐
│ ⚡ Call Copilot                           ● Live  ■ Stop    │
├──────────────────────────────────┬──────────────────────────┤
│ TRANSCRIPT                       │ CLAUDE                   │
│                                  │                          │
│ [YOU] Hey, can you walk me       │ Key points to address:   │
│ through the Q3 numbers?          │ • Ask about margin vs    │
│                                  │   prior quarter          │
│ [CALLER] Sure, revenue was up    │ • Confirm timeline for   │
│ 12% but margins compressed       │   Q4 guidance            │
│ slightly due to headcount...     │ • Action: follow up on   │
│                                  │   headcount freeze       │
├──────────────────────────────────┴──────────────────────────┤
│ ● YOU (Mic)    ● CALLER (Loopback)    ● Claude              │
└─────────────────────────────────────────────────────────────┘
```

---

## Requirements

- **Windows 10/11** (loopback audio capture is Windows-specific)
- **Python 3.10+**
- **VB-Audio Virtual Cable** (free) — routes system audio for loopback capture
- **Anthropic API key**

---

## Setup

### 1. Install VB-Audio Virtual Cable

Download and install from [vb-audio.com/Cable](https://vb-audio.com/Cable/).
Set it as your default playback device so call audio routes through it.

### 2. Clone and install dependencies

```bash
git clone https://github.com/yourusername/call-copilot.git
cd call-copilot
pip install -r requirements.txt
```

### 3. Set your API key

```bash
# Windows
set ANTHROPIC_API_KEY=your-api-key-here

# Or create a .env file (never commit this)
echo ANTHROPIC_API_KEY=your-api-key-here > .env
```

### 4. Find your audio device indices

```bash
python -m sounddevice
```

Look for your microphone and the VB-Audio Virtual Cable device. Update these constants at the top of `call_copilot.py`:

```python
MIC_DEVICE_INDEX      = 1   # Your microphone
LOOPBACK_DEVICE_INDEX = 27  # VB-Audio Virtual Cable
```

### 5. Run

```bash
python call_copilot.py
```

---

## Configuration

All tunable settings live at the top of `call_copilot.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `MIC_DEVICE_INDEX` | `1` | Sounddevice index for your mic |
| `LOOPBACK_DEVICE_INDEX` | `27` | Sounddevice index for VB-Audio |
| `MIC_SAMPLE_RATE` | `44100` | Native sample rate of your mic |
| `LOOPBACK_SAMPLE_RATE` | `48000` | Native sample rate of loopback |
| `SILENCE_THRESHOLD` | `0.005` | RMS below which audio is ignored |
| `CHUNK_SECONDS` | `3` | Audio buffer size before transcription |
| `WHISPER_MODEL` | `base` | faster-whisper model size |
| `CLAUDE_MODEL` | `claude-sonnet-4-5` | Anthropic model to use |

### Whisper model tradeoffs

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny` | 75MB | Fastest | Lower |
| `base` | 145MB | Fast | Good |
| `small` | 465MB | Moderate | Better |
| `medium` | 1.5GB | Slow | Best (CPU) |

`base` is the recommended starting point for most machines.

---

## How It Works

```
Microphone ──────────────────────────────┐
                                         ▼
                              AudioStream (per device)
                                  │ resample to 16kHz
                                  ▼
                          TranscriptionWorker
                              │ faster-whisper
                              ▼
                          [YOU] / [CALLER] transcript
                                  │
                                  ▼
                            ClaudeWorker
                              │ streams via Anthropic API
                              ▼
                        PyQt6 HUD (color-coded)

VB-Audio Loopback ───────────────────────┘
```

1. `AudioStream` captures audio from each device independently using `sounddevice`
2. Audio is resampled from native device rate to 16kHz using `scipy.signal.resample_poly`
3. Chunks above the silence threshold are queued for transcription
4. `TranscriptionWorker` runs faster-whisper locally to produce text
5. Transcripts are labeled with the speaker (`[YOU]` or `[CALLER]`) and appended to the HUD
6. Every few turns, the full conversation history is sent to Claude
7. Claude's response streams back in real time via the Anthropic streaming API

---

## Troubleshooting

**No audio captured**
Run `python -m sounddevice` and verify your device indices match what's set in the config.

**Transcription is slow**
Switch to `WHISPER_MODEL = "tiny"` for faster processing, or use a GPU with `WHISPER_DEVICE = "cuda"`.

**Caller audio not captured**
Make sure VB-Audio Virtual Cable is set as the default Windows playback device for the application routing call audio.

**API errors**
Verify `ANTHROPIC_API_KEY` is set in your environment. Check that you have sufficient API credits.

---

## License

MIT
