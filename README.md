# MoodLoop AI 🎬🖤

> **A Semi-Offline LLM-Powered Short-Form Content Intelligence Engine**
>
> Generate dark aesthetic, Gen Z-style vertical videos — complete with AI-written quotes, Ken Burns visuals, and trending hashtags — entirely from your local machine.

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Project Structure](#-project-structure)
4. [Installation](#-installation)
5. [Ollama Setup](#-ollama-setup)
6. [FFmpeg Setup](#-ffmpeg-setup)
7. [How to Run](#-how-to-run)
8. [Example Output](#-example-output)
9. [Module Reference](#-module-reference)
10. [Future Improvements](#-future-improvements)

---

## 🎯 Project Overview

MoodLoop AI automates the creation of 30-second **9:16 vertical videos** (Instagram Reels / YouTube Shorts) by chaining together:

| Step | What happens | Technology |
|------|-------------|------------|
| 1 | Fetch live trending topics | Google Trends via `pytrends` |
| 2 | Pick a content theme | Weighted-random with trend-boost |
| 3 | Resolve background image + music | Local `assets/` directory |
| 4 | Generate quote, title, caption | **Ollama / llama3** (local LLM) |
| 5 | Build platform hashtag block | Layered `HashtagEngine` |
| 6 | Render 30-second MP4 | **FFmpeg** (Ken Burns + drawtext) |
| 7 | Log metadata | CSV via `RunLogger` |

The system is **semi-offline**: Ollama runs locally (no OpenAI API key needed) and the only external call is the optional Google Trends query.

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         main.py (CLI)                        │
│   parse args → run_pipeline() → 6 ordered pipeline stages   │
└────────────────────────┬─────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────────────────┐
         ▼               ▼                           ▼
  ┌─────────────┐  ┌───────────────┐       ┌────────────────┐
  │TrendAnalyzer│  │ ThemeSelector │       │   AssetManager │
  │  (pytrends) │  │ (7 themes,    │       │ (bg + music,   │
  │  batched,   │  │  weighted-    │       │  anti-repeat,  │
  │  retry)     │  │  random)      │       │  theme dirs)   │
  └──────┬──────┘  └───────┬───────┘       └───────┬────────┘
         │                 │                        │
         └────────────┬────┘                        │
                      ▼                             │
             ┌────────────────┐                     │
             │   LLMEngine    │◄────────────────────┘
             │ (Ollama HTTP,  │
             │  quote/title/  │
             │  caption gen)  │
             └───────┬────────┘
                     │
          ┌──────────┼─────────────┐
          ▼          ▼             ▼
  ┌──────────────┐ ┌───────────┐ ┌───────────────┐
  │CaptionEngine │ │ Hashtag   │ │ media/Video   │
  │(LLM/template,│ │ Engine    │ │ Renderer      │
  │ platform     │ │ (5-layer) │ │ (FFmpeg,      │
  │ char-limit)  │ │           │ │  Ken Burns,   │
  └──────────────┘ └───────────┘ │  drawtext)    │
                                 └───────┬───────┘
                                         │
                                 ┌───────▼───────┐
                                 │  utils/Logger │
                                 │  (CSV, thread │
                                 │   -safe)      │
                                 └───────────────┘
```

### Design Principles

- **Strict OOP** — every subsystem is a class with type hints and docstrings
- **Pathlib throughout** — no raw string paths
- **Graceful degradation** — each stage logs warnings and continues if optional services (Trends, LLM) are unavailable
- **Modular** — swap any module (e.g. replace `LLMEngine` with GPT-4) without touching others
- **No global state** — all configuration flows through constructor parameters or CLI args

---

## 📁 Project Structure

```
moodloop_ai/
│
├── main.py                     ← CLI entry point + pipeline orchestrator
├── requirements.txt
├── metadata.csv                ← auto-created on first run
│
├── core/                       ← Domain logic
│   ├── __init__.py
│   ├── TrendAnalyzer.py        ← Google Trends fetcher (pytrends)
│   ├── ThemeSelector.py        ← 7 themes, weighted-random selection
│   ├── LLMEngine.py            ← Ollama HTTP client (quote/title/caption)
│   ├── CaptionEngine.py        ← Caption builder (LLM + template modes)
│   ├── HashtagEngine.py        ← 5-layer hashtag block generator
│   ├── TextOverlay.py          ← Pillow quote-card compositor
│   └── VideoRenderer.py        ← (core) FFmpeg still-image encoder
│
├── media/                      ← Media handling
│   ├── __init__.py
│   ├── AssetManager.py         ← bg + music selection, anti-repetition
│   └── VideoRenderer.py        ← (media) FFmpeg Ken Burns renderer ★
│
├── utils/                      ← Shared utilities
│   ├── __init__.py
│   └── Logger.py               ← Thread-safe CSV RunLogger
│
└── assets/
    ├── backgrounds/             ← Drop your .jpg/.png images here
    │   ├── dark_aesthetic/      ← Optional per-theme sub-folders
    │   └── lofi_nostalgia/
    └── music/                   ← Drop your .mp3/.wav tracks here
        └── dark_aesthetic/
```

---

## ⚙️ Installation

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.10 | `python --version` |
| Ollama | latest | local LLM runtime |
| FFmpeg | ≥ 5.0 | video encoding |

### 1 · Clone / navigate to the project

```bash
cd "autimation -2/moodloop_ai"
```

### 2 · Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3 · Install Python dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**

| Package | Purpose |
|---------|---------|
| `Pillow` | Quote-card image compositing |
| `requests` | Ollama HTTP API calls |
| `pytrends` | Google Trends data |
| `pandas` | Data handling within pytrends |
| `python-dotenv` | Optional `.env` config |
| `loguru` | Rich logging (optional enhancement) |

---

## 🤖 Ollama Setup

Ollama is a local LLM runner — no GPU required for llama3 (runs on CPU).

### 1 · Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows  →  download from https://ollama.com/download
```

### 2 · Pull the llama3 model

```bash
ollama pull llama3
```

> First pull is ~4.7 GB. Subsequent runs use the cached model.

### 3 · Start the Ollama server

```bash
ollama serve
```

Ollama listens on `http://localhost:11434` by default.
MoodLoop AI will automatically health-check this before running.

### Verify

```bash
curl http://localhost:11434
# Expected: "Ollama is running"
```

---

## 🎞 FFmpeg Setup

FFmpeg handles all video encoding — Ken Burns zoom, text overlay, audio mixing.

### Windows

1. Download the **full build** from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/)
2. Extract to `C:\ffmpeg\`
3. Add `C:\ffmpeg\bin` to your **System PATH**:
   - Search → *Edit the system environment variables* → Environment Variables → Path → New → `C:\ffmpeg\bin`
4. Restart your terminal, then verify:

```powershell
ffmpeg -version
```

### macOS

```bash
brew install ffmpeg
```

### Linux (Debian / Ubuntu)

```bash
sudo apt update && sudo apt install ffmpeg -y
```

---

## ▶️ How to Run

### Basic run (full pipeline with trend biasing)

```bash
python main.py
```

### Skip Google Trends (fastest, pure-random theme)

```bash
python main.py --no-trends
```

### Text-only run — skip video rendering (useful for testing)

```bash
python main.py --no-render
```

### Custom font for the text overlay

```bash
python main.py --font-path "C:/Windows/Fonts/arialbd.ttf"
```

### Target a specific country's trends

```bash
python main.py --geo IN          # India
python main.py --geo GB          # United Kingdom
```

### Use a different Ollama model

```bash
python main.py --model mistral
python main.py --model llama3:8b
```

### Reproducible run (fixed seed)

```bash
python main.py --seed 42
```

### All flags at once

```bash
python main.py \
  --geo US \
  --model llama3 \
  --font-path path/to/font.ttf \
  --seed 7 \
  --no-trends
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--geo CODE` | `US` | Google Trends country code |
| `--model TAG` | `llama3` | Ollama model tag |
| `--ollama-url URL` | `http://localhost:11434` | Ollama server URL |
| `--no-trends` | off | Skip Google Trends |
| `--no-render` | off | Skip video rendering |
| `--font-path PATH` | none | Custom .ttf font for overlay |
| `--seed N` | none | RNG seed for reproducibility |

---

## 📊 Example Output

After a successful run you will see:

```
────────────────────────────────────────────────────────────
  MoodLoop AI — Run 20240224_201500
────────────────────────────────────────────────────────────
  Theme   : Dark Aesthetic
  Mood    : mysterious and introspective
  BG      : night_city.jpg
  Music   : dark_ambient_01.mp3
  Video   : ✓  output/20240224_201500.mp4
────────────────────────────────────────────────────────────
  QUOTE
  you stopped explaining yourself the day you realised
  silence hits harder than any sentence.
────────────────────────────────────────────────────────────
  TITLE
  silence is the loudest thing you never said
────────────────────────────────────────────────────────────
  CAPTION
  some things live rent-free in your mind at 3am.

  "you stopped explaining yourself the day you realised
  silence hits harder than any sentence."

  this one is for the ones who feel everything too deeply. 🖤

  #DarkAesthetic #DarkVibes #AestheticQuotes #MidnightMood
  #FYP #Shorts #ViralQuotes #Reels
────────────────────────────────────────────────────────────
```

### Output files

| File | Description |
|------|-------------|
| `output/YYYYMMDD_HHMMSS.mp4` | 1080×1920 H.264/AAC video, ~30 s |
| `metadata.csv` | Appended run record |
| `logs/moodloop.log` | Full pipeline log |

### What's inside the MP4

```
Duration  : 30 seconds
Resolution: 1080 × 1920 (9:16 portrait)
Video     : H.264, CRF 22, 30 fps, yuv420p
Audio     : AAC 192 kbps (stereo)
Effect    : Ken Burns slow zoom (1.0× → 1.08×)
Text      : Centred quote, box backdrop, drop shadow
Fades     : 1 s fade-in / fade-out (video + audio)
```

### metadata.csv columns

| Column | Example |
|--------|---------|
| `run_id` | `20240224_201500` |
| `timestamp` | `2024-02-24T20:15:00` |
| `theme_name` | `dark_aesthetic` |
| `mood` | `mysterious and introspective` |
| `quote` | *generated text* |
| `title` | *generated text* |
| `caption` | *generated text* |
| `background_file` | `assets/backgrounds/night_city.jpg` |
| `music_file` | `assets/music/dark_ambient_01.mp3` |
| `video_output` | `output/20240224_201500.mp4` |
| `model` | `llama3` |
| `trending_topics` | `dark aesthetic; Gen Z quotes; …` |

---

## 📦 Module Reference

| Module | Class | Key method(s) |
|--------|-------|--------------|
| `core/TrendAnalyzer.py` | `TrendAnalyzer` | `get_trending_topics()`, `get_top_topic()` |
| `core/ThemeSelector.py` | `ThemeSelector` | `select(trending_topics)` |
| `core/LLMEngine.py` | `LLMEngine` | `generate_all()`, `health_check()` |
| `core/CaptionEngine.py` | `CaptionEngine` | `generate(theme_name, quote, …)` |
| `core/HashtagEngine.py` | `HashtagEngine` | `build(theme_name, trending_topics)` |
| `core/TextOverlay.py` | `TextOverlay` | `render(background, quote, output_path)` |
| `core/VideoRenderer.py` | `VideoRenderer` | `render(run_id, bg, music, quote, title)` |
| `media/AssetManager.py` | `AssetManager` | `get_background()`, `get_music()` |
| `media/VideoRenderer.py` | `VideoRenderer` | `render()`, `render_with_random_ken_burns()` |
| `utils/Logger.py` | `RunLogger` | `log_run(**kwargs)`, `row_count()` |

---

## 🚀 Future Improvements

### Content Intelligence
- [ ] **Sentiment-aware theming** — analyse quote sentiment (VADER / TextBlob) and auto-select the most fitting theme rather than weighting by keyword overlap
- [ ] **Multi-language support** — generate quotes in Hindi, Telugu, Spanish via multilingual Ollama models
- [ ] **Quote library cache** — save generated quotes to SQLite; avoid duplicates across runs

### Visual Pipeline
- [ ] **Dynamic transitions** — support multiple background images per video with crossfade transitions (already structured for it in `VideoRenderer`)
- [ ] **Animated text** — use FFmpeg's `drawtext:enable` expression to fade text in mid-video
- [ ] **AI background generation** — integrate Stable Diffusion (local via `diffusers`) to generate unique backgrounds instead of relying on local folders
- [ ] **Subtitle .srt export** — auto-generate a timed subtitle file alongside each video

### Distribution & Automation
- [ ] **YouTube Shorts uploader** — `youtube_uploader.py` using the YouTube Data API v3 with OAuth2
- [ ] **Instagram Reels uploader** — via the Instagram Graph API
- [ ] **GitHub Actions scheduler** — run the full pipeline twice daily in the cloud (`.github/workflows/generate.yml`)
- [ ] **Google Drive sync** — upload rendered MP4s to Drive for cloud backup

### Developer Experience
- [ ] **`.env` configuration** — move all defaults (geo, model, font path) to a `.env` file loaded by `python-dotenv`
- [ ] **Web dashboard** — FastAPI + HTMX admin panel to browse `metadata.csv`, preview thumbnails, and trigger runs
- [ ] **Unit test suite** — pytest tests for `ThemeSelector`, `HashtagEngine`, `CaptionEngine` (template mode), and `RunLogger`
- [ ] **Docker image** — `Dockerfile` with FFmpeg + Python pre-installed; mounts `assets/` as a volume

---

## 📄 License

MIT — do whatever you like, ship whatever you build. Attribution is always appreciated. 🖤
