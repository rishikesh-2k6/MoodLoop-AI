# 🌙 MoodLoop AI

**Automated short-form video content engine** — generates aesthetic, mood-driven quote videos for YouTube Shorts and Instagram Reels.

MoodLoop AI combines Google Trends analysis, local LLM generation (Ollama/llama3), and FFmpeg video rendering into a single automated pipeline that produces ready-to-upload vertical videos.

---

## ✨ Features

- 🔍 **Trend Analysis** — Fetches real-time trending topics via Google Trends
- 🎨 **Theme Selection** — Picks from curated aesthetic themes (sad banger, Gen Z existential, motivational chaos, etc.)
- 🤖 **LLM Content Generation** — Generates original quotes, titles, and captions using Ollama (llama3)
- 🎬 **Video Rendering** — Produces 30-second 1080×1920 (9:16) MP4 videos with:
  - Ken Burns slow-zoom effect
  - Styled quote text overlay with semi-transparent panel
  - Background music with fade-in/fade-out
  - H.264 + AAC encoding
- 📊 **Metadata Logging** — Tracks every run in a CSV for upload automation
- #️⃣ **Hashtag Engine** — Generates platform-optimized hashtags

---

## 🏗️ Architecture

```
moodloop_ai/
├── main.py                  # CLI entry point & pipeline orchestrator
├── core/
│   ├── TrendAnalyzer.py     # Google Trends data fetcher
│   ├── ThemeSelector.py     # Theme + asset selector
│   ├── LLMEngine.py         # Ollama API integration
│   ├── TextOverlay.py       # Pillow-based quote card renderer
│   ├── VideoRenderer.py     # FFmpeg video encoder
│   ├── CaptionEngine.py     # Caption/description generator
│   └── HashtagEngine.py     # Hashtag generator
├── media/
│   └── VideoRenderer.py     # Advanced renderer with Ken Burns
├── assets/
│   ├── backgrounds/         # Background images (.png)
│   └── music/               # Royalty-free music (.mp3)
├── utils/
│   └── Logger.py            # Logging utilities
├── output/                  # Rendered videos (git-ignored)
├── logs/                    # Run logs (git-ignored)
├── requirements.txt
└── generate_upload_csv.py   # Export upload-ready CSV
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **FFmpeg** installed and on PATH ([download](https://www.gyan.dev/ffmpeg/builds/))
- **Ollama** running locally ([install](https://ollama.ai/))

### Installation

```bash
# Clone the repo
git clone https://github.com/rishikesh-2k6/MoodLoop-AI.git
cd MoodLoop-AI

# Install Python dependencies
pip install -r requirements.txt

# Pull the LLM model
ollama pull llama3
```

### Add Assets

Place your files in:
- `assets/backgrounds/` — Background images (`.png`, `.jpg`, `.webp`)
- `assets/music/` — Royalty-free audio tracks (`.mp3`, `.wav`)

---

## ▶️ Usage

```bash
# Full pipeline (trends + LLM + video render)
python main.py

# Skip Google Trends (random theme)
python main.py --no-trends

# Content generation only (no video)
python main.py --no-render

# Custom Ollama model
python main.py --model llama3:8b

# Set country for trends
python main.py --geo IN
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--geo` | `US` | Google Trends country code |
| `--model` | `llama3` | Ollama model tag |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL |
| `--no-trends` | off | Skip trends, use random theme |
| `--no-render` | off | Skip video rendering |
| `--font-path` | system default | Custom `.ttf` font for overlays |
| `--seed` | random | RNG seed for reproducibility |

---

## 📤 Export for Upload

Generate a CSV with video filenames, titles, and captions:

```bash
python generate_upload_csv.py
```

Output: `upload_info.csv` with columns `video_name`, `title`, `caption`.

---

## 🎨 Themes

| Theme | Mood |
|-------|------|
| Late Night Thoughts | Raw and vulnerable |
| Gen Z Existential | Detached yet searching |
| Sad Banger | Deeply emotional and melancholic |
| Motivational Chaos | Electric and unapologetic |
| Healing Era | Soft hope after darkness |
| Villain Arc | Cold confidence |

---

## 📝 Pipeline Output

Each run produces:
1. **Video** → `output/{run_id}.mp4` (30s, 1080×1920)
2. **Metadata** → appended row in `metadata.csv`
3. **Console summary** — quote, title, caption, theme info

---

## 🛠️ Tech Stack

- **Python 3.10+** — Core runtime
- **Ollama / llama3** — Local LLM for content generation
- **FFmpeg** — Video rendering & encoding
- **Pillow** — Quote card image compositing
- **pytrends** — Google Trends API
- **pandas** — Data handling

---

## 🔧 Setup Check

Run the setup checker to verify everything is ready:

```bash
python setup.py
```

This verifies Python version, FFmpeg, Ollama, directories, and counts your assets.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Built with 🖤 by <a href="https://github.com/rishikesh-2k6">rishikesh-2k6</a>
</p>