# Contributing to MoodLoop AI

Thanks for your interest! Here's how to contribute.

## 🛠️ Setup

```bash
git clone https://github.com/rishikesh-2k6/MoodLoop-AI.git
cd MoodLoop-AI
pip install -r requirements.txt
ollama pull llama3
```

## 📁 Project Structure

- `core/` — Pipeline modules (TrendAnalyzer, ThemeSelector, LLMEngine, etc.)
- `media/` — Advanced video renderer with Ken Burns effects
- `assets/` — Backgrounds, music, and fonts
- `utils/` — Logging and helper utilities
- `main.py` — CLI entry point and pipeline orchestrator

## 🎨 Adding New Themes

Edit `core/ThemeSelector.py` and add a new `Theme` entry:

```python
Theme(
    name="your_theme",
    display_name="Your Theme",
    keywords=("keyword1", "keyword2"),
    mood="describe the mood",
    music_keywords=("dark", "ambient"),     # matches audio filenames
    font_file="FontName-Bold.ttf",           # from assets/fonts/
),
```

## 🎵 Adding Music

Drop royalty-free `.mp3` files into `assets/music/`. Name them descriptively — the mood-matching system uses filename keywords to pick the right track for each theme.

## 🖼️ Adding Backgrounds

Place `.png` / `.jpg` images in `assets/backgrounds/`. Ideal size: **1080×1920** (vertical).

## 🔤 Adding Fonts

Place `.ttf` / `.otf` fonts in `assets/fonts/` and reference them via `font_file` in the theme definition.

## 📝 Commit Guidelines

- Use clear, descriptive commit messages
- One feature per commit
- Test with `python main.py --no-trends --no-render` before pushing
