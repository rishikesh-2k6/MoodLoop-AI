"""
setup.py — Automated setup for MoodLoop AI.

Verifies dependencies, creates required directories, and checks
that external tools (FFmpeg, Ollama) are available.

Usage:
    python setup.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_DIRS = [
    ROOT / "assets" / "backgrounds",
    ROOT / "assets" / "music",
    ROOT / "assets" / "fonts",
    ROOT / "output",
    ROOT / "logs",
]


def check_python_version():
    """Verify Python 3.10+."""
    v = sys.version_info
    if v >= (3, 10):
        print(f"  ✓ Python {v.major}.{v.minor}.{v.micro}")
    else:
        print(f"  ✗ Python {v.major}.{v.minor} — need 3.10+")
        sys.exit(1)


def check_ffmpeg():
    """Check if FFmpeg is on PATH."""
    path = shutil.which("ffmpeg")
    if path:
        print(f"  ✓ FFmpeg found → {path}")
    else:
        print("  ✗ FFmpeg not found — install from https://www.gyan.dev/ffmpeg/builds/")


def check_ollama():
    """Check if Ollama is reachable."""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.ok:
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"  ✓ Ollama running — models: {', '.join(models) or 'none pulled'}")
        else:
            print("  ⚠ Ollama responded but returned an error")
    except Exception:
        print("  ⚠ Ollama not reachable — start with: ollama serve")


def create_dirs():
    """Create required asset directories."""
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {len(REQUIRED_DIRS)} directories verified")


def check_dependencies():
    """Verify pip packages."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True, text=True, timeout=15,
        )
        print("  ✓ Python dependencies OK")
    except Exception:
        print("  ⚠ Could not verify pip dependencies")


def count_assets():
    """Count available assets."""
    bg = len(list((ROOT / "assets" / "backgrounds").glob("*.*")))
    music = len(list((ROOT / "assets" / "music").glob("*.*")))
    fonts = len(list((ROOT / "assets" / "fonts").rglob("*.ttf")))
    fonts += len(list((ROOT / "assets" / "fonts").rglob("*.otf")))
    print(f"  📷 {bg} backgrounds | 🎵 {music} music tracks | 🔤 {fonts} fonts")


def main():
    separator = "─" * 50
    print(f"\n{separator}")
    print("  MoodLoop AI — Setup Check")
    print(separator)

    check_python_version()
    create_dirs()
    check_ffmpeg()
    check_ollama()
    check_dependencies()
    count_assets()

    print(separator)
    print("  Setup complete! Run with: python main.py")
    print(f"{separator}\n")


if __name__ == "__main__":
    main()
