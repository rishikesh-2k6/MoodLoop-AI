"""
main.py
───────
MoodLoop AI — CLI entry point and pipeline orchestrator.

Run with:
    python main.py

Optional flags (all have defaults):
    --geo          Google Trends country code       (default: US)
    --model        Ollama model tag                 (default: llama3)
    --ollama-url   Ollama server base URL           (default: http://localhost:11434)
    --no-trends    Skip Google Trends (pure random theme selection)
    --no-render    Skip video rendering (content generation only)
    --font-path    Path to a .ttf font for quote overlay
    --seed         Integer seed for reproducibility

Pipeline stages
───────────────
  1. Google Trends analysis (optional)
  2. Theme + background/music asset selection
  3. LLM content generation via Ollama / llama3
  4. Video rendering via FFmpeg + Pillow quote overlay
  5. CSV metadata logging
  6. Human-readable summary
"""

from __future__ import annotations

from typing import Optional

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────── #
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.TrendAnalyzer import TrendAnalyzer
from core.ThemeSelector import ThemeSelector, SelectedAssets
from core.LLMEngine import LLMEngine, GeneratedContent
from core.VideoRenderer import VideoRenderer, RenderResult

# ------------------------------------------------------------------ #
#  Logging configuration                                               #
# ------------------------------------------------------------------ #

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "moodloop.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("moodloop.main")

# ------------------------------------------------------------------ #
#  Constants / paths                                                   #
# ------------------------------------------------------------------ #

ASSETS_DIR = ROOT / "assets"
BACKGROUNDS_DIR = ASSETS_DIR / "backgrounds"
MUSIC_DIR = ASSETS_DIR / "music"
OUTPUT_DIR = ROOT / "output"
METADATA_CSV = ROOT / "metadata.csv"

CSV_FIELDNAMES: list[str] = [
    "run_id",
    "timestamp",
    "theme_name",
    "mood",
    "quote",
    "title",
    "caption",
    "background_file",
    "music_file",
    "video_output",
    "model",
    "trending_topics",
]


# ------------------------------------------------------------------ #
#  Pipeline steps                                                      #
# ------------------------------------------------------------------ #

def step_fetch_trends(args: argparse.Namespace) -> list[str]:
    """
    Step 1 — Fetch trending topics from Google Trends.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    list[str]
        Ranked trending keywords; empty list if trends are disabled or fail.
    """
    if args.no_trends:
        logger.info("[Step 1] Trends disabled via --no-trends flag.")
        return []

    logger.info("[Step 1] Fetching Google Trends data (geo=%s)…", args.geo)
    try:
        analyzer = TrendAnalyzer(geo=args.geo)
        topics = analyzer.get_trending_topics()
        logger.info("[Step 1] %d trending topics retrieved.", len(topics))
        return topics
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[Step 1] Google Trends fetch failed (%s). Proceeding without trends.", exc
        )
        return []


def step_select_theme(
    trending_topics: list[str], seed: int | None
) -> SelectedAssets:
    """
    Step 2 — Select a theme and resolve background / music assets.

    Parameters
    ----------
    trending_topics : list[str]
        Output from step_fetch_trends.
    seed : int | None
        Optional RNG seed for reproducibility.

    Returns
    -------
    SelectedAssets
        Resolved theme + optional file paths.
    """
    logger.info("[Step 2] Selecting theme and resolving assets…")
    selector = ThemeSelector(
        backgrounds_dir=BACKGROUNDS_DIR,
        music_dir=MUSIC_DIR,
        seed=seed,
    )
    assets = selector.select(trending_topics)

    if not assets.is_complete():
        logger.warning(
            "[Step 2] Asset selection incomplete — "
            "bg=%s | music=%s. "
            "Add files to assets/backgrounds/ and assets/music/.",
            assets.background_path,
            assets.music_path,
        )
    return assets


def step_generate_content(
    assets: SelectedAssets, args: argparse.Namespace
) -> GeneratedContent:
    """
    Step 3 — Call Ollama (llama3) to generate quote, title, and caption.

    Parameters
    ----------
    assets : SelectedAssets
        Theme metadata used to craft prompts.
    args : argparse.Namespace
        CLI args carrying model and Ollama URL.

    Returns
    -------
    GeneratedContent
        All LLM-generated text fields.

    Raises
    ------
    SystemExit
        If Ollama is unreachable after health check.
    """
    logger.info("[Step 3] Initialising LLM engine (model=%s)…", args.model)
    engine = LLMEngine(
        base_url=args.ollama_url,
        model=args.model,
    )

    if not engine.health_check():
        logger.error(
            "[Step 3] Cannot reach Ollama at %s. "
            "Start it with: ollama serve",
            args.ollama_url,
        )
        sys.exit(1)

    content = engine.generate_all(
        theme_name=assets.theme.display_name,
        mood=assets.theme.mood,
    )
    return content


def step_render_video(
    run_id: str,
    assets: SelectedAssets,
    content: GeneratedContent,
    args: argparse.Namespace,
) -> Optional["RenderResult"]:
    """
    Step 4 — Render a 30-second vertical video via FFmpeg.

    Parameters
    ----------
    run_id : str
        Unique pipeline run identifier used for the output filename.
    assets : SelectedAssets
        Resolved background + music file paths.
    content : GeneratedContent
        LLM-generated text (quote, title).
    args : argparse.Namespace
        CLI args (no-render flag, font path, ffmpeg path).

    Returns
    -------
    RenderResult | None
        Render result object, or ``None`` if rendering was skipped.
    """
    if args.no_render:
        logger.info("[Step 4] Video rendering skipped (--no-render).")
        return None

    if not assets.is_complete():
        logger.warning(
            "[Step 4] Skipping render — missing background or music. "
            "Add files to assets/backgrounds/ and assets/music/."
        )
        return None

    logger.info("[Step 4] Starting video render (run_id=%s)…", run_id)
    try:
        font_path = Path(args.font_path) if args.font_path else None
        renderer = VideoRenderer(
            output_dir=OUTPUT_DIR,
            quote_font_path=font_path,
        )
        result = renderer.render(
            run_id=run_id,
            background_path=assets.background_path,
            music_path=assets.music_path,
            quote=content.quote,
            title=content.title,
        )
        if result.success:
            logger.info("[Step 4] ✓ Video ready → %s", result.output_path)
        else:
            logger.error("[Step 4] Render failed: %s", result.error_message)
        return result
    except EnvironmentError as exc:
        logger.error("[Step 4] %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("[Step 4] Unexpected render error: %s", exc)
        return None


def step_log_metadata(
    run_id: str,
    assets: SelectedAssets,
    content: GeneratedContent,
    trending_topics: list[str],
    render_result: Optional["RenderResult"] = None,
) -> None:
    """
    Step 5 — Append a metadata row to the CSV log.

    Parameters
    ----------
    run_id : str
        Unique identifier for this pipeline run (timestamp-based).
    assets : SelectedAssets
        Resolved theme and file references.
    content : GeneratedContent
        LLM-generated text fields.
    trending_topics : list[str]
        Raw trending topics list from step 1.
    render_result : RenderResult | None
        Optional render result; output path recorded when present.
    """
    logger.info("[Step 5] Logging metadata to %s…", METADATA_CSV)

    write_header = not METADATA_CSV.exists()

    video_out = ""
    if render_result is not None and render_result.success:
        video_out = str(render_result.output_path)

    row: dict[str, str | None] = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "theme_name": assets.theme.name,
        "mood": assets.theme.mood,
        "quote": content.quote,
        "title": content.title,
        "caption": content.caption,
        "background_file": str(assets.background_path) if assets.background_path else "",
        "music_file": str(assets.music_path) if assets.music_path else "",
        "video_output": video_out,
        "model": content.model,
        "trending_topics": "; ".join(trending_topics[:5]),
    }

    try:
        with METADATA_CSV.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        logger.info("[Step 5] Metadata row written (run_id=%s).", run_id)
    except OSError as exc:
        logger.error("[Step 5] Failed to write metadata CSV: %s", exc)


# ------------------------------------------------------------------ #
#  Summary printer                                                     #
# ------------------------------------------------------------------ #

def print_run_summary(
    run_id: str,
    assets: SelectedAssets,
    content: GeneratedContent,
    render_result: Optional["RenderResult"] = None,
) -> None:
    """Pretty-print pipeline results to stdout."""
    separator = "─" * 60
    print(f"\n{separator}")
    print(f"  MoodLoop AI — Run {run_id}")
    print(separator)
    print(f"  Theme   : {assets.theme.display_name}")
    print(f"  Mood    : {assets.theme.mood}")
    print(f"  BG      : {assets.background_path or '⚠  No background found'}")
    print(f"  Music   : {assets.music_path or '⚠  No music file found'}")
    if render_result is not None:
        status = "✓" if render_result.success else "✗ FAILED"
        print(f"  Video   : {status}  {render_result.output_path}")
    else:
        print("  Video   : ⚠  Skipped (no assets or --no-render)")
    print(separator)
    print(f"  QUOTE\n  {content.quote}")
    print(separator)
    print(f"  TITLE\n  {content.title}")
    print(separator)
    print(f"  CAPTION\n  {content.caption}")
    print(separator)


# ------------------------------------------------------------------ #
#  CLI argument parser                                                 #
# ------------------------------------------------------------------ #

def build_arg_parser() -> argparse.ArgumentParser:
    """Construct and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="moodloop",
        description="MoodLoop AI — Semi-offline LLM-powered short-form content engine.",
    )
    parser.add_argument(
        "--geo",
        default="US",
        metavar="CODE",
        help="ISO 3166-1 country code for Google Trends (default: US).",
    )
    parser.add_argument(
        "--model",
        default="llama3",
        help="Ollama model tag to use for generation (default: llama3).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        dest="ollama_url",
        metavar="URL",
        help="Base URL of the Ollama API server (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--no-trends",
        action="store_true",
        help="Skip Google Trends and use pure random theme selection.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip video rendering; only generate text content.",
    )
    parser.add_argument(
        "--font-path",
        default=None,
        dest="font_path",
        metavar="PATH",
        help="Path to a .ttf font for the quote text overlay.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Integer seed for reproducible theme/asset selection.",
    )
    return parser


# ------------------------------------------------------------------ #
#  Orchestrator                                                        #
# ------------------------------------------------------------------ #

def run_pipeline(args: argparse.Namespace) -> None:
    """
    Execute the full MoodLoop AI pipeline.

    Pipeline stages
    ---------------
    1. Fetch Google Trends data (optional).
    2. Select theme + resolve background/music assets.
    3. Generate quote, title, and caption via Ollama.
    4. Render 30-second vertical video via FFmpeg (optional).
    5. Log metadata to CSV.
    6. Print human-readable summary.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("═══ MoodLoop AI — pipeline start (run_id=%s) ═══", run_id)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1 ── Trend analysis
    trending_topics = step_fetch_trends(args)

    # Stage 2 ── Theme + asset selection
    assets = step_select_theme(trending_topics, seed=args.seed)

    # Stage 3 ── LLM content generation
    content = step_generate_content(assets, args)

    # Stage 4 ── Video rendering
    render_result = step_render_video(run_id, assets, content, args)

    # Stage 5 ── Metadata logging
    step_log_metadata(run_id, assets, content, trending_topics, render_result)

    # Stage 6 ── Summary
    print_run_summary(run_id, assets, content, render_result)

    logger.info("═══ Pipeline complete (run_id=%s) ═══", run_id)


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def main() -> None:
    """Parse CLI arguments and launch the pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
