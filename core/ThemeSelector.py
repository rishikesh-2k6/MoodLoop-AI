"""
core/ThemeSelector.py
─────────────────────
Selects the active content theme for the current run by combining a
hard-coded theme catalogue with an optional trending-topic signal from
TrendAnalyzer.  The selector applies a weighted-random algorithm: themes
whose keywords match trending topics receive a boosted probability.

Also responsible for picking the concrete background image and music
track from the local asset directories.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Data model                                                          #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class Theme:
    """
    Immutable descriptor for a single content theme.

    Attributes
    ----------
    name : str
        Short identifier, e.g. ``"dark_aesthetic"``.
    display_name : str
        Human-readable label used in LLM prompts.
    keywords : tuple[str, ...]
        Lowercase terms that link this theme to trending topics.
    mood : str
        Emotional tone descriptor forwarded to the LLM.
    base_weight : float
        Default sampling weight (before trend boosting).
    """

    name: str
    display_name: str
    keywords: tuple[str, ...]
    mood: str
    base_weight: float = 1.0


@dataclass
class SelectedAssets:
    """
    Holds the resolved theme plus concrete file paths chosen for this run.

    Attributes
    ----------
    theme : Theme
        The selected theme object.
    background_path : Path | None
        Absolute path to the chosen background image.
    music_path : Path | None
        Absolute path to the chosen music track.
    """

    theme: Theme
    background_path: Optional[Path] = field(default=None)
    music_path: Optional[Path] = field(default=None)

    def is_complete(self) -> bool:
        """Return ``True`` only when both a background and music are resolved."""
        return self.background_path is not None and self.music_path is not None


# ------------------------------------------------------------------ #
#  ThemeSelector                                                       #
# ------------------------------------------------------------------ #

class ThemeSelector:
    """
    Selects a ``Theme`` using weighted-random sampling that can be biased
    by live trend data, then resolves concrete background and music assets
    from local directories.

    Parameters
    ----------
    backgrounds_dir : Path
        Directory containing background image files.
    music_dir : Path
        Directory containing music/audio files.
    trend_boost_factor : float
        Multiplier applied to the base weight of themes whose keywords
        appear in the trending topics list (default: ``2.5``).
    seed : int | None
        Optional random seed for reproducibility.
    """

    # ── Catalogue ──────────────────────────────────────────────────── #
    THEMES: tuple[Theme, ...] = (
        Theme(
            name="dark_aesthetic",
            display_name="Dark Aesthetic",
            keywords=("dark aesthetic", "dark", "gothic", "shadow", "night"),
            mood="mysterious and introspective",
        ),
        Theme(
            name="genz_existential",
            display_name="Gen Z Existential",
            keywords=("gen z", "existential", "nihilism", "purpose", "meaning"),
            mood="detached yet searching",
        ),
        Theme(
            name="late_night_thoughts",
            display_name="Late Night Thoughts",
            keywords=("night thoughts", "3am", "overthinking", "insomnia", "alone"),
            mood="raw and vulnerable",
        ),
        Theme(
            name="lofi_nostalgia",
            display_name="Lo-Fi Nostalgia",
            keywords=("lofi", "nostalgia", "retro", "chill", "vintage aesthetic"),
            mood="warm and wistful",
        ),
        Theme(
            name="sad_banger",
            display_name="Sad Banger",
            keywords=("sad", "heartbreak", "emotional", "cry", "sad lyrics"),
            mood="deeply emotional and melancholic",
        ),
        Theme(
            name="motivational_chaos",
            display_name="Motivational Chaos",
            keywords=("motivational", "hustle", "grind", "success", "ambition"),
            mood="electric and unapologetic",
        ),
        Theme(
            name="chill_vibes",
            display_name="Chill Vibes",
            keywords=("chill", "relax", "calm", "peace", "aesthetic vibes"),
            mood="serene and grounded",
        ),
    )

    # Supported media file extensions
    IMAGE_EXTENSIONS: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    AUDIO_EXTENSIONS: frozenset[str] = frozenset(
        {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"}
    )

    def __init__(
        self,
        backgrounds_dir: Path,
        music_dir: Path,
        trend_boost_factor: float = 2.5,
        seed: Optional[int] = None,
    ) -> None:
        self.backgrounds_dir = Path(backgrounds_dir)
        self.music_dir = Path(music_dir)
        self.trend_boost_factor = trend_boost_factor
        self._rng = random.Random(seed)

        self._validate_dirs()
        logger.debug(
            "ThemeSelector ready (backgrounds=%s, music=%s)",
            self.backgrounds_dir,
            self.music_dir,
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def select(
        self, trending_topics: Optional[list[str]] = None
    ) -> SelectedAssets:
        """
        Select a theme and resolve background / music assets.

        Parameters
        ----------
        trending_topics : list[str] | None
            Ranked list of trending keywords from ``TrendAnalyzer``.
            If ``None`` or empty, pure base-weight sampling is used.

        Returns
        -------
        SelectedAssets
            Resolved theme + file paths.
        """
        theme = self._pick_theme(trending_topics or [])
        background = self._pick_asset(self.backgrounds_dir, self.IMAGE_EXTENSIONS)
        music = self._pick_asset(self.music_dir, self.AUDIO_EXTENSIONS)

        assets = SelectedAssets(
            theme=theme,
            background_path=background,
            music_path=music,
        )

        logger.info(
            "Selected theme=%r | bg=%s | music=%s",
            theme.name,
            background.name if background else "NONE",
            music.name if music else "NONE",
        )
        return assets

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _pick_theme(self, trending_topics: list[str]) -> Theme:
        """
        Weighted-random theme selection.  Themes whose keywords intersect
        with *trending_topics* receive a boosted weight.

        Parameters
        ----------
        trending_topics : list[str]
            Normalised (lowercased) trending keyword strings.

        Returns
        -------
        Theme
            The selected theme.
        """
        normalised_trends = {t.lower() for t in trending_topics}
        weights: list[float] = []

        for theme in self.THEMES:
            if normalised_trends.intersection(theme.keywords):
                weights.append(theme.base_weight * self.trend_boost_factor)
                logger.debug(
                    "Boosting theme=%r (trending overlap found)", theme.name
                )
            else:
                weights.append(theme.base_weight)

        (chosen,) = self._rng.choices(self.THEMES, weights=weights, k=1)
        return chosen

    def _pick_asset(
        self, directory: Path, extensions: frozenset[str]
    ) -> Optional[Path]:
        """
        Randomly pick one file from *directory* with an extension in
        *extensions*.

        Parameters
        ----------
        directory : Path
            Directory to search.
        extensions : frozenset[str]
            Allowed file extensions (lowercase, including leading dot).

        Returns
        -------
        Path | None
            A random matching file, or ``None`` if none are found.
        """
        candidates = [
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]

        if not candidates:
            logger.warning(
                "No eligible assets found in %s (extensions=%s)",
                directory,
                extensions,
            )
            return None

        return self._rng.choice(candidates)

    def _validate_dirs(self) -> None:
        """Warn (don't crash) when asset directories are missing."""
        for path in (self.backgrounds_dir, self.music_dir):
            if not path.exists():
                logger.warning(
                    "Asset directory does not exist: %s — it will be created.", path
                )
                path.mkdir(parents=True, exist_ok=True)
