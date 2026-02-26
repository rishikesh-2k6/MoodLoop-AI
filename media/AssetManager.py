"""
media/AssetManager.py
─────────────────────
Manages random selection of local background images and music tracks
with an anti-repetition guard: the same asset will never be returned
twice in a row.

Supports sub-theme routing — if a ``theme_name`` is supplied the manager
first looks for a matching sub-directory inside ``backgrounds_dir``.  If
no sub-directory exists it falls back to the root of ``backgrounds_dir``.

Directory layout expected (all files optional):
    assets/
        backgrounds/
            dark_aesthetic/   ← optional per-theme folder
            lofi_nostalgia/
            *.jpg / *.png     ← fallback pool
        music/
            dark_aesthetic/
            *.mp3             ← fallback pool
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AssetManager:
    """
    Selects background images and music tracks from local asset directories
    with anti-repetition logic so the same file is never returned twice in
    immediate succession.

    Parameters
    ----------
    backgrounds_dir : Path
        Root directory that holds background images.
    music_dir : Path
        Root directory that holds music / audio files.
    seed : int | None
        Optional RNG seed for reproducible selections.
    """

    # Supported file extensions
    IMAGE_EXT: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    )
    AUDIO_EXT: frozenset[str] = frozenset(
        {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"}
    )

    def __init__(
        self,
        backgrounds_dir: Path,
        music_dir: Path,
        seed: Optional[int] = None,
    ) -> None:
        self.backgrounds_dir = Path(backgrounds_dir)
        self.music_dir = Path(music_dir)
        self._rng = random.Random(seed)

        # Anti-repetition state: stores the last returned path per category
        self._last_background: Optional[Path] = None
        self._last_music: Optional[Path] = None

        self._ensure_dirs()
        logger.debug(
            "AssetManager ready (bg=%s, music=%s)",
            self.backgrounds_dir,
            self.music_dir,
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_background(self, theme_name: Optional[str] = None) -> Optional[Path]:
        """
        Return a random background image, preferring theme-specific assets.

        The returned file will differ from the previously returned file
        whenever the pool contains more than one option (anti-repetition).

        Parameters
        ----------
        theme_name : str | None
            If provided, checks for a sub-directory whose name matches
            ``theme_name`` inside ``backgrounds_dir``.

        Returns
        -------
        Path | None
            Absolute path to the chosen image, or ``None`` if the pool
            is empty.
        """
        pool = self._build_pool(self.backgrounds_dir, self.IMAGE_EXT, theme_name)
        chosen = self._pick(pool, self._last_background)
        if chosen:
            self._last_background = chosen
            logger.info("AssetManager: background → %s", chosen.name)
        else:
            logger.warning(
                "AssetManager: no background images found in %s (theme=%r).",
                self.backgrounds_dir,
                theme_name,
            )
        return chosen

    def get_music(self, theme_name: Optional[str] = None) -> Optional[Path]:
        """
        Return a random music track, preferring theme-specific assets.

        Parameters
        ----------
        theme_name : str | None
            Optional theme name for sub-directory routing.

        Returns
        -------
        Path | None
            Absolute path to the chosen audio file, or ``None`` if empty.
        """
        pool = self._build_pool(self.music_dir, self.AUDIO_EXT, theme_name)
        chosen = self._pick(pool, self._last_music)
        if chosen:
            self._last_music = chosen
            logger.info("AssetManager: music → %s", chosen.name)
        else:
            logger.warning(
                "AssetManager: no music files found in %s (theme=%r).",
                self.music_dir,
                theme_name,
            )
        return chosen

    def refresh(self) -> None:
        """Reset the anti-repetition cache so any asset can be returned next."""
        self._last_background = None
        self._last_music = None
        logger.debug("AssetManager: anti-repetition cache cleared.")

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_pool(
        self,
        root: Path,
        extensions: frozenset[str],
        theme_name: Optional[str],
    ) -> list[Path]:
        """
        Collect eligible files from *root*, with optional theme sub-directory.

        Search order:
          1. ``root / theme_name /`` (theme-specific, if folder exists)
          2. ``root /`` (flat fallback pool)

        Parameters
        ----------
        root : Path
            Base directory to search.
        extensions : frozenset[str]
            Allowed file extensions (lowercase, leading dot).
        theme_name : str | None
            Theme label used to look up a sub-directory.

        Returns
        -------
        list[Path]
            All matching file paths (may be empty).
        """
        candidates: list[Path] = []

        # 1. Theme-specific sub-directory
        if theme_name:
            theme_dir = root / theme_name
            if theme_dir.is_dir():
                candidates = self._list_files(theme_dir, extensions)
                if candidates:
                    logger.debug(
                        "AssetManager: found %d files in theme sub-dir %s",
                        len(candidates),
                        theme_dir,
                    )
                    return candidates

        # 2. Flat root pool (all files, excluding sub-directories)
        candidates = self._list_files(root, extensions)
        logger.debug(
            "AssetManager: flat pool → %d files in %s", len(candidates), root
        )
        return candidates

    def _pick(
        self,
        pool: list[Path],
        last: Optional[Path],
    ) -> Optional[Path]:
        """
        Choose a random file from *pool* that differs from *last*.

        If the pool has only one item it is returned even if it matches
        *last* (no alternative exists).

        Parameters
        ----------
        pool : list[Path]
            Available files.
        last : Path | None
            Previously selected file to avoid.

        Returns
        -------
        Path | None
        """
        if not pool:
            return None

        # Filter out the previous pick when more than one option exists
        filtered = [p for p in pool if p != last] if last and len(pool) > 1 else pool
        return self._rng.choice(filtered)

    @staticmethod
    def _list_files(directory: Path, extensions: frozenset[str]) -> list[Path]:
        """Return all files directly inside *directory* with matching extensions."""
        return [
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]

    def _ensure_dirs(self) -> None:
        """Create asset directories if they are missing (warn, don't crash)."""
        for path in (self.backgrounds_dir, self.music_dir):
            if not path.exists():
                logger.warning(
                    "AssetManager: directory %s missing — creating it.", path
                )
                path.mkdir(parents=True, exist_ok=True)
