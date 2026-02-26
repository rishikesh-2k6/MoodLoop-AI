"""
core/CaptionEngine.py
─────────────────────
Generates polished social-media captions for MoodLoop AI videos.

Responsibilities:
  • Build a structured caption from the generated quote, title, and theme.
  • Support two modes:
      - LLM mode  — delegates to an :class:`~core.LLMEngine.LLMEngine`
                    instance for creative, context-aware captions.
      - Template mode — offline fallback using configurable string
                        templates; no network calls required.
  • Post-process output: strip excess whitespace, enforce character limits,
    apply platform-specific formatting (YouTube Shorts vs Instagram).
  • Optionally append a hashtag block produced by
    :class:`~core.HashtagEngine.HashtagEngine`.

This class never calls FFmpeg or Pillow — it is purely text processing.
"""

from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.LLMEngine import LLMEngine

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────── #

class Platform(Enum):
    """Supported social publishing platforms with their caption limits."""
    YOUTUBE_SHORTS = auto()    # 5 000-char hard limit; first line = title
    INSTAGRAM_REELS = auto()  # 2 200-char soft limit; hashtags at end
    GENERIC = auto()           # No platform-specific formatting


class CaptionMode(Enum):
    """Selects whether captions are AI-generated or built from templates."""
    LLM = auto()
    TEMPLATE = auto()


# ── Data models ───────────────────────────────────────────────────── #

@dataclass
class CaptionResult:
    """
    The finished caption for one pipeline run.

    Attributes
    ----------
    body : str
        Caption body text (no hashtags).
    hashtag_block : str
        Raw hashtag string (may be empty).
    full_caption : str
        Concatenation of ``body`` + newline + ``hashtag_block``.
    platform : Platform
        Target platform this caption was formatted for.
    char_count : int
        Total character count of ``full_caption``.
    truncated : bool
        ``True`` if the caption was trimmed to fit the platform limit.
    """

    body: str
    hashtag_block: str = ""
    platform: Platform = Platform.GENERIC
    truncated: bool = False

    @property
    def full_caption(self) -> str:
        """Body + hashtag block joined by a double newline."""
        parts = [self.body.strip()]
        if self.hashtag_block.strip():
            parts.append(self.hashtag_block.strip())
        return "\n\n".join(parts)

    @property
    def char_count(self) -> int:
        return len(self.full_caption)


# ── Platform limits ───────────────────────────────────────────────── #

_PLATFORM_LIMITS: dict[Platform, int] = {
    Platform.YOUTUBE_SHORTS: 5_000,
    Platform.INSTAGRAM_REELS: 2_200,
    Platform.GENERIC: 10_000,
}

# ── Built-in templates ────────────────────────────────────────────── #

_TEMPLATES: dict[str, str] = {
    "dark_aesthetic": (
        "some things live rent-free in your mind at 3am.\n\n"
        '"{quote}"\n\n'
        "this one is for the ones who feel everything too deeply. 🖤"
    ),
    "genz_existential": (
        "okay but why does this hit so different—\n\n"
        '"{quote}"\n\n'
        "your brain is a philosopher and it doesn't clock out. 💭"
    ),
    "late_night_thoughts": (
        "another late night, another spiral.\n\n"
        '"{quote}"\n\n'
        "if you needed a sign to feel seen — here it is. 🌙"
    ),
    "lofi_nostalgia": (
        "somewhere between yesterday and nowhere.\n\n"
        '"{quote}"\n\n'
        "lo-fi hits different when you're chasing a feeling. 🎵"
    ),
    "sad_banger": (
        "this one's going to hurt a little. sorry not sorry.\n\n"
        '"{quote}"\n\n'
        "some songs are just therapy with a beat drop. 💔"
    ),
    "motivational_chaos": (
        "your villain era? actually your main character era.\n\n"
        '"{quote}"\n\n'
        "the chaos is the plan. trust it. ⚡"
    ),
    "chill_vibes": (
        "slow down. breathe. you don't have to be everywhere at once.\n\n"
        '"{quote}"\n\n'
        "soft life is a whole aesthetic and we're here for it. 🍃"
    ),
    "default": (
        '"{quote}"\n\n'
        "save this if you needed to hear it today. 💫"
    ),
}


# ------------------------------------------------------------------ #
#  CaptionEngine                                                       #
# ------------------------------------------------------------------ #

class CaptionEngine:
    """
    Generates, formats, and optionally truncates social-media captions.

    Parameters
    ----------
    mode : CaptionMode
        ``CaptionMode.LLM`` delegates to an :class:`LLMEngine`;
        ``CaptionMode.TEMPLATE`` uses built-in templates (offline).
    llm_engine : LLMEngine | None
        Required when *mode* is ``CaptionMode.LLM``.
    platform : Platform
        Target platform — controls character-limit enforcement.
    line_width : int
        Maximum characters per line for body text wrapping (default 72).
    """

    def __init__(
        self,
        mode: CaptionMode = CaptionMode.TEMPLATE,
        llm_engine: Optional["LLMEngine"] = None,
        platform: Platform = Platform.INSTAGRAM_REELS,
        line_width: int = 72,
    ) -> None:
        if mode is CaptionMode.LLM and llm_engine is None:
            raise ValueError(
                "CaptionEngine: llm_engine must be provided when mode=CaptionMode.LLM."
            )
        self.mode = mode
        self.llm_engine = llm_engine
        self.platform = platform
        self.line_width = line_width
        logger.debug(
            "CaptionEngine ready (mode=%s, platform=%s)", mode.name, platform.name
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        theme_name: str,
        quote: str,
        title: str,
        mood: str,
        hashtag_block: str = "",
    ) -> CaptionResult:
        """
        Generate a complete caption for one pipeline run.

        Parameters
        ----------
        theme_name : str
            Theme identifier (e.g. ``"dark_aesthetic"``).
        quote : str
            LLM-generated quote text.
        title : str
            LLM-generated video title.
        mood : str
            Theme mood descriptor.
        hashtag_block : str
            Pre-built hashtag string from :class:`HashtagEngine`.

        Returns
        -------
        CaptionResult
            Fully formatted caption object.
        """
        if self.mode is CaptionMode.LLM:
            body = self._generate_llm(theme_name, quote, title, mood)
        else:
            body = self._generate_template(theme_name, quote, title)

        body = self._post_process(body)
        result = CaptionResult(
            body=body,
            hashtag_block=hashtag_block,
            platform=self.platform,
        )
        result = self._enforce_limit(result)

        logger.info(
            "CaptionEngine: %d chars | truncated=%s | platform=%s",
            result.char_count,
            result.truncated,
            result.platform.name,
        )
        return result

    # ------------------------------------------------------------------ #
    #  Generation strategies                                               #
    # ------------------------------------------------------------------ #

    def _generate_llm(
        self, theme_name: str, quote: str, title: str, mood: str
    ) -> str:
        """
        Delegate caption generation to the LLM engine.

        Parameters
        ----------
        theme_name : str
        quote : str
        title : str
        mood : str

        Returns
        -------
        str
            Raw LLM caption body (before post-processing).
        """
        assert self.llm_engine is not None
        logger.debug("CaptionEngine: calling LLM for caption.")
        try:
            return self.llm_engine.generate_caption(
                theme_name=theme_name, quote=quote, title=title
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CaptionEngine: LLM caption failed (%s) — falling back to template.", exc
            )
            return self._generate_template(theme_name, quote, title)

    def _generate_template(
        self, theme_name: str, quote: str, title: str
    ) -> str:
        """
        Build a caption from the built-in template catalogue.

        Parameters
        ----------
        theme_name : str
            Used to look up the matching template.
        quote : str
            Substituted into the ``{quote}`` placeholder.
        title : str
            Not currently used in the default templates (reserved).

        Returns
        -------
        str
            Rendered caption body.
        """
        template = _TEMPLATES.get(theme_name, _TEMPLATES["default"])
        body = template.format(quote=quote, title=title)
        logger.debug(
            "CaptionEngine: template used for theme=%r.", theme_name
        )
        return body

    # ------------------------------------------------------------------ #
    #  Post-processing                                                     #
    # ------------------------------------------------------------------ #

    def _post_process(self, text: str) -> str:
        """
        Clean generated text: strip leading/trailing whitespace, collapse
        more-than-two consecutive blank lines, and wrap long lines.

        Parameters
        ----------
        text : str
            Raw generated caption.

        Returns
        -------
        str
            Cleaned caption string.
        """
        # Collapse 3+ consecutive newlines → 2
        text = re.sub(r"\n{3,}", "\n\n", text.strip())
        # Wrap individual long lines while preserving paragraph breaks
        paragraphs = text.split("\n\n")
        wrapped = []
        for para in paragraphs:
            # Don't wrap hashtag lines
            if para.strip().startswith("#"):
                wrapped.append(para)
            else:
                wrapped.append(
                    textwrap.fill(para, width=self.line_width, break_long_words=False)
                )
        return "\n\n".join(wrapped)

    def _enforce_limit(self, result: CaptionResult) -> CaptionResult:
        """
        Truncate *result* if it exceeds the platform character limit.

        Truncation always cuts the body text; the hashtag block is kept
        intact because platform algorithms use it for distribution.

        Parameters
        ----------
        result : CaptionResult

        Returns
        -------
        CaptionResult
            Possibly truncated result (new object when truncation occurs).
        """
        limit = _PLATFORM_LIMITS[self.platform]
        if result.char_count <= limit:
            return result

        # Reserve space for hashtag block + separator
        reserved = len(result.hashtag_block) + 3  # "\n\n" + safety
        body_limit = limit - reserved
        truncated_body = result.body[:body_limit].rstrip() + "…"

        logger.warning(
            "CaptionEngine: caption truncated from %d → %d chars to fit %s limit.",
            result.char_count,
            limit,
            self.platform.name,
        )
        return CaptionResult(
            body=truncated_body,
            hashtag_block=result.hashtag_block,
            platform=result.platform,
            truncated=True,
        )

    # ------------------------------------------------------------------ #
    #  Factory helpers                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_template(
        cls, platform: Platform = Platform.INSTAGRAM_REELS
    ) -> "CaptionEngine":
        """
        Create a template-mode engine — no LLM required.

        Parameters
        ----------
        platform : Platform

        Returns
        -------
        CaptionEngine
        """
        return cls(mode=CaptionMode.TEMPLATE, platform=platform)

    @classmethod
    def from_llm(
        cls,
        llm_engine: "LLMEngine",
        platform: Platform = Platform.INSTAGRAM_REELS,
    ) -> "CaptionEngine":
        """
        Create an LLM-mode engine.

        Parameters
        ----------
        llm_engine : LLMEngine
        platform : Platform

        Returns
        -------
        CaptionEngine
        """
        return cls(mode=CaptionMode.LLM, llm_engine=llm_engine, platform=platform)
