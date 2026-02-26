"""
core/HashtagEngine.py
─────────────────────
Builds a curated, ranked hashtag block for each MoodLoop AI video.

Strategy (layered, highest relevance first):
  1. Theme-specific core tags          — always included
  2. Trend-boosted tags                — added when a topic matches
  3. Broad reach / evergreen tags      — padded up to *max_tags*
  4. Platform-specific tags            — appended based on *platform*

The engine deduplicates across all layers, preserves insertion order,
and enforces a hard cap (*max_tags*) to avoid spam-filter penalties.

No external API calls are made — this module is fully offline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────── #

class HashtagPlatform(Enum):
    """Target platform — affects the final platform-specific tags."""
    INSTAGRAM = auto()
    YOUTUBE = auto()
    TIKTOK = auto()
    GENERIC = auto()


# ── Hashtag dictionaries ──────────────────────────────────────────── #

# Core tags per theme (always included first)
_THEME_TAGS: dict[str, list[str]] = {
    "dark_aesthetic": [
        "#DarkAesthetic", "#DarkVibes", "#AestheticQuotes",
        "#DarkQuotes", "#MidnightMood", "#GothAesthetic",
    ],
    "genz_existential": [
        "#GenZ", "#ExistentialCrisis", "#BigThoughts",
        "#NihilismVibes", "#PhilosophyTok", "#DeepThoughts",
    ],
    "late_night_thoughts": [
        "#LateNightThoughts", "#3AM", "#Overthinking",
        "#CantSleep", "#NightOwl", "#InsomniaQuotes",
    ],
    "lofi_nostalgia": [
        "#LoFi", "#Nostalgia", "#VintageAesthetic",
        "#RetroVibes", "#LoFiHipHop", "#ChillBeats",
    ],
    "sad_banger": [
        "#SadQuotes", "#Heartbreak", "#Emotional",
        "#SadButTrue", "#FeelYourFeelings", "#CryingToMusic",
    ],
    "motivational_chaos": [
        "#Motivation", "#HardWork", "#Grindset",
        "#MainCharacterEnergy", "#UnlockYourPotential", "#NoExcuses",
    ],
    "chill_vibes": [
        "#ChillVibes", "#PeaceWithin", "#SoftLife",
        "#SlowLiving", "#AestheticVibes", "#CalmMind",
    ],
}

# Tags associated with trending keywords; engine checks overlap with
# the live trending topics list passed in at runtime
_TREND_TAG_MAP: dict[str, list[str]] = {
    "dark": ["#DarkCore", "#DarkEdit"],
    "aesthetic": ["#AestheticEdits", "#AestheticContent"],
    "gen z": ["#GenZHumor", "#ZillennialVibes"],
    "lofi": ["#LoFiBeats", "#StudyWithMe"],
    "motivational": ["#DailyMotivation", "#YouGotThis"],
    "sad": ["#SadTok", "#SadCore"],
    "chill": ["#ChillHop", "#RelaxVibes"],
    "night": ["#NightMode", "#NightPhotography"],
    "vintage": ["#VintageMood", "#FilmPhotography"],
    "existential": ["#PhilosophyMemes", "#ExistentialMemes"],
}

# Broad reach / evergreen tags used as padding
_EVERGREEN_TAGS: list[str] = [
    "#Shorts", "#ShortVideo", "#ViralQuotes",
    "#QuoteOfTheDay", "#MoodBoard", "#Mindset",
    "#PositiveVibes", "#Relatable", "#FYP",
    "#ForYou", "#ExplorePage", "#ContentCreator",
    "#VideoOfTheDay", "#AestheticContent", "#Viral",
]

# Platform-specific tags appended at the very end
_PLATFORM_TAGS: dict[HashtagPlatform, list[str]] = {
    HashtagPlatform.INSTAGRAM: ["#InstagramReels", "#Reels", "#InstaQuotes"],
    HashtagPlatform.YOUTUBE: ["#YouTubeShorts", "#Shorts", "#YTShorts"],
    HashtagPlatform.TIKTOK: ["#TikTok", "#FYP", "#ForYouPage"],
    HashtagPlatform.GENERIC: [],
}


# ------------------------------------------------------------------ #
#  HashtagResult                                                       #
# ------------------------------------------------------------------ #

@dataclass
class HashtagResult:
    """
    Output of a single :meth:`HashtagEngine.build` call.

    Attributes
    ----------
    tags : list[str]
        Ordered, deduplicated hashtag list.
    block : str
        All tags joined by single spaces — ready for appending to a caption.
    count : int
        Total number of tags.
    trend_matched : list[str]
        Tags that were included because of a trending topic match.
    """

    tags: list[str]
    trend_matched: list[str]

    @property
    def block(self) -> str:
        """Space-separated hashtag string."""
        return " ".join(self.tags)

    @property
    def count(self) -> int:
        return len(self.tags)

    def __str__(self) -> str:
        return self.block


# ------------------------------------------------------------------ #
#  HashtagEngine                                                       #
# ------------------------------------------------------------------ #

class HashtagEngine:
    """
    Builds a curated, layered hashtag block for short-form video captions.

    Parameters
    ----------
    platform : HashtagPlatform
        Target platform (affects final platform-specific tags).
    max_tags : int
        Hard cap on the total number of hashtags returned (default: 25).
    include_evergreen : bool
        Whether evergreen reach-tags are used as padding (default: True).
    """

    def __init__(
        self,
        platform: HashtagPlatform = HashtagPlatform.INSTAGRAM,
        max_tags: int = 25,
        include_evergreen: bool = True,
    ) -> None:
        if not 1 <= max_tags <= 40:
            raise ValueError("max_tags must be between 1 and 40.")
        self.platform = platform
        self.max_tags = max_tags
        self.include_evergreen = include_evergreen
        logger.debug(
            "HashtagEngine ready (platform=%s, max=%d)", platform.name, max_tags
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def build(
        self,
        theme_name: str,
        trending_topics: Optional[list[str]] = None,
        extra_tags: Optional[list[str]] = None,
    ) -> HashtagResult:
        """
        Construct a ranked hashtag block for one video run.

        Parameters
        ----------
        theme_name : str
            Theme identifier used to select core tags.
        trending_topics : list[str] | None
            Live topic strings from :class:`~core.TrendAnalyzer.TrendAnalyzer`.
            Used to perform keyword-overlap matching for trend-boosted tags.
        extra_tags : list[str] | None
            Caller-supplied additional tags (prepended after core tags;
            normalised automatically).

        Returns
        -------
        HashtagResult
        """
        trending_topics = trending_topics or []
        extra_tags = extra_tags or []
        seen: set[str] = set()
        result: list[str] = []
        trend_matched: list[str] = []

        def _add(tag: str) -> bool:
            """Normalise and add a tag; return True if it was new."""
            normalised = self._normalise(tag)
            if not normalised or normalised.lower() in seen:
                return False
            seen.add(normalised.lower())
            result.append(normalised)
            return True

        # ── Layer 1: theme-specific core tags ───────────────────────── #
        for tag in _THEME_TAGS.get(theme_name, []):
            _add(tag)

        # ── Layer 2: caller-supplied extras ─────────────────────────── #
        for tag in extra_tags:
            _add(tag)

        # ── Layer 3: trend-boosted tags ──────────────────────────────── #
        normalised_trends = {t.lower() for t in trending_topics}
        for keyword, tags in _TREND_TAG_MAP.items():
            if any(keyword in trend for trend in normalised_trends):
                for tag in tags:
                    if _add(tag):
                        trend_matched.append(tag)

        # ── Layer 4: evergreen padding ───────────────────────────────── #
        if self.include_evergreen:
            for tag in _EVERGREEN_TAGS:
                if len(result) >= self.max_tags:
                    break
                _add(tag)

        # ── Layer 5: platform-specific tags ──────────────────────────── #
        for tag in _PLATFORM_TAGS.get(self.platform, []):
            if len(result) >= self.max_tags:
                break
            _add(tag)

        # Enforce hard cap
        final = result[: self.max_tags]

        logger.info(
            "HashtagEngine: %d tags built (%d trend-matched) for theme=%r.",
            len(final),
            len(trend_matched),
            theme_name,
        )
        return HashtagResult(tags=final, trend_matched=trend_matched)

    def build_block(
        self,
        theme_name: str,
        trending_topics: Optional[list[str]] = None,
        extra_tags: Optional[list[str]] = None,
    ) -> str:
        """
        Convenience wrapper — returns only the space-separated hashtag string.

        Parameters
        ----------
        theme_name : str
        trending_topics : list[str] | None
        extra_tags : list[str] | None

        Returns
        -------
        str
            e.g. ``"#DarkAesthetic #MidnightMood #FYP …"``
        """
        return self.build(theme_name, trending_topics, extra_tags).block

    # ------------------------------------------------------------------ #
    #  Static helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise(tag: str) -> str:
        """
        Ensure *tag* starts with ``#``, strip whitespace, remove illegal chars.

        Parameters
        ----------
        tag : str
            Raw tag (with or without ``#``).

        Returns
        -------
        str
            Normalised hashtag string, or empty string if invalid.
        """
        tag = tag.strip()
        if not tag:
            return ""
        if not tag.startswith("#"):
            tag = "#" + tag
        # Keep only alphanumeric and underscore after the '#'
        body = re.sub(r"[^\w]", "", tag[1:])
        if not body:
            return ""
        return "#" + body

    @staticmethod
    def from_string(raw: str, separator: str = " ") -> list[str]:
        """
        Parse a raw hashtag string (e.g. from config or a previous run)
        back into a list of normalised tags.

        Parameters
        ----------
        raw : str
            Space / newline / comma separated hashtag strings.
        separator : str
            Token separator (default: space).

        Returns
        -------
        list[str]
            Normalised hashtag list.
        """
        tokens = re.split(r"[\s,]+", raw.strip())
        return [HashtagEngine._normalise(t) for t in tokens if t]
