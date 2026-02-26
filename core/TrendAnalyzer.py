"""
core/TrendAnalyzer.py
─────────────────────
Fetches real-time Google Trends data via pytrends and exposes a ranked
list of trending topics that the rest of the pipeline can use to bias
theme selection.
"""

from __future__ import annotations

import time
import random
import logging
from typing import Optional

from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    Fetches trending topics from Google Trends and returns a ranked list
    that can be used to bias theme selection in the content pipeline.

    Attributes
    ----------
    geo : str
        ISO 3166-1 alpha-2 country code used for trend queries (default "US").
    language : str
        Language/locale string forwarded to pytrends (default "en-US").
    max_retries : int
        Number of times to retry a failed Trends request before giving up.
    retry_delay : float
        Seconds to wait between retries (uses simple exponential back-off).
    _pytrends : TrendReq
        Underlying pytrends session object.
    """

    # Topics we care about for short-form vertical content
    DEFAULT_KEYWORDS: list[str] = [
        "dark aesthetic",
        "Gen Z quotes",
        "motivational",
        "aesthetic vibes",
        "lofi mood",
        "night thoughts",
        "existential",
        "chill vibes",
        "sad lyrics",
        "vintage aesthetic",
    ]

    def __init__(
        self,
        geo: str = "US",
        language: str = "en-US",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        """
        Initialise TrendAnalyzer with geo/language settings.

        Parameters
        ----------
        geo : str
            Country code for trend data (e.g. "US", "IN").
        language : str
            Locale string (e.g. "en-US").
        max_retries : int
            Maximum retry attempts on transient failures.
        retry_delay : float
            Base delay in seconds between retries.
        """
        self.geo = geo
        self.language = language
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._pytrends = TrendReq(hl=self.language, tz=360, timeout=(10, 25))
        logger.debug("TrendAnalyzer initialised (geo=%s, lang=%s)", geo, language)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_trending_topics(
        self, keywords: Optional[list[str]] = None, timeframe: str = "now 1-d"
    ) -> list[str]:
        """
        Fetch interest-over-time for the given keywords and return them
        sorted by total search interest (highest first).

        Parameters
        ----------
        keywords : list[str] | None
            Keywords to evaluate. Defaults to ``DEFAULT_KEYWORDS``.
        timeframe : str
            pytrends timeframe string, e.g. ``"now 1-d"``, ``"now 7-d"``.

        Returns
        -------
        list[str]
            Keywords sorted by descending trend score.
            Empty list on failure.
        """
        keywords = keywords or self.DEFAULT_KEYWORDS
        # pytrends accepts max 5 keywords per request
        batches = self._chunk(keywords, size=5)
        scores: dict[str, float] = {}

        for batch in batches:
            batch_scores = self._fetch_batch(batch, timeframe)
            scores.update(batch_scores)

        if not scores:
            logger.warning("TrendAnalyzer: no trend data retrieved — returning empty list.")
            return []

        ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
        logger.info("Trending topics (top 5): %s", ranked[:5])
        return ranked

    def get_top_topic(
        self, keywords: Optional[list[str]] = None, timeframe: str = "now 1-d"
    ) -> Optional[str]:
        """
        Convenience method — returns only the single most-trending keyword.

        Parameters
        ----------
        keywords : list[str] | None
            Pool of keywords to evaluate.
        timeframe : str
            pytrends timeframe string.

        Returns
        -------
        str | None
            The most-trending keyword, or ``None`` on failure.
        """
        ranked = self.get_trending_topics(keywords, timeframe)
        return ranked[0] if ranked else None

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _fetch_batch(
        self, keywords: list[str], timeframe: str
    ) -> dict[str, float]:
        """
        Query pytrends for a single batch (≤ 5 keywords) with retry logic.

        Parameters
        ----------
        keywords : list[str]
            Batch of up to 5 keywords.
        timeframe : str
            pytrends timeframe string.

        Returns
        -------
        dict[str, float]
            Keyword → average interest score mapping.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                self._pytrends.build_payload(
                    keywords, cat=0, timeframe=timeframe, geo=self.geo, gprop=""
                )
                df = self._pytrends.interest_over_time()

                if df.empty:
                    logger.warning("Empty trend DataFrame for keywords: %s", keywords)
                    return {kw: 0.0 for kw in keywords}

                # Drop the 'isPartial' column if present
                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])

                return {kw: float(df[kw].mean()) for kw in keywords if kw in df.columns}

            except ResponseError as exc:
                logger.error(
                    "TrendAnalyzer ResponseError (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "TrendAnalyzer unexpected error (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

            if attempt < self.max_retries:
                sleep_time = self.retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                logger.debug("Retrying in %.2f s…", sleep_time)
                time.sleep(sleep_time)

        # All retries exhausted — return zero scores so pipeline can continue
        return {kw: 0.0 for kw in keywords}

    @staticmethod
    def _chunk(lst: list, size: int) -> list[list]:
        """Split *lst* into sub-lists of at most *size* elements."""
        return [lst[i : i + size] for i in range(0, len(lst), size)]
