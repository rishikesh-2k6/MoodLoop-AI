"""
core/LLMEngine.py
─────────────────
Thin, strongly-typed client around the Ollama HTTP API (llama3).
Provides dedicated methods for every content generation step:
  • dark aesthetic Gen Z quote
  • video title
  • caption / hashtag block

All methods enforce JSON-free plain-text responses and apply
prompt-engineering best-practices for short-form vertical video.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Data models                                                         #
# ------------------------------------------------------------------ #

@dataclass
class GeneratedContent:
    """
    Holds all LLM-generated text for one video run.

    Attributes
    ----------
    quote : str
        Dark aesthetic Gen Z quote (1–2 sentences).
    title : str
        Short, punchy video title (≤ 60 characters recommended).
    caption : str
        Social-media caption including hashtag block.
    theme_name : str
        Identifier of the theme used for generation.
    model : str
        Ollama model tag that produced this content.
    """

    quote: str
    title: str
    caption: str
    theme_name: str
    model: str


# ------------------------------------------------------------------ #
#  LLMEngine                                                           #
# ------------------------------------------------------------------ #

class LLMEngine:
    """
    Communicates with a locally-running Ollama instance via its HTTP API
    and generates all text content needed for one MoodLoop AI video.

    Parameters
    ----------
    base_url : str
        Root URL of the Ollama API (default ``"http://localhost:11434"``).
    model : str
        Ollama model tag to use (default ``"llama3"``).
    temperature : float
        Sampling temperature forwarded to the model (default ``0.85``).
    timeout : int
        HTTP request timeout in seconds (default ``60``).
    max_retries : int
        Number of retry attempts on transient HTTP failures (default ``3``).
    retry_delay : float
        Base delay in seconds between retries (default ``2.0``).
    """

    _GENERATE_ENDPOINT = "/api/generate"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        temperature: float = 0.85,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._session = requests.Session()
        logger.debug(
            "LLMEngine initialised (model=%s, url=%s, temp=%.2f)",
            model,
            base_url,
            temperature,
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate_all(self, theme_name: str, mood: str) -> GeneratedContent:
        """
        Generate quote, title, and caption for the given theme in one call.

        Parameters
        ----------
        theme_name : str
            Human-readable theme display name (e.g. ``"Dark Aesthetic"``).
        mood : str
            Mood descriptor from the selected ``Theme`` object.

        Returns
        -------
        GeneratedContent
            Fully populated content object ready for the pipeline.

        Raises
        ------
        RuntimeError
            If any individual generation step fails after all retries.
        """
        logger.info("LLMEngine: generating content for theme=%r mood=%r", theme_name, mood)

        quote = self.generate_quote(theme_name, mood)
        title = self.generate_title(theme_name, quote)
        caption = self.generate_caption(theme_name, quote, title)

        content = GeneratedContent(
            quote=quote,
            title=title,
            caption=caption,
            theme_name=theme_name,
            model=self.model,
        )
        logger.info("LLMEngine: content generation complete.")
        return content

    def generate_quote(self, theme_name: str, mood: str) -> str:
        """
        Generate a single dark aesthetic Gen Z quote.

        Parameters
        ----------
        theme_name : str
            Theme display name for prompt context.
        mood : str
            Emotional tone (e.g. ``"mysterious and introspective"``).

        Returns
        -------
        str
            A 1–2 sentence quote stripped of surrounding whitespace.
        """
        prompt = (
            f"You are a dark aesthetic Gen Z poet who writes short, viral quotes for "
            f"short-form videos. The theme is '{theme_name}' and the mood is '{mood}'.\n\n"
            "Write exactly ONE quote that:\n"
            "- Is 1 to 2 sentences long\n"
            "- Has a dark, poetic, introspective or existential tone\n"
            "- Sounds natural to Gen Z (no cringe, no corporate language)\n"
            "- Can be overlaid on a video without any explanation\n"
            "- Does NOT include quotation marks, hashtags, or emojis\n\n"
            "Output ONLY the quote text. Nothing else."
        )
        return self._generate(prompt, label="quote")

    def generate_title(self, theme_name: str, quote: str) -> str:
        """
        Generate a short, engaging video title based on the theme and quote.

        Parameters
        ----------
        theme_name : str
            Theme display name.
        quote : str
            The previously generated quote (provides context).

        Returns
        -------
        str
            A concise title (ideally ≤ 60 characters).
        """
        prompt = (
            f"You are a viral short-form content strategist specialising in YouTube Shorts "
            f"and Instagram Reels.\n\n"
            f"Theme: {theme_name}\n"
            f"Quote: {quote}\n\n"
            "Write a single punchy video TITLE that:\n"
            "- Is under 60 characters\n"
            "- Sparks curiosity or emotional resonance\n"
            "- Suits the dark-aesthetic Gen Z niche\n"
            "- Does NOT include hashtags or emojis in the title itself\n\n"
            "Output ONLY the title text. Nothing else."
        )
        return self._generate(prompt, label="title")

    def generate_caption(
        self, theme_name: str, quote: str, title: str
    ) -> str:
        """
        Generate a social-media caption with hashtag block.

        Parameters
        ----------
        theme_name : str
            Theme display name.
        quote : str
            The generated quote text.
        title : str
            The generated video title.

        Returns
        -------
        str
            A multi-line caption string including up to 15 relevant hashtags.
        """
        prompt = (
            f"You are a social-media copywriter who specialises in Gen Z short-form content.\n\n"
            f"Theme: {theme_name}\n"
            f"Title: {title}\n"
            f"Quote: {quote}\n\n"
            "Write an Instagram / YouTube Shorts CAPTION that:\n"
            "- Opens with one engaging hook line (no emojis yet)\n"
            "- Has 2–3 lines of body copy that deepen the mood\n"
            "- Ends with up to 15 relevant hashtags on a new line\n"
            "- Uses a dark aesthetic, Gen Z tone throughout\n"
            "- May use 1–3 emojis sparingly for emphasis\n\n"
            "Output ONLY the caption text with hashtags. Nothing else."
        )
        return self._generate(prompt, label="caption")

    # ------------------------------------------------------------------ #
    #  Core HTTP layer                                                     #
    # ------------------------------------------------------------------ #

    def _generate(self, prompt: str, label: str = "content") -> str:
        """
        Send a generation request to Ollama and return the response text.

        Parameters
        ----------
        prompt : str
            Full prompt string to send to the model.
        label : str
            Human-readable label used only in log messages.

        Returns
        -------
        str
            Stripped response text from the model.

        Raises
        ------
        RuntimeError
            Raised after exhausting all retry attempts.
        """
        url = f"{self.base_url}{self._GENERATE_ENDPOINT}"
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 256,
            },
        }

        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "LLM %s request (attempt %d/%d) → %s",
                    label,
                    attempt,
                    self.max_retries,
                    url,
                )
                response = self._session.post(
                    url, json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                text: str = data.get("response", "").strip()

                if not text:
                    raise ValueError(
                        f"Ollama returned an empty 'response' field for label='{label}'"
                    )

                logger.debug("LLM %s OK — %d chars", label, len(text))
                return text

            except requests.exceptions.ConnectionError as exc:
                logger.error(
                    "Ollama connection error (attempt %d/%d): %s — "
                    "is 'ollama serve' running?",
                    attempt,
                    self.max_retries,
                    exc,
                )
                last_error = exc

            except requests.exceptions.Timeout as exc:
                logger.error(
                    "Ollama request timed out (attempt %d/%d, timeout=%ds).",
                    attempt,
                    self.max_retries,
                    self.timeout,
                )
                last_error = exc

            except requests.exceptions.HTTPError as exc:
                logger.error(
                    "Ollama HTTP error %s (attempt %d/%d): %s",
                    exc.response.status_code if exc.response else "?",
                    attempt,
                    self.max_retries,
                    exc,
                )
                last_error = exc

            except (KeyError, ValueError) as exc:
                logger.error("LLM response parsing error: %s", exc)
                last_error = exc

            if attempt < self.max_retries:
                sleep_time = self.retry_delay * (2 ** (attempt - 1))
                logger.debug("Retrying in %.2f s…", sleep_time)
                time.sleep(sleep_time)

        raise RuntimeError(
            f"LLMEngine failed to generate '{label}' after "
            f"{self.max_retries} attempts. Last error: {last_error}"
        )

    def health_check(self) -> bool:
        """
        Ping the Ollama server to verify it is reachable.

        Returns
        -------
        bool
            ``True`` if the server responds with HTTP 200, ``False`` otherwise.
        """
        try:
            resp = self._session.get(self.base_url, timeout=5)
            ok = resp.status_code == 200
            if ok:
                logger.info("Ollama health check: OK (%s)", self.base_url)
            else:
                logger.warning(
                    "Ollama health check returned HTTP %d", resp.status_code
                )
            return ok
        except requests.exceptions.RequestException as exc:
            logger.error("Ollama health check failed: %s", exc)
            return False
