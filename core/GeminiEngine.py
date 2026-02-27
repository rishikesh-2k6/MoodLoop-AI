"""
core/GeminiEngine.py
────────────────────
AI engine with a 3-tier fallback chain for all content generation
and Gemini Vision-based asset selection / frame validation.

Priority order (text generation):
  1. Google Gemini  (gemini-1.5-flash)   ← primary
  2. HuggingFace    (free online API)    ← fallback #1
  3. Ollama         (llama3, local PC)   ← last resort

Priority order (vision tasks — image/font validation):
  1. Google Gemini Vision   (multimodal)
  2. Scored heuristic       (filename keyword fallback — no network needed)

Usage:
    engine = GeminiEngine(gemini_api_key="AIza...")
    content = engine.generate_all("Sad Banger", "deeply emotional")
    best_bg  = engine.pick_best_image(quote, mood, [Path(...), ...])
    ok       = engine.validate_frame(frame_path, quote)
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Data models                                                         #
# ------------------------------------------------------------------ #

@dataclass
class GeneratedContent:
    """
    Holds all AI-generated text for one video run.

    Attributes
    ----------
    quote : str
        Short, punchy hook quote (1 sentence max).
    title : str
        Viral video title (≤ 50 chars).
    caption : str
        Social-media caption with hashtags.
    theme_name : str
        Theme identifier used for generation.
    model : str
        Which model/backend actually produced the output.
    """
    quote: str
    title: str
    caption: str
    theme_name: str
    model: str


# ------------------------------------------------------------------ #
#  GeminiEngine                                                        #
# ------------------------------------------------------------------ #

class GeminiEngine:
    """
    Multi-backend AI engine for MoodLoop AI.

    Falls back automatically:
        Gemini → Ollama → HuggingFace (text)
        Gemini Vision → heuristic keyword score (image/font validation)

    Parameters
    ----------
    gemini_api_key : str | None
        Google AI Studio API key.  If ``None``, tier-1 is skipped.
    ollama_url : str
        Ollama base URL (default ``"http://localhost:11434"``).
    ollama_model : str
        Ollama model tag (default ``"llama3"``).
    hf_api_key : str | None
        HuggingFace Inference API token (optional — free tier works
        without a key but with rate limits).
    timeout : int
        HTTP timeout in seconds per request (default ``30``).
    """

    _GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    _HF_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    _OLLAMA_ENDPOINT = "/api/generate"

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama3",
        hf_api_key: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        self.hf_api_key = hf_api_key or os.getenv("HF_API_KEY", "")
        self.timeout = timeout
        self._session = requests.Session()

        # Track which backend is alive to avoid repeated health checks
        self._gemini_ok: Optional[bool] = None
        self._ollama_ok: Optional[bool] = None

        logger.info(
            "GeminiEngine ready — gemini=%s, ollama=%s, hf=%s",
            "✓" if self.gemini_api_key else "✗",
            ollama_url,
            "✓" if self.hf_api_key else "free-tier",
        )

    # ------------------------------------------------------------------ #
    #  Public — text generation                                           #
    # ------------------------------------------------------------------ #

    def generate_all(self, theme_name: str, mood: str) -> GeneratedContent:
        """Generate quote, title, and caption with fallback chain."""
        logger.info("GeminiEngine: generating content — theme=%r mood=%r", theme_name, mood)
        quote = self.generate_quote(theme_name, mood)
        title = self.generate_title(theme_name, quote)
        caption = self.generate_caption(theme_name, quote, title)
        _, backend = self._generate_with_fallback("ping", label="ping", _dry=True)
        logger.info("GeminiEngine: generation complete via backend=%s", backend)
        return GeneratedContent(
            quote=quote, title=title, caption=caption,
            theme_name=theme_name, model=backend,
        )

    def generate_quote(self, theme_name: str, mood: str) -> str:
        """
        Generate one short, punchy hook quote.

        Designed to be immediately understandable — no abstract
        metaphors, no clichés.  One clear, emotionally resonant
        sentence that people stop scrolling for.
        """
        prompt = (
            f"You write viral one-line quotes for short-form videos.\n"
            f"Theme: {theme_name}. Mood: {mood}.\n\n"
            "Write ONE quote that:\n"
            "- Is a SINGLE sentence (max 12 words)\n"
            "- Is instantly understandable — no complex metaphors\n"
            "- Hits an emotion hard (relatable, raw, or surprising)\n"
            "- Sounds like something a real person would say or think\n"
            "- Has no quotation marks, hashtags, or emojis\n\n"
            "Output ONLY the quote. Nothing else."
        )
        text, _ = self._generate_with_fallback(prompt, label="quote")
        # Clean up any stray quotes/newlines
        return text.strip().strip('"').strip("'").splitlines()[0].strip()

    def generate_title(self, theme_name: str, quote: str) -> str:
        """Generate a short, clickable video title (≤ 50 chars)."""
        prompt = (
            f"Create a YouTube Shorts / Instagram Reels title.\n"
            f"Theme: {theme_name}\nQuote: {quote}\n\n"
            "Rules:\n"
            "- Under 50 characters\n"
            "- Creates curiosity or emotional pull\n"
            "- Plain words — no hashtags, no emojis\n\n"
            "Output ONLY the title."
        )
        text, _ = self._generate_with_fallback(prompt, label="title")
        return text.strip().strip('"').splitlines()[0].strip()[:60]

    def generate_caption(self, theme_name: str, quote: str, title: str) -> str:
        """Generate a social media caption with hashtags."""
        prompt = (
            f"Write an Instagram/YouTube Shorts caption.\n"
            f"Theme: {theme_name}\nTitle: {title}\nQuote: {quote}\n\n"
            "Format:\n"
            "- 1 hook line (no emoji)\n"
            "- 2–3 lines deepening the mood\n"
            "- 10–15 relevant hashtags on the last line\n"
            "- Max 2 emojis total\n\n"
            "Output ONLY the caption text."
        )
        text, _ = self._generate_with_fallback(prompt, label="caption")
        return text.strip()

    # ------------------------------------------------------------------ #
    #  Public — vision / asset selection                                  #
    # ------------------------------------------------------------------ #

    def pick_best_image(
        self,
        quote: str,
        mood: str,
        image_paths: list[Path],
    ) -> Path:
        """
        Use Gemini Vision to pick the background image that best suits the
        quote and mood.  Falls back to heuristic filename scoring.

        Parameters
        ----------
        quote : str
            The generated quote text.
        mood : str
            Theme mood descriptor.
        image_paths : list[Path]
            List of available background image files.

        Returns
        -------
        Path
            The best-matching image path.
        """
        if not image_paths:
            raise ValueError("No image paths provided")

        if len(image_paths) == 1:
            return image_paths[0]

        # Try Gemini Vision
        if self.gemini_api_key:
            try:
                result = self._gemini_pick_image(quote, mood, image_paths)
                if result:
                    logger.info("pick_best_image: Gemini selected '%s'", result.name)
                    return result
            except Exception as exc:
                logger.warning("pick_best_image: Gemini Vision failed: %s — using heuristic", exc)

        # Heuristic fallback: score by filename keyword overlap with mood words
        result = self._heuristic_pick(image_paths, mood + " " + quote)
        logger.info("pick_best_image: heuristic selected '%s'", result.name)
        return result

    def pick_best_font(
        self,
        quote: str,
        mood: str,
        font_paths: list[Path],
        theme_font: Optional[Path] = None,
    ) -> Path:
        """
        Ask Gemini which font name from the available list best suits the
        quote text and mood.  Falls back to the theme-preset font.

        Parameters
        ----------
        quote : str
            The generated quote.
        mood : str
            Mood descriptor.
        font_paths : list[Path]
            Available font files.
        theme_font : Path | None
            Pre-selected font from theme config (used as fallback).

        Returns
        -------
        Path
            Best-matching font path.
        """
        if not font_paths:
            return theme_font or _FALLBACK_FONT

        if theme_font and len(font_paths) <= 3:
            # Theme already made a fine choice when options are limited
            return theme_font

        if self.gemini_api_key:
            try:
                result = self._gemini_pick_font(quote, mood, font_paths)
                if result:
                    logger.info("pick_best_font: Gemini selected '%s'", result.name)
                    return result
            except Exception as exc:
                logger.warning("pick_best_font: Gemini failed: %s — using theme font", exc)

        return theme_font or font_paths[0]

    def pick_best_music(
        self,
        mood: str,
        music_paths: list[Path],
        theme_music: Optional[Path] = None,
    ) -> Path:
        """
        Ask Gemini which music track (by filename) best matches the mood.

        Parameters
        ----------
        mood : str
            Theme mood descriptor.
        music_paths : list[Path]
            Available audio files.
        theme_music : Path | None
            Pre-selected track from ThemeSelector (fallback).

        Returns
        -------
        Path
            Best-matching music path.
        """
        if not music_paths:
            return theme_music

        if self.gemini_api_key:
            try:
                result = self._gemini_pick_music(mood, music_paths)
                if result:
                    logger.info("pick_best_music: Gemini selected '%s'", result.name)
                    return result
            except Exception as exc:
                logger.warning("pick_best_music: Gemini failed: %s — using theme music", exc)

        return theme_music or music_paths[0]

    def validate_frame(self, frame_path: Path, quote: str) -> dict:
        """
        Send a rendered quote card image to Gemini Vision and ask whether
        the text is readable and the image suits the quote.

        Parameters
        ----------
        frame_path : Path
            Path to the rendered frame PNG.
        quote : str
            The quote text overlaid on the frame.

        Returns
        -------
        dict
            {
              "ok": bool,          # True = looks good
              "readable": bool,    # Text legibility
              "mood_match": bool,  # Image suits the quote
              "feedback": str,     # Short human-readable comment
              "backend": str,      # Which backend answered
            }
        """
        if not self.gemini_api_key:
            return {"ok": True, "readable": True, "mood_match": True,
                    "feedback": "Skipped (no Gemini key)", "backend": "skipped"}

        try:
            return self._gemini_validate_frame(frame_path, quote)
        except Exception as exc:
            logger.warning("validate_frame: Gemini failed: %s — skipping validation", exc)
            return {"ok": True, "readable": True, "mood_match": True,
                    "feedback": f"Validation skipped: {exc}", "backend": "skipped"}

    def health_check(self) -> dict:
        """Return availability of each backend."""
        results = {}

        # Gemini
        try:
            resp = self._session.post(
                self._GEMINI_URL,
                params={"key": self.gemini_api_key},
                json={"contents": [{"parts": [{"text": "hi"}]}]},
                timeout=10,
            )
            results["gemini"] = resp.status_code == 200
        except Exception:
            results["gemini"] = False

        # Ollama
        try:
            resp = self._session.get(self.ollama_url, timeout=5)
            results["ollama"] = resp.status_code == 200
        except Exception:
            results["ollama"] = False

        # HuggingFace
        try:
            headers = {"Authorization": f"Bearer {self.hf_api_key}"} if self.hf_api_key else {}
            resp = self._session.post(
                self._HF_URL,
                json={"inputs": "hi", "parameters": {"max_new_tokens": 5}},
                headers=headers,
                timeout=10,
            )
            results["huggingface"] = resp.status_code in (200, 503)  # 503 = model loading
        except Exception:
            results["huggingface"] = False

        logger.info("Health check: %s", results)
        return results

    # ------------------------------------------------------------------ #
    #  Private — text generation fallback chain                          #
    # ------------------------------------------------------------------ #

    def _generate_with_fallback(
        self, prompt: str, label: str = "content", _dry: bool = False
    ) -> tuple[str, str]:
        """
        Try Gemini → Ollama → HuggingFace in order.

        Returns (text, backend_name).
        """
        if _dry:
            # Just return which backend would be used (for logging)
            if self.gemini_api_key:
                return ("", "gemini-1.5-flash")
            return ("", "huggingface")

        # ── Tier 1: Gemini ──────────────────────────────────────────── #
        if self.gemini_api_key:
            try:
                text = self._gemini_generate(prompt)
                if text:
                    logger.debug("[%s] Gemini OK", label)
                    return text, "gemini-1.5-flash"
            except Exception as exc:
                logger.warning("[%s] Gemini failed (%s) — trying HuggingFace", label, exc)

        # ── Tier 2: HuggingFace free online API ─────────────────────── #
        try:
            text = self._hf_generate(prompt)
            if text:
                logger.debug("[%s] HuggingFace OK", label)
                return text, "huggingface"
        except Exception as exc:
            logger.warning("[%s] HuggingFace failed (%s) — trying Ollama", label, exc)

        # ── Tier 3: Ollama (local llama) — last resort ───────────────── #
        if self._ollama_alive():
            try:
                text = self._ollama_generate(prompt)
                if text:
                    logger.debug("[%s] Ollama OK", label)
                    return text, f"ollama/{self.ollama_model}"
            except Exception as exc:
                logger.error("[%s] Ollama also failed: %s", label, exc)

        raise RuntimeError(
            f"GeminiEngine: all backends failed for '{label}'. "
            "Check API key, internet connection, and Ollama status."
        )

    def _gemini_generate(self, prompt: str) -> str:
        """Call Gemini 1.5 Flash text API."""
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.9,
                "maxOutputTokens": 300,
                "topP": 0.95,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        }
        resp = self._session.post(
            self._GEMINI_URL,
            params={"key": self.gemini_api_key},
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini returned no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            raise ValueError("Gemini returned empty text")
        return text

    def _ollama_generate(self, prompt: str) -> str:
        """Call local Ollama API."""
        url = f"{self.ollama_url}{self._OLLAMA_ENDPOINT}"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.85, "num_predict": 300},
        }
        resp = self._session.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        if not text:
            raise ValueError("Ollama returned empty response")
        return text

    def _hf_generate(self, prompt: str) -> str:
        """Call HuggingFace Inference API (free tier, no key needed)."""
        headers = {}
        if self.hf_api_key:
            headers["Authorization"] = f"Bearer {self.hf_api_key}"

        # Format as instruction for Mistral
        formatted = f"[INST] {prompt} [/INST]"
        payload = {
            "inputs": formatted,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.85,
                "return_full_text": False,
            },
        }

        for attempt in range(3):
            resp = self._session.post(
                self._HF_URL, json=payload, headers=headers, timeout=60
            )
            if resp.status_code == 503:
                # Model is loading, wait and retry
                wait = int(resp.headers.get("X-Wait-For-Model", 20))
                logger.info("HuggingFace model loading, waiting %ds…", wait)
                time.sleep(min(wait, 30))
                continue
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                text = data[0].get("generated_text", "").strip()
                if text:
                    return text
            break

        raise ValueError("HuggingFace returned no usable text")

    def _ollama_alive(self) -> bool:
        """Check if Ollama is reachable (cached)."""
        if self._ollama_ok is not None:
            return self._ollama_ok
        try:
            resp = self._session.get(self.ollama_url, timeout=4)
            self._ollama_ok = resp.status_code == 200
        except Exception:
            self._ollama_ok = False
        return self._ollama_ok

    # ------------------------------------------------------------------ #
    #  Private — Gemini Vision asset selection                            #
    # ------------------------------------------------------------------ #

    def _gemini_vision_request(self, prompt: str, image_path: Path) -> str:
        """Send a single image + text prompt to Gemini Vision."""
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        img_bytes = image_path.read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode()
        # Detect MIME type
        suffix = image_path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".webp": "image/webp"}
        mime = mime_map.get(suffix, "image/jpeg")

        payload = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": mime, "data": img_b64}},
                    {"text": prompt},
                ]
            }],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 100},
        }
        resp = self._session.post(
            url, params={"key": self.gemini_api_key}, json=payload, timeout=45
        )
        resp.raise_for_status()
        data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()

    def _gemini_pick_image(
        self, quote: str, mood: str, image_paths: list[Path]
    ) -> Optional[Path]:
        """
        Show all images to Gemini (one at a time), score each, pick best.
        Avoids sending giant base64 blobs by asking a yes/no per image.
        """
        scores: list[tuple[int, Path]] = []
        prompt_template = (
            f'Quote: "{quote}"\nMood: {mood}\n\n'
            "Does this background image suit the quote and mood? "
            "Reply with ONLY a number from 1-10 (10=perfect match)."
        )
        for path in image_paths[:6]:  # cap at 6 to stay within rate limits
            try:
                reply = self._gemini_vision_request(prompt_template, path)
                # Extract the first number found in the reply
                nums = [int(c) for c in reply if c.isdigit()]
                score = nums[0] if nums else 5
                scores.append((score, path))
                logger.debug("Image '%s' scored %d by Gemini", path.name, score)
            except Exception as exc:
                logger.debug("Could not score image '%s': %s", path.name, exc)
                scores.append((5, path))  # neutral score

        if not scores:
            return None
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]

    def _gemini_pick_font(
        self, quote: str, mood: str, font_paths: list[Path]
    ) -> Optional[Path]:
        """Ask Gemini to pick the best font by name."""
        # Build a numbered list of font names (no images — just names)
        font_names = [f"{i+1}. {p.stem}" for i, p in enumerate(font_paths[:20])]
        prompt = (
            f'Quote: "{quote}"\nMood: {mood}\n\n'
            "Available fonts:\n" + "\n".join(font_names) + "\n\n"
            "Which font number best suits the quote's mood and readability? "
            "Reply with ONLY the number."
        )
        try:
            reply, _ = self._generate_with_fallback(prompt, label="font_pick")
            nums = [int(c) for c in reply.split() if c.isdigit()]
            if nums:
                idx = max(0, min(nums[0] - 1, len(font_paths) - 1))
                return font_paths[idx]
        except Exception as exc:
            logger.debug("Font pick failed: %s", exc)
        return None

    def _gemini_pick_music(
        self, mood: str, music_paths: list[Path]
    ) -> Optional[Path]:
        """Ask Gemini to pick the best music track by filename."""
        track_names = [f"{i+1}. {p.stem}" for i, p in enumerate(music_paths)]
        prompt = (
            f"Mood: {mood}\n\n"
            "Available music tracks:\n" + "\n".join(track_names) + "\n\n"
            "Which track number best suits this mood? "
            "Reply with ONLY the number."
        )
        try:
            reply, _ = self._generate_with_fallback(prompt, label="music_pick")
            nums = [int(c) for c in reply.split() if c.isdigit()]
            if nums:
                idx = max(0, min(nums[0] - 1, len(music_paths) - 1))
                return music_paths[idx]
        except Exception as exc:
            logger.debug("Music pick failed: %s", exc)
        return None

    def _gemini_validate_frame(self, frame_path: Path, quote: str) -> dict:
        """Send the rendered frame to Gemini Vision for validation."""
        prompt = (
            f'The text on this image should read: "{quote}"\n\n'
            "Evaluate this video frame:\n"
            "1. Is the text clearly readable? (yes/no)\n"
            "2. Does the background image suit the text mood? (yes/no)\n"
            "3. Rate overall quality 1-10\n\n"
            "Reply in this exact format:\n"
            "readable: yes/no\n"
            "mood_match: yes/no\n"
            "score: N\n"
            "feedback: one sentence"
        )
        reply = self._gemini_vision_request(prompt, frame_path)
        lines = {
            k.strip(): v.strip()
            for line in reply.splitlines()
            if ":" in line
            for k, v in [line.split(":", 1)]
        }
        readable = lines.get("readable", "yes").lower().startswith("y")
        mood_match = lines.get("mood_match", "yes").lower().startswith("y")
        feedback = lines.get("feedback", "Looks good")
        ok = readable and mood_match
        logger.info(
            "Frame validation — readable=%s mood_match=%s feedback=%r",
            readable, mood_match, feedback
        )
        return {
            "ok": ok,
            "readable": readable,
            "mood_match": mood_match,
            "feedback": feedback,
            "backend": "gemini-vision",
        }

    # ------------------------------------------------------------------ #
    #  Private — heuristic fallbacks                                      #
    # ------------------------------------------------------------------ #

    def _heuristic_pick(self, paths: list[Path], context: str) -> Path:
        """Score paths by how many context words appear in the filename."""
        words = {w.lower() for w in context.split() if len(w) > 3}
        scored = []
        for p in paths:
            name = p.stem.lower()
            score = sum(1 for w in words if w in name)
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else paths[0]
