"""
core/TextOverlay.py
───────────────────
Renders a styled quote card as a PNG image using Pillow.

The card is designed for 9:16 vertical video (1080 × 1920 px) and
applies:
  • Semi-transparent dark gradient panel
  • Centred, word-wrapped quote text with configurable font
  • Optional attribution line (e.g. '@MoodLoopAI')
  • Returns the rendered image as a Pillow Image object *or* saves to disk

This module has no FFmpeg dependency — it is pure Python / Pillow.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont, ImageFilter

logger = logging.getLogger(__name__)

# ── Default visual constants ─────────────────────────────────────── #
_CANVAS_W = 1080
_CANVAS_H = 1920

_PANEL_X0_RATIO = 0.04
_PANEL_X1_RATIO = 0.96
_PANEL_Y_CENTER_RATIO = 0.50   # vertical mid-point of the panel on the canvas

_QUOTE_FONT_SIZE = 72
_ATTR_FONT_SIZE = 38
_LINE_SPACING = 1.35            # multiplier over font size

_TEXT_COLOR = (255, 255, 255, 255)       # white
_ATTR_COLOR = (200, 200, 200, 180)       # soft grey / semi-transparent
_PANEL_COLOR = (10, 10, 10, 170)         # near-black, 67 % opacity


class TextOverlay:
    """
    Renders quote text onto an image canvas designed for vertical video.

    Parameters
    ----------
    canvas_size : tuple[int, int]
        Width × height of the output image (default: 1080 × 1920).
    quote_font_path : Path | None
        Path to a .ttf / .otf font file for the quote.  Falls back to
        Pillow's built-in default when ``None``.
    attr_font_path : Path | None
        Path to a font for the attribution line.  Falls back to default.
    quote_font_size : int
        Point size for the quote font (default: 72).
    attr_font_size : int
        Point size for the attribution font (default: 38).
    attribution : str
        Watermark / handle text (default: ``"@MoodLoopAI"``).
    max_chars_per_line : int
        Maximum characters before ``textwrap`` inserts a line break (default: 28).
    """

    def __init__(
        self,
        canvas_size: tuple[int, int] = (_CANVAS_W, _CANVAS_H),
        quote_font_path: Optional[Path] = None,
        attr_font_path: Optional[Path] = None,
        quote_font_size: int = _QUOTE_FONT_SIZE,
        attr_font_size: int = _ATTR_FONT_SIZE,
        attribution: str = "@MoodLoopAI",
        max_chars_per_line: int = 28,
    ) -> None:
        self.canvas_w, self.canvas_h = canvas_size
        self.attribution = attribution
        self.max_chars_per_line = max_chars_per_line

        self._quote_font = self._load_font(quote_font_path, quote_font_size)
        self._attr_font = self._load_font(attr_font_path, attr_font_size)
        logger.debug("TextOverlay ready (canvas=%dx%d)", self.canvas_w, self.canvas_h)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def render(
        self,
        background: Image.Image,
        quote: str,
        output_path: Optional[Path] = None,
    ) -> Image.Image:
        """
        Composite a quote card onto *background* and optionally save it.

        Parameters
        ----------
        background : Image.Image
            Background image (will be resized to canvas dimensions).
        quote : str
            Quote text to render.
        output_path : Path | None
            If provided, the composed image is saved as PNG.

        Returns
        -------
        Image.Image
            Composited RGBA image.
        """
        # 1. Resize + convert background
        bg = background.convert("RGBA").resize(
            (self.canvas_w, self.canvas_h), Image.LANCZOS
        )

        # 2. Optionally apply subtle blur so text pops
        bg = bg.filter(ImageFilter.GaussianBlur(radius=3))

        # 3. Build semi-transparent panel
        panel_layer = self._make_panel(quote)

        # 4. Composite
        composed = Image.alpha_composite(bg, panel_layer)

        # 5. Draw attribution watermark
        composed = self._draw_attribution(composed)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            composed.save(str(output_path), format="PNG")
            logger.info("TextOverlay saved → %s", output_path)

        return composed

    def render_from_path(
        self,
        background_path: Path,
        quote: str,
        output_path: Optional[Path] = None,
    ) -> Image.Image:
        """
        Convenience wrapper that loads the background from *background_path*.

        Parameters
        ----------
        background_path : Path
            Path to the background image file.
        quote : str
            Quote text.
        output_path : Path | None
            Optional save path.

        Returns
        -------
        Image.Image
            Composited image.
        """
        bg = Image.open(str(background_path))
        return self.render(bg, quote, output_path)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _make_panel(self, quote: str) -> Image.Image:
        """
        Build a transparent layer containing the dark panel + quote text.

        Parameters
        ----------
        quote : str
            Quote text to render inside the panel.

        Returns
        -------
        Image.Image
            RGBA layer the same size as the canvas.
        """
        layer = Image.new("RGBA", (self.canvas_w, self.canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        # Word-wrap the quote
        wrapped_lines = textwrap.wrap(quote, width=self.max_chars_per_line)
        if not wrapped_lines:
            wrapped_lines = [""]

        # Measure total text block height
        line_h = int(self._quote_font.size * _LINE_SPACING)
        total_text_h = line_h * len(wrapped_lines)

        # Panel geometry
        padding_x = int(self.canvas_w * _PANEL_X0_RATIO)
        padding_y = int(total_text_h * 0.4)
        panel_x0 = padding_x
        panel_x1 = int(self.canvas_w * _PANEL_X1_RATIO)

        center_y = int(self.canvas_h * _PANEL_Y_CENTER_RATIO)
        panel_y0 = center_y - total_text_h // 2 - padding_y
        panel_y1 = center_y + total_text_h // 2 + padding_y

        # Draw rounded rectangle panel
        draw.rounded_rectangle(
            [panel_x0, panel_y0, panel_x1, panel_y1],
            radius=32,
            fill=_PANEL_COLOR,
        )

        # Draw each line centred on the panel
        y_cursor = panel_y0 + padding_y
        for line in wrapped_lines:
            try:
                bbox = draw.textbbox((0, 0), line, font=self._quote_font)
                line_w = bbox[2] - bbox[0]
            except AttributeError:
                # Pillow < 9 fallback
                line_w, _ = draw.textsize(line, font=self._quote_font)  # type: ignore[attr-defined]

            x = (self.canvas_w - line_w) // 2
            draw.text((x, y_cursor), line, font=self._quote_font, fill=_TEXT_COLOR)
            y_cursor += line_h

        return layer

    def _draw_attribution(self, image: Image.Image) -> Image.Image:
        """
        Draw the attribution handle in the bottom-right corner.

        Parameters
        ----------
        image : Image.Image
            Canvas to draw on (in-place).

        Returns
        -------
        Image.Image
            Modified image.
        """
        draw = ImageDraw.Draw(image)
        margin = 40
        try:
            bbox = draw.textbbox((0, 0), self.attribution, font=self._attr_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(self.attribution, font=self._attr_font)  # type: ignore[attr-defined]

        x = self.canvas_w - text_w - margin
        y = self.canvas_h - text_h - margin
        draw.text((x, y), self.attribution, font=self._attr_font, fill=_ATTR_COLOR)
        return image

    @staticmethod
    def _load_font(path: Optional[Path], size: int) -> ImageFont.FreeTypeFont:
        """
        Load a TrueType font; fall back to Pillow's default on failure.

        Parameters
        ----------
        path : Path | None
            Font file path, or ``None`` to use the built-in default.
        size : int
            Desired point size.

        Returns
        -------
        ImageFont.FreeTypeFont | ImageFont.ImageFont
            Loaded font object.
        """
        if path is not None:
            font_path = Path(path)
            if font_path.exists():
                try:
                    return ImageFont.truetype(str(font_path), size=size)
                except OSError as exc:
                    logger.warning("Could not load font %s: %s — using default.", font_path, exc)
            else:
                logger.warning("Font path not found: %s — using default.", font_path)

        # Try bundled Playfair Display first (premium editorial serif)
        _FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
        _BUNDLED_FONT = _FONTS_DIR / "PlayfairDisplay.ttf"
        for candidate in [_BUNDLED_FONT]:
            if candidate.exists():
                try:
                    return ImageFont.truetype(str(candidate), size=size)
                except OSError as exc:
                    logger.warning("Could not load bundled font %s: %s", candidate, exc)

        try:
            # Try Georgia Bold as system fallback
            return ImageFont.truetype("georgiab.ttf", size=size)
        except OSError:
            pass

        try:
            return ImageFont.truetype("arial.ttf", size=size)
        except OSError:
            pass

        logger.info("Using Pillow built-in default font (no TTF available).")
        return ImageFont.load_default()
