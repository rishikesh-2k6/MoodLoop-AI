"""
media/VideoRenderer.py
──────────────────────
Production-grade FFmpeg-based video renderer for MoodLoop AI.

Generates a 30-second 1080 × 1920 (9:16) vertical MP4 with:

  • Ken Burns slow-zoom effect (configurable start/end scale & position)
  • Centred multi-line text overlay drawn with FFmpeg's drawtext filter
    (no Pillow dependency in this module)
  • Background music with fade-in / fade-out
  • H.264 video + AAC audio, web-optimised output (faststart)

This module is intentionally self-contained: it only relies on the
Python standard library + FFmpeg on the system PATH.
"""

from __future__ import annotations

import logging
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Video constants ───────────────────────────────────────────────── #
_WIDTH: int = 1080
_HEIGHT: int = 1920
_FPS: int = 30
_DURATION: int = 30           # seconds
_FADE_SEC: float = 1.0        # video + audio fade in / out duration
_VIDEO_CODEC: str = "libx264"
_AUDIO_CODEC: str = "aac"
_AUDIO_BITRATE: str = "192k"
_CRF: int = 22
_PRESET: str = "medium"
_PIX_FMT: str = "yuv420p"

# Ken Burns defaults
_KB_ZOOM_START: float = 1.0    # scale factor at frame 0  (1.0 = fill canvas)
_KB_ZOOM_END: float = 1.08     # scale factor at last frame (~8 % zoom-in)

# Text overlay defaults
_FONT_SIZE: int = 68
_FONT_COLOR: str = "white"
_FONT_SHADOW_COLOR: str = "black@0.65"
_BOX_COLOR: str = "black@0.45"
_BOX_BORDER: int = 28          # px padding around text box
_MAX_LINE_LEN: int = 28        # chars before wrapping


# ------------------------------------------------------------------ #
#  Data models                                                         #
# ------------------------------------------------------------------ #

@dataclass
class KenBurnsConfig:
    """
    Parameters for the Ken Burns slow-zoom / pan effect.

    Attributes
    ----------
    zoom_start : float
        Image scale multiplier at the first frame (≥ 1.0).
    zoom_end : float
        Image scale multiplier at the last frame.  Use a value
        slightly larger than *zoom_start* for a slow zoom-in.
    x_drift_px : int
        Horizontal pan distance in pixels over *duration* seconds.
        Positive = pan right; negative = pan left; 0 = no pan.
    y_drift_px : int
        Vertical pan distance in pixels over *duration* seconds.
    ease : bool
        Apply a smoothstep interpolation so motion decelerates near
        the ends (not yet implemented; reserved for future use).
    """

    zoom_start: float = _KB_ZOOM_START
    zoom_end: float = _KB_ZOOM_END
    x_drift_px: int = 0
    y_drift_px: int = 0
    ease: bool = True


@dataclass
class TextConfig:
    """
    Visual parameters for the centred text overlay.

    Attributes
    ----------
    text : str
        The raw quote / caption string.
    font_path : Path | None
        Absolute path to a .ttf/.otf font file.  When ``None`` the
        drawtext filter uses FFmpeg's built-in default font.
    font_size : int
        Base font size in points.
    font_color : str
        FFmpeg colour expression for the text (default: ``"white"``).
    box : bool
        Draw a semi-transparent background box behind the text.
    box_color : str
        FFmpeg colour expression for the backdrop box.
    box_border : int
        Padding in pixels around the text within the box.
    line_spacing : int
        Extra vertical pixels between wrapped lines.
    max_chars : int
        Characters per line before a forced break is inserted.
    """

    text: str
    font_path: Optional[Path] = None
    font_size: int = _FONT_SIZE
    font_color: str = _FONT_COLOR
    box: bool = True
    box_color: str = _BOX_COLOR
    box_border: int = _BOX_BORDER
    line_spacing: int = 12
    max_chars: int = _MAX_LINE_LEN


@dataclass
class RenderResult:
    """
    Outcome of a single :class:`VideoRenderer` render call.

    Attributes
    ----------
    output_path : Path
        Absolute path to the rendered MP4 (may not exist on failure).
    success : bool
        ``True`` when FFmpeg exited with return code 0.
    duration_sec : int
        Target video length that was requested.
    width : int
        Frame width of the output.
    height : int
        Frame height of the output.
    file_size_mb : float
        Size of the output file in megabytes (``0.0`` on failure).
    error_message : str | None
        Tail of FFmpeg stderr when *success* is ``False``.
    """

    output_path: Path
    success: bool
    duration_sec: int = _DURATION
    width: int = _WIDTH
    height: int = _HEIGHT
    file_size_mb: float = 0.0
    error_message: Optional[str] = field(default=None)


# ------------------------------------------------------------------ #
#  VideoRenderer                                                       #
# ------------------------------------------------------------------ #

class VideoRenderer:
    """
    Renders a 30-second vertical short-form video using FFmpeg.

    The render pipeline:
      1. Scale & pad the background image to 1080 × 1920.
      2. Apply Ken Burns slow-zoom via FFmpeg's ``zoompan`` filter.
      3. Burn centred, word-wrapped text onto every frame with ``drawtext``.
      4. Apply video fade-in / fade-out.
      5. Mix the background music track, trim to *duration*, apply
         audio fade-in / fade-out.
      6. Encode H.264 / AAC, ``-movflags +faststart``.

    Parameters
    ----------
    output_dir : Path
        Directory where MP4 files are written.
    ffmpeg_path : str
        FFmpeg binary name or absolute path (default: ``"ffmpeg"``).
    duration_sec : int
        Target video length in seconds (default: 30).
    fps : int
        Frame rate of the output video (default: 30).
    fade_sec : float
        Duration of video + audio fade transitions (default: 1.0 s).
    """

    def __init__(
        self,
        output_dir: Path,
        ffmpeg_path: str = "ffmpeg",
        duration_sec: int = _DURATION,
        fps: int = _FPS,
        fade_sec: float = _FADE_SEC,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_path = ffmpeg_path
        self.duration_sec = duration_sec
        self.fps = fps
        self.fade_sec = fade_sec

        self._verify_ffmpeg()
        self._ffmpeg_major = self._detect_ffmpeg_major()
        if self._ffmpeg_major >= 4:
            logger.info("[VideoRenderer] FFmpeg %d detected — Ken Burns (zoompan) enabled.", self._ffmpeg_major)
        else:
            logger.warning(
                "[VideoRenderer] FFmpeg %d detected (< 4) — Ken Burns unavailable. "
                "Using scale+pad fallback. Upgrade FFmpeg for zoom effect: "
                "https://www.gyan.dev/ffmpeg/builds/",
                self._ffmpeg_major,
            )
        logger.debug(
            "VideoRenderer ready (out=%s, fps=%d, dur=%ds)",
            self.output_dir,
            self.fps,
            self.duration_sec,
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def render(
        self,
        run_id: str,
        background_path: Path,
        music_path: Path,
        text_config: TextConfig,
        ken_burns: Optional[KenBurnsConfig] = None,
    ) -> RenderResult:
        """
        Render a complete short-form video and save it to *output_dir*.

        Parameters
        ----------
        run_id : str
            Unique identifier; used as the MP4 filename stem.
        background_path : Path
            Source background image (JPG / PNG / WEBP …).
        music_path : Path
            Source audio file (MP3 / WAV / AAC …).
        text_config : TextConfig
            Visual parameters for the quote text overlay.
        ken_burns : KenBurnsConfig | None
            Ken Burns parameters.  A default (8 % slow zoom-in) is
            used when ``None``.

        Returns
        -------
        RenderResult
            Populated result object.
        """
        if ken_burns is None:
            ken_burns = KenBurnsConfig()

        output_path = self.output_dir / f"{run_id}.mp4"
        logger.info("[VideoRenderer] Rendering → %s", output_path.name)

        cmd = self._build_command(
            background_path=background_path,
            music_path=music_path,
            output_path=output_path,
            text_config=text_config,
            ken_burns=ken_burns,
        )

        logger.debug("[VideoRenderer] FFmpeg command:\n  %s", " ".join(cmd))
        return self._run_ffmpeg(cmd, output_path)

    def render_with_random_ken_burns(
        self,
        run_id: str,
        background_path: Path,
        music_path: Path,
        text_config: TextConfig,
    ) -> RenderResult:
        """
        Render with a randomised Ken Burns configuration for variety.

        Each call picks a random zoom range (1.0–1.12) and a random
        small pan drift (±30 px) so no two videos look identical.

        Parameters
        ----------
        run_id : str
            Unique pipeline run identifier.
        background_path : Path
            Background image file.
        music_path : Path
            Music audio file.
        text_config : TextConfig
            Text overlay parameters.

        Returns
        -------
        RenderResult
        """
        kb = KenBurnsConfig(
            zoom_start=round(random.uniform(1.0, 1.04), 3),
            zoom_end=round(random.uniform(1.06, 1.12), 3),
            x_drift_px=random.randint(-30, 30),
            y_drift_px=random.randint(-20, 20),
        )
        logger.debug(
            "[VideoRenderer] Random Ken Burns: zoom %.3f→%.3f, drift=(%d,%d)",
            kb.zoom_start,
            kb.zoom_end,
            kb.x_drift_px,
            kb.y_drift_px,
        )
        return self.render(run_id, background_path, music_path, text_config, kb)

    # ------------------------------------------------------------------ #
    #  Filter graph builders                                               #
    # ------------------------------------------------------------------ #

    def _build_command(
        self,
        background_path: Path,
        music_path: Path,
        output_path: Path,
        text_config: TextConfig,
        ken_burns: KenBurnsConfig,
    ) -> list[str]:
        """
        Construct the complete FFmpeg CLI command.

        Automatically selects between two filter graph strategies
        depending on the detected FFmpeg version:

        **FFmpeg ≥ 4 (modern)**
          [0:v]
            scale          → upscale to cover 1080×1920 with headroom
            zoompan        → Ken Burns slow-zoom (outputs 1080×1920 @fps)
            drawtext       → centred quote text with box backdrop
            fade (video)   → fade-in at t=0, fade-out near end
          [1:a]
            atrim + afade in/out

        **FFmpeg < 4 (legacy fallback)**
          [0:v]
            scale+pad      → fit to 1080×1920, black bars if needed
            drawtext       → same text overlay
            fade in/out    → same fades
          [1:a]
            same audio chain

        Parameters
        ----------
        background_path : Path
        music_path : Path
        output_path : Path
        text_config : TextConfig
        ken_burns : KenBurnsConfig

        Returns
        -------
        list[str]
            Fully quoted FFmpeg argument list.
        """
        if self._ffmpeg_major >= 4:
            vf = self._vf_ken_burns(ken_burns, text_config)
        else:
            vf = self._vf_compat(text_config)

        fade_out_start = self.duration_sec - self.fade_sec
        af = (
            f"atrim=0:{self.duration_sec},"
            f"asetpts=PTS-STARTPTS,"
            f"afade=t=in:st=0:d={self.fade_sec},"
            f"afade=t=out:st={fade_out_start}:d={self.fade_sec}"
        )

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-loop", "1",
            "-i", str(background_path),
            "-i", str(music_path),
            "-t", str(self.duration_sec),
            "-vf", vf,
            "-af", af,
            "-c:v", _VIDEO_CODEC,
            "-preset", _PRESET,
            "-crf", str(_CRF),
            "-c:a", _AUDIO_CODEC,
            "-b:a", _AUDIO_BITRATE,
            "-pix_fmt", _PIX_FMT,
            "-r", str(self.fps),
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ]
        return cmd

    def _vf_ken_burns(self, ken_burns: KenBurnsConfig, text_config: TextConfig) -> str:
        """
        Build the modern (FFmpeg ≥ 4) video filter chain with Ken Burns
        zoompan effect and drawtext overlay.

        Parameters
        ----------
        ken_burns : KenBurnsConfig
        text_config : TextConfig

        Returns
        -------
        str
            FFmpeg -vf expression.
        """
        total_frames = self.duration_sec * self.fps
        fade_out_start = self.duration_sec - self.fade_sec
        z_start = ken_burns.zoom_start
        z_end = ken_burns.zoom_end

        zoom_expr = f"'min({z_start}+({z_end}-{z_start})*on/{total_frames},{z_end})'"

        x_expr = (
            f"'(iw-ow)/2+{ken_burns.x_drift_px}*on/{total_frames}'"
            if ken_burns.x_drift_px != 0
            else "'(iw-ow)/2'"
        )
        y_expr = (
            f"'(ih-oh)/2+{ken_burns.y_drift_px}*on/{total_frames}'"
            if ken_burns.y_drift_px != 0
            else "'(ih-oh)/2'"
        )

        zoompan = (
            f"zoompan=z={zoom_expr}:x={x_expr}:y={y_expr}"
            f":d={total_frames}:s={_WIDTH}x{_HEIGHT}:fps={self.fps}"
        )

        pad_factor = max(z_end, 1.0) + 0.02
        scale_w = int(_WIDTH * pad_factor / 2) * 2
        scale_h = int(_HEIGHT * pad_factor / 2) * 2
        scale = (
            f"scale={scale_w}:{scale_h}:"
            f"force_original_aspect_ratio=increase,"
            f"crop={scale_w}:{scale_h}"
        )

        drawtext_filters = self._build_drawtext_chain(text_config)
        vfade = (
            f"fade=t=in:st=0:d={self.fade_sec},"
            f"fade=t=out:st={fade_out_start}:d={self.fade_sec}"
        )
        return f"{scale},{zoompan},{drawtext_filters},{vfade}"

    def _vf_compat(self, text_config: TextConfig) -> str:
        """
        Build a legacy-compatible (FFmpeg 2.x / 3.x) video filter chain.

        Uses simple scale+pad instead of zoompan so it works on any
        FFmpeg version — ideal for the user's installed 2013 build while
        they update to a modern release.

        Parameters
        ----------
        text_config : TextConfig

        Returns
        -------
        str
            FFmpeg -vf expression.
        """
        fade_out_start = self.duration_sec - self.fade_sec

        # Scale to fill canvas (crop to exact 1080×1920)
        scale = (
            f"scale={_WIDTH}:{_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={_WIDTH}:{_HEIGHT}"
        )

        drawtext_filters = self._build_drawtext_chain(text_config)
        vfade = (
            f"fade=t=in:st=0:d={self.fade_sec},"
            f"fade=t=out:st={fade_out_start}:d={self.fade_sec}"
        )
        return f"{scale},{drawtext_filters},{vfade}"

    def _build_drawtext_chain(self, cfg: TextConfig) -> str:
        """
        Build a chain of FFmpeg ``drawtext`` filter expressions — one per
        wrapped line — centred vertically and horizontally on the canvas.

        Parameters
        ----------
        cfg : TextConfig
            Text visual configuration.

        Returns
        -------
        str
            Comma-separated drawtext filter string ready to insert into
            a ``-vf`` chain.
        """
        lines = self._wrap_text(cfg.text, cfg.max_chars)
        if not lines:
            return "null"   # pass-through when no text

        line_h = cfg.font_size + cfg.line_spacing
        total_text_h = line_h * len(lines)

        # Vertical start: centre block on the lower-middle of the frame
        # (60 % down) so the top portion stays clean for aesthetic shots
        centre_y = int(_HEIGHT * 0.60)
        y_start = centre_y - total_text_h // 2

        font_arg = ""
        if cfg.font_path and Path(cfg.font_path).exists():
            # FFmpeg on Windows needs forward slashes and escaped colons
            font_str = str(cfg.font_path).replace("\\", "/").replace(":", "\\:")
            font_arg = f":fontfile='{font_str}'"

        box_arg = ""
        if cfg.box:
            box_arg = (
                f":box=1:boxcolor={cfg.box_color}"
                f":boxborderw={cfg.box_border}"
            )

        filters: list[str] = []
        for i, line in enumerate(lines):
            escaped = self._escape_drawtext(line)
            y_pos = y_start + i * line_h
            dt = (
                f"drawtext=text='{escaped}'"
                f"{font_arg}"
                f":fontsize={cfg.font_size}"
                f":fontcolor={cfg.font_color}"
                f":x=(w-text_w)/2"
                f":y={y_pos}"
                f"{box_arg}"
                f":shadowx=2:shadowy=2:shadowcolor={_FONT_SHADOW_COLOR}"
            )
            filters.append(dt)

        return ",".join(filters)

    # ------------------------------------------------------------------ #
    #  FFmpeg execution                                                    #
    # ------------------------------------------------------------------ #

    def _run_ffmpeg(self, cmd: list[str], output_path: Path) -> RenderResult:
        """
        Execute *cmd* via subprocess and return a populated :class:`RenderResult`.

        Parameters
        ----------
        cmd : list[str]
            Fully constructed FFmpeg argument list.
        output_path : Path
            Expected output file location.

        Returns
        -------
        RenderResult
        """
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.duration_sec * 15,
            )
        except subprocess.TimeoutExpired:
            msg = f"FFmpeg timed out after {self.duration_sec * 15}s"
            logger.error("[VideoRenderer] %s", msg)
            return RenderResult(
                output_path=output_path,
                success=False,
                error_message=msg,
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"FFmpeg subprocess error: {exc}"
            logger.error("[VideoRenderer] %s", msg)
            return RenderResult(
                output_path=output_path,
                success=False,
                error_message=msg,
            )

        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "")[-600:]
            logger.error(
                "[VideoRenderer] FFmpeg exited %d:\n%s",
                proc.returncode,
                stderr_tail,
            )
            return RenderResult(
                output_path=output_path,
                success=False,
                error_message=stderr_tail,
            )

        size_mb = output_path.stat().st_size / 1_048_576
        logger.info(
            "[VideoRenderer] ✓ %s rendered (%.1f MB, %ds)",
            output_path.name,
            size_mb,
            self.duration_sec,
        )
        return RenderResult(
            output_path=output_path,
            success=True,
            duration_sec=self.duration_sec,
            width=_WIDTH,
            height=_HEIGHT,
            file_size_mb=round(size_mb, 2),
        )

    # ------------------------------------------------------------------ #
    #  Static utilities                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _wrap_text(text: str, max_chars: int) -> list[str]:
        """
        Split *text* into lines of at most *max_chars* characters,
        respecting existing word boundaries.

        Parameters
        ----------
        text : str
            Raw quote string (may contain ``\\n``).
        max_chars : int
            Maximum characters per line.

        Returns
        -------
        list[str]
            List of wrapped line strings.
        """
        import textwrap
        lines: list[str] = []
        for paragraph in text.splitlines():
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            wrapped = textwrap.wrap(paragraph, width=max_chars)
            lines.extend(wrapped)
        return lines or [text.strip()]

    @staticmethod
    def _escape_drawtext(text: str) -> str:
        """
        Escape special characters for FFmpeg's ``drawtext`` filter.

        Characters that must be escaped: ``'``, ``:``, ``\\``.

        Parameters
        ----------
        text : str
            Raw line of text.

        Returns
        -------
        str
            Escaped string safe to embed in a drawtext expression.
        """
        text = text.replace("\\", "\\\\")
        text = text.replace("'", "\\'")
        text = text.replace(":", "\\:")
        return text

    def _verify_ffmpeg(self) -> None:
        """
        Confirm the FFmpeg binary is reachable.

        Raises
        ------
        EnvironmentError
            If ``shutil.which`` cannot find the binary.
        """
        resolved = shutil.which(self.ffmpeg_path)
        if resolved is None:
            raise EnvironmentError(
                f"FFmpeg not found at '{self.ffmpeg_path}'. "
                "Install FFmpeg and add it to your system PATH, or pass the "
                "full path via ffmpeg_path= during construction.\n"
                "Windows: https://www.gyan.dev/ffmpeg/builds/\n"
                "macOS:   brew install ffmpeg\n"
                "Linux:   sudo apt install ffmpeg"
            )
        logger.info("[VideoRenderer] FFmpeg found → %s", resolved)

    def _detect_ffmpeg_major(self) -> int:
        """
        Parse ``ffmpeg -version`` to extract the major version number.

        Returns
        -------
        int
            Major version (e.g. ``6`` for FFmpeg 6.1.1).
            Returns ``0`` if detection fails so the compat path is used.
        """
        import re
        try:
            proc = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # First line looks like: "ffmpeg version 6.1.1 ..."
            # or legacy:             "ffmpeg version N-55702-g920046a ..."
            first_line = (proc.stdout or proc.stderr or "").splitlines()[0]
            m = re.search(r"version\s+(?:N-\d+-\w+|([\d]+))", first_line)
            if m and m.group(1):
                major = int(m.group(1))
                logger.debug("[VideoRenderer] FFmpeg major version: %d", major)
                return major
            # Legacy build string like N-55702-g920046a has no numeric version
            logger.debug("[VideoRenderer] FFmpeg version string '%s' — treating as legacy (< 4).", first_line.strip())
            return 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("[VideoRenderer] Could not detect FFmpeg version: %s — using compat mode.", exc)
            return 0
