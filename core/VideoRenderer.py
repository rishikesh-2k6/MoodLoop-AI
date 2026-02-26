"""
core/VideoRenderer.py
─────────────────────
Renders a 30-second 9:16 vertical short-form video using FFmpeg (via
subprocess) by combining:

  1. A still background image (already processed by TextOverlay)
  2. A music / audio track
  3. Fade-in / fade-out transitions

The renderer calls FFmpeg as an external process.  No Python video
library is required — only a system FFmpeg installation accessible on
PATH (or at a custom path supplied during construction).

Output: H.264 MP4 at 1080 × 1920, 30 fps, AAC audio, ~30 seconds.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from core.TextOverlay import TextOverlay

logger = logging.getLogger(__name__)

# ── Video constants ───────────────────────────────────────────────── #
_WIDTH = 1080
_HEIGHT = 1920
_FPS = 30
_DURATION_SEC = 30
_FADE_SEC = 1.0       # length of fade-in / fade-out in seconds
_VIDEO_CODEC = "libx264"
_AUDIO_CODEC = "aac"
_AUDIO_BITRATE = "256k"
_CRF = 18              # constant rate factor (lower = better quality, 18 ≈ visually lossless)
_PRESET = "medium"     # FFmpeg preset (ultrafast → veryslow)


# ------------------------------------------------------------------ #
#  Data model                                                          #
# ------------------------------------------------------------------ #

@dataclass
class RenderResult:
    """
    Result of a single VideoRenderer run.

    Attributes
    ----------
    output_path : Path
        Absolute path to the rendered MP4.
    duration_sec : int
        Target video duration in seconds.
    width : int
        Video width in pixels.
    height : int
        Video height in pixels.
    success : bool
        ``True`` if FFmpeg exited with code 0.
    error_message : str | None
        FFmpeg stderr (last 500 chars) if rendering failed.
    """

    output_path: Path
    duration_sec: int
    width: int
    height: int
    success: bool
    error_message: Optional[str] = None


# ------------------------------------------------------------------ #
#  VideoRenderer                                                       #
# ------------------------------------------------------------------ #

class VideoRenderer:
    """
    Orchestrates FFmpeg to produce a 30-second vertical video.

    Parameters
    ----------
    output_dir : Path
        Directory where rendered MP4 files are written.
    ffmpeg_path : str
        Path or command name for the FFmpeg binary (default: ``"ffmpeg"``).
    duration_sec : int
        Target video length in seconds (default: 30).
    fps : int
        Output frame rate (default: 30).
    fade_sec : float
        Duration of video fade-in and fade-out (default: 1.0).
    text_overlay : TextOverlay | None
        Pre-configured ``TextOverlay`` instance.  A default one is
        created if ``None``.
    quote_font_path : Path | None
        Forwarded to the default ``TextOverlay`` if one is created.
    """

    def __init__(
        self,
        output_dir: Path,
        ffmpeg_path: str = "ffmpeg",
        duration_sec: int = _DURATION_SEC,
        fps: int = _FPS,
        fade_sec: float = _FADE_SEC,
        text_overlay: Optional[TextOverlay] = None,
        quote_font_path: Optional[Path] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ffmpeg_path = ffmpeg_path
        self.duration_sec = duration_sec
        self.fps = fps
        self.fade_sec = fade_sec

        self._overlay = text_overlay or TextOverlay(
            canvas_size=(_WIDTH, _HEIGHT),
            quote_font_path=quote_font_path,
        )

        self._verify_ffmpeg()
        logger.debug(
            "VideoRenderer ready (output=%s, ffmpeg=%s, dur=%ds)",
            self.output_dir,
            self.ffmpeg_path,
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
        quote: str,
        title: str,
    ) -> RenderResult:
        """
        Render a complete 30-second vertical video.

        Pipeline:
          1. Composite quote text onto background image (Pillow).
          2. Encode still image + audio → MP4 via FFmpeg.

        Parameters
        ----------
        run_id : str
            Unique identifier used for naming the output file.
        background_path : Path
            Source background image (jpg / png / webp …).
        music_path : Path
            Source audio file (mp3 / wav / aac …).
        quote : str
            Quote text to render onto the video frame.
        title : str
            Not rendered on video in this phase (reserved for metadata).

        Returns
        -------
        RenderResult
            Result object containing output path and success status.
        """
        output_path = self.output_dir / f"{run_id}.mp4"
        logger.info("[VideoRenderer] Rendering %s…", output_path.name)

        # ── Step 1: Composite quote card ─────────────────────────────── #
        with tempfile.TemporaryDirectory() as tmp:
            frame_path = Path(tmp) / "frame.png"
            logger.debug("[VideoRenderer] Compositing quote card → %s", frame_path)

            try:
                bg_image = Image.open(str(background_path))
                self._overlay.render(bg_image, quote, output_path=frame_path)
            except Exception as exc:  # noqa: BLE001
                msg = f"TextOverlay compositing failed: {exc}"
                logger.error("[VideoRenderer] %s", msg)
                return RenderResult(
                    output_path=output_path,
                    duration_sec=self.duration_sec,
                    width=_WIDTH,
                    height=_HEIGHT,
                    success=False,
                    error_message=msg,
                )

            # ── Step 2: FFmpeg encode ────────────────────────────────── #
            result = self._ffmpeg_encode(
                frame_path=frame_path,
                audio_path=music_path,
                output_path=output_path,
            )

        return result

    # ------------------------------------------------------------------ #
    #  FFmpeg helpers                                                      #
    # ------------------------------------------------------------------ #

    def _ffmpeg_encode(
        self,
        frame_path: Path,
        audio_path: Path,
        output_path: Path,
    ) -> RenderResult:
        """
        Build and execute the FFmpeg command that combines still + audio.

        Filter graph:
          • [0:v] loop the single image for *duration_sec* seconds
          • apply fade-in / fade-out on video
          • [1:a] trim audio to *duration_sec*, apply afade in/out
          • scale + pad to 1080 × 1920 (safety net for mismatched input)

        Parameters
        ----------
        frame_path : Path
            Composited frame PNG.
        audio_path : Path
            Music / audio file.
        output_path : Path
            Desired MP4 output path.

        Returns
        -------
        RenderResult
            Populated result object.
        """
        fade_out_start = self.duration_sec - self.fade_sec

        vf = (
            f"scale={_WIDTH}:{_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={_WIDTH}:{_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,"
            f"fade=t=in:st=0:d={self.fade_sec},"
            f"fade=t=out:st={fade_out_start}:d={self.fade_sec}"
        )

        af = (
            f"afade=t=in:st=0:d={self.fade_sec},"
            f"afade=t=out:st={fade_out_start}:d={self.fade_sec}"
        )

        cmd = [
            self.ffmpeg_path,
            "-y",                             # overwrite without prompt
            "-loop", "1",                     # loop still image
            "-i", str(frame_path),            # input 0: image
            "-i", str(audio_path),            # input 1: audio
            "-t", str(self.duration_sec),     # stop after N seconds
            "-vf", vf,                        # video filter chain
            "-af", af,                        # audio filter chain
            "-r", str(self.fps),              # frame rate
            "-c:v", _VIDEO_CODEC,
            "-preset", _PRESET,
            "-crf", str(_CRF),
            "-c:a", _AUDIO_CODEC,
            "-strict", "-2",                  # enable experimental AAC on legacy FFmpeg
            "-b:a", _AUDIO_BITRATE,
            "-pix_fmt", "yuv420p",            # broadest compatibility
            "-shortest",                      # stop when shortest stream ends
            "-movflags", "+faststart",        # web-optimised atom placement
            str(output_path),
        ]

        logger.debug("[VideoRenderer] FFmpeg command:\n  %s", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.duration_sec * 10,   # generous headroom
            )
        except subprocess.TimeoutExpired:
            msg = f"FFmpeg timed out after {self.duration_sec * 10}s"
            logger.error("[VideoRenderer] %s", msg)
            return RenderResult(
                output_path=output_path,
                duration_sec=self.duration_sec,
                width=_WIDTH,
                height=_HEIGHT,
                success=False,
                error_message=msg,
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"FFmpeg subprocess error: {exc}"
            logger.error("[VideoRenderer] %s", msg)
            return RenderResult(
                output_path=output_path,
                duration_sec=self.duration_sec,
                width=_WIDTH,
                height=_HEIGHT,
                success=False,
                error_message=msg,
            )

        if proc.returncode != 0:
            stderr_tail = proc.stderr[-500:] if proc.stderr else "(no stderr)"
            logger.error(
                "[VideoRenderer] FFmpeg exited %d:\n%s", proc.returncode, stderr_tail
            )
            return RenderResult(
                output_path=output_path,
                duration_sec=self.duration_sec,
                width=_WIDTH,
                height=_HEIGHT,
                success=False,
                error_message=stderr_tail,
            )

        size_mb = output_path.stat().st_size / 1_048_576
        logger.info(
            "[VideoRenderer] ✓ Rendered %s (%.1f MB)", output_path.name, size_mb
        )
        return RenderResult(
            output_path=output_path,
            duration_sec=self.duration_sec,
            width=_WIDTH,
            height=_HEIGHT,
            success=True,
        )

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def _verify_ffmpeg(self) -> None:
        """
        Confirm that FFmpeg is available on the system.

        Raises
        ------
        EnvironmentError
            If the FFmpeg binary cannot be found.
        """
        resolved = shutil.which(self.ffmpeg_path)
        if resolved is None:
            raise EnvironmentError(
                f"FFmpeg not found at '{self.ffmpeg_path}'. "
                "Install FFmpeg and ensure it is on your system PATH, "
                "or pass the full path via ffmpeg_path= during construction."
            )
        logger.info("[VideoRenderer] FFmpeg found at: %s", resolved)
