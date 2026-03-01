"""
Video processing for glitch art.
"""

import io
import tempfile
from pathlib import Path
from typing import Union, Optional, Callable

import numpy as np

def _check_cv2():
    """Check if cv2 is importable. Returns (ok, error_msg)."""
    try:
        import cv2  # noqa: F401
        return True, None
    except ImportError as e:
        return False, "opencv-python not installed"
    except OSError as e:
        return False, f"opencv-python installed but failed to load ({e})"
    except Exception as e:
        return False, str(e)


HAS_CV2, CV2_ERROR = _check_cv2()


def generate_glitch_video_bytes(
    video_input: Union[str, Path, bytes],
    effects_per_frame: int = 2,
    preserve_mask_fn: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None,
    frame_skip: int = 0,
    max_frames: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> bytes:
    """
    Process video with glitch effects, return as MP4 bytes.
    
    video_input: file path, Path, or raw bytes
    effects_per_frame: number of random effects per frame
    preserve_mask_fn: called with (frame) -> mask or None to preserve center
    frame_skip: process every Nth frame (0 = all), others get glitch from nearby
    max_frames: cap total frames (None = no limit)
    progress_callback: (current, total) for progress bar
    """
    if not HAS_CV2:
        raise ImportError("opencv-python required for video: pip install opencv-python")

    from .effects import glitch_random
    from .core import apply_glitch_mask

    # Ensure we have a file path for OpenCV
    if isinstance(video_input, bytes):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_input)
            video_path = f.name
        cleanup_input = True
    else:
        video_path = str(video_input)
        cleanup_input = False

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames and total_frames > 0:
            total_frames = min(total_frames, max_frames)

        # Write to temp file (OpenCV doesn't write to buffer easily)
        out_path = tempfile.mktemp(suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_idx = 0
        last_glitched = None

        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break

            # BGR -> RGB for glitch
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            if frame_skip == 0 or frame_idx % (frame_skip + 1) == 0:
                glitched = glitch_random(rgb.copy(), num_effects=effects_per_frame, seed=frame_idx)
                glitched = np.clip(glitched, 0, 255).astype(np.uint8)
                last_glitched = glitched
            else:
                glitched = last_glitched if last_glitched is not None else rgb.copy()

            # Apply preserve mask
            if preserve_mask_fn is not None:
                mask = preserve_mask_fn(rgb)
                if mask is not None:
                    glitched = apply_glitch_mask(rgb, glitched, mask)

            # RGB -> BGR for output
            out.write(cv2.cvtColor(glitched, cv2.COLOR_RGB2BGR))

            if progress_callback and total_frames > 0:
                progress_callback(frame_idx + 1, total_frames)

            frame_idx += 1

        cap.release()
        out.release()

        with open(out_path, "rb") as f:
            result = f.read()

        try:
            Path(out_path).unlink(missing_ok=True)
        except OSError:
            pass

        return result

    finally:
        if cleanup_input:
            try:
                Path(video_path).unlink(missing_ok=True)
            except OSError:
                pass
