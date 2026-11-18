"""Local face-aware vertical reframing with a blur-fill fallback."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import math
import subprocess
from typing import Any


def _probe_dimensions(video_path: str | Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        width, height = result.stdout.strip().split("x", maxsplit=1)
        return int(width), int(height)
    except (ValueError, TypeError) as error:
        raise RuntimeError(f"Unable to read video dimensions for {video_path}") from error


def _face_centers(
    video_path: str | Path, start: float, duration: float, sample_rate: float = 3.0
) -> list[tuple[float, float]]:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError(
            "Face tracking requires mediapipe and opencv-python-headless in the virtualenv."
        ) from error

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for face tracking: {video_path}")

    detections: list[tuple[float, float]] = []
    interval = 1.0 / max(0.5, sample_rate)
    try:
        detector_module = getattr(mp, "solutions", None)
        if detector_module is None:
            raise RuntimeError(
                "The installed MediaPipe build does not expose FaceDetection solutions."
            )
        with detector_module.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        ) as detector:
            next_sample = start
            while next_sample < start + duration:
                capture.set(cv2.CAP_PROP_POS_MSEC, next_sample * 1000)
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.process(rgb)
                if result.detections:
                    best = max(
                        result.detections,
                        key=lambda item: item.score[0] if item.score else 0,
                    )
                    center_x = best.location_data.relative_bounding_box.xmin + (
                        best.location_data.relative_bounding_box.width / 2
                    )
                    detections.append((next_sample - start, center_x))
                next_sample += interval
    finally:
        capture.release()
    return detections


def _smoothed_crop_expression(
    detections: list[tuple[float, float]],
    source_width: int,
    source_height: int,
    clip_duration: float,
) -> str | None:
    crop_width = min(source_width, source_height * 9 / 16)
    crop_height = min(source_height, source_width * 16 / 9)
    if not detections or crop_width <= 0 or crop_height <= 0:
        return None

    half_crop = crop_width / 2
    deadzone = 0.035
    rolling: deque[float] = deque(maxlen=5)
    samples: list[tuple[float, float]] = []
    last_center: float | None = None
    for timestamp, relative_x in detections:
        center = min(source_width, max(0.0, relative_x * source_width))
        if last_center is not None and abs(center - last_center) / source_width < deadzone:
            center = last_center
        rolling.append(center)
        last_center = sum(rolling) / len(rolling)
        crop_x = min(source_width - crop_width, max(0.0, last_center - half_crop))
        samples.append((min(max(timestamp, 0), clip_duration), crop_x))

    if len(samples) == 1:
        return str(round(samples[0][1], 2))

    expression = str(round(samples[-1][1], 2))
    for index in range(len(samples) - 2, -1, -1):
        t0, x0 = samples[index]
        t1, x1 = samples[index + 1]
        if t1 <= t0:
            segment = str(round(x1, 2))
        else:
            segment = f"({x0:.2f}+({x1 - x0:.2f})*(t-{t0:.3f})/{t1 - t0:.3f})"
        expression = f"if(lt(t,{t1:.3f}),{segment},{expression})"
    return expression


def _blur_fill_filter() -> str:
    return (
        "[0:v]split=2[background][foreground];"
        "[background]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=30:10[blurred];"
        "[foreground]scale=1080:1920:force_original_aspect_ratio=decrease[clear];"
        "[blurred][clear]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )


def build_reframe_filter(
    video_path: str | Path,
    mode: str,
    start: float,
    duration: float,
    detect_faces: bool = True,
) -> tuple[str, str]:
    """Return a video filter graph and the effective mode actually applied."""
    if mode == "blur":
        return _blur_fill_filter(), "blur"
    width, height = _probe_dimensions(video_path)
    crop_width = min(width, height * 9 / 16)
    crop_height = min(height, width * 16 / 9)

    if mode == "face" and detect_faces:
        detections = _face_centers(video_path, start, duration)
        expression = _smoothed_crop_expression(detections, width, height, duration)
        if expression is None:
            return _blur_fill_filter(), "blur"
        graph = (
            f"crop={crop_width:.2f}:{crop_height:.2f}:"
            f"x='{expression}':y='(ih-oh)/2':eval=frame,"
            "scale=1080:1920:flags=lanczos,setsar=1"
        )
        return graph, "face"

    return (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,setsar=1",
        "center",
    )


def _escape_filter_path(path: str | Path) -> str:
    return (
        str(Path(path).resolve())
        .replace("\\", r"\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(",", r"\,")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )


def render_vertical_clip(
    source: str | Path,
    output: str | Path,
    start: float,
    end: float,
    framing_mode: str = "face",
    caption_file: str | Path | None = None,
) -> dict[str, Any]:
    """Render a source interval as a local H.264 1080x1920 MP4."""
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("Clip bounds must be finite and end must be after start.")
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = {"face-track": "face", "blur-fill": "blur", "center-crop": "center"}.get(
        framing_mode, framing_mode
    )
    if mode not in {"face", "blur", "center"}:
        raise ValueError(f"Unsupported framing mode: {framing_mode}")

    filter_graph, effective_mode = build_reframe_filter(
        source_path, mode, start, end - start
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{end - start:.3f}",
        "-map",
        "0:a?",
    ]
    if effective_mode == "blur":
        if caption_file:
            filter_graph += f",ass='{_escape_filter_path(caption_file)}'"
        command.extend(["-filter_complex", f"{filter_graph}[vout]", "-map", "[vout]"])
    else:
        if caption_file:
            filter_graph += f",ass='{_escape_filter_path(caption_file)}'"
        command.extend(["-map", "0:v:0", "-vf", filter_graph])
    command.extend([
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ])
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip()[-4000:] if error.stderr else "unknown FFmpeg error"
        raise RuntimeError(f"FFmpeg render failed: {detail}") from error
    return {
        "path": str(output_path),
        "duration": round(end - start, 3),
        "width": 1080,
        "height": 1920,
        "framing_mode": effective_mode,
    }
