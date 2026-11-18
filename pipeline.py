"""Local media pipeline and durable project storage."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from captions import words_from_caption, write_ass
from reframer import render_vertical_clip


ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
PROJECTS_FILE = ROOT / "projects.json"
for directory in (UPLOAD_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

_PROJECTS_LOCK = threading.RLock()
ProgressCallback = Callable[[str, str, float], None]


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _read_projects() -> list[dict[str, Any]]:
    if not PROJECTS_FILE.exists():
        return []
    try:
        content = json.loads(PROJECTS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to read project archive: {error}") from error
    if not isinstance(content, list):
        raise RuntimeError("projects.json must contain a JSON array.")
    return content


def list_projects() -> list[dict[str, Any]]:
    with _PROJECTS_LOCK:
        return _read_projects()


def get_project(project_id: str) -> dict[str, Any]:
    project = next(
        (item for item in list_projects() if item.get("id") == project_id), None
    )
    if project is None:
        raise KeyError(f"Project not found: {project_id}")
    return project


def _save_project(project: dict[str, Any]) -> None:
    with _PROJECTS_LOCK:
        projects = _read_projects()
        existing_index = next(
            (i for i, item in enumerate(projects) if item.get("id") == project["id"]),
            None,
        )
        if existing_index is None:
            projects.insert(0, project)
        else:
            projects[existing_index] = project
        temporary = PROJECTS_FILE.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(projects, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(PROJECTS_FILE)


def _ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:format_tags=title",
            "-show_entries",
            "stream=codec_type,width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        metadata = json.loads(result.stdout)
        duration = float(metadata.get("format", {}).get("duration", 0))
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read media metadata for {path}") from error
    if duration <= 0:
        raise RuntimeError("The selected source has no playable video duration.")
    return {
        "duration": duration,
        "title": metadata.get("format", {}).get("tags", {}).get("title"),
        "has_video": any(
            stream.get("codec_type") == "video" for stream in metadata.get("streams", [])
        ),
    }


def _ingest_youtube(url: str, progress: ProgressCallback) -> tuple[Path, str]:
    from urllib.parse import urlparse

    parsed = urlparse(url.strip())
    allowed_hosts = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
    if parsed.scheme not in {"http", "https"} or (parsed.hostname or "").lower() not in allowed_hosts:
        raise ValueError("Only a YouTube or youtu.be URL can be downloaded.")

    try:
        import yt_dlp
    except ImportError as error:
        raise RuntimeError("yt-dlp is not installed in the active virtualenv.") from error

    output_template = str(UPLOAD_DIR / "%(id)s.%(ext)s")
    progress("Ingesting", "Reading video metadata from YouTube", 3)
    options = {
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=True)
        candidate = Path(downloader.prepare_filename(info))
        if not candidate.is_file():
            requested = info.get("requested_downloads") or []
            candidate_path = requested[0].get("filepath") if requested else None
            candidate = Path(candidate_path) if candidate_path else candidate
        if not candidate.is_file():
            merged = candidate.with_suffix(".mp4")
            if merged.is_file():
                candidate = merged
        if not candidate.is_file():
            raise RuntimeError("yt-dlp finished without producing a local video file.")
        return candidate.resolve(), str(info.get("title") or candidate.stem)


def _transcribe(
    source: Path, duration: float, progress: ProgressCallback
) -> list[dict[str, Any]]:
    model_path = Path(
        os.environ.get("WHISPER_MODEL_PATH", str(ROOT / "models" / "whisper-small"))
    ).expanduser()
    if not model_path.is_dir():
        raise RuntimeError(
            "Local Whisper model files are missing. Place a CTranslate2 Whisper model "
            f"at {model_path} or set WHISPER_MODEL_PATH. The app will not download models."
        )
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError("faster-whisper is not installed in the active virtualenv.") from error

    progress("Transcribing", "Running faster-whisper on CPU (int8)", 15)
    model = WhisperModel(
        str(model_path),
        device="cpu",
        compute_type="int8",
        cpu_threads=max(1, os.cpu_count() or 1),
        local_files_only=True,
    )
    segments, _info = model.transcribe(
        str(source),
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
    )
    transcript: list[dict[str, Any]] = []
    last_progress = 15.0
    for segment in segments:
        words = getattr(segment, "words", None) or []
        if words:
            for word in words:
                token = str(word.word).strip()
                if token:
                    transcript.append(
                        {
                            "word": token,
                            "start": float(word.start),
                            "end": float(word.end),
                        }
                    )
        else:
            fallback_words = words_from_caption(
                segment.text, segment.end - segment.start
            )
            for item in fallback_words:
                item["start"] += float(segment.start)
                item["end"] += float(segment.start)
            transcript.extend(fallback_words)
        segment_progress = min(
            39.0, 15.0 + max(0.0, float(segment.end)) / max(0.1, duration) * 24.0
        )
        if segment_progress - last_progress >= 1.0:
            progress(
                "Transcribing",
                f"Transcribed through {min(float(segment.end), duration):.1f}s of "
                f"{duration:.1f}s",
                segment_progress,
            )
            last_progress = segment_progress
    return transcript


def _model_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                return str(text)
    raise RuntimeError("Gemini returned no text content.")


def generate_with_fallback(
    client: Any, prompt: str, progress: ProgressCallback | None = None
) -> str:
    """Try available flash models in order, then known and error-recommended models."""
    candidates: list[str] = []
    if progress:
        progress("Detecting Viral Hooks", "Checking available Gemini Flash models", 42)
    try:
        for model in client.models.list(config={"page_size": 100}):
            name = str(getattr(model, "name", "")).removeprefix("models/")
            supported = getattr(model, "supported_actions", None)
            if not name or (supported and "generateContent" not in supported):
                continue
            lowered = name.lower()
            if "flash" not in lowered or any(
                excluded in lowered for excluded in ("image", "audio", "tts", "live", "embedding")
            ):
                continue
            candidates.append(name)
    except Exception as error:
        if progress:
            progress(
                "Detecting Viral Hooks",
                f"Model discovery unavailable; trying configured fallbacks ({error})",
                44,
            )

    def priority(name: str) -> tuple[int, str]:
        match = re.search(r"gemini-(\d+)", name.lower())
        major = int(match.group(1)) if match else 0
        return (-major, name)

    candidates.sort(key=priority)
    candidates.extend(
        ["gemini-3.0-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"]
    )
    tried: set[str] = set()
    errors: list[str] = []
    position = 0
    while position < len(candidates):
        model_name = candidates[position]
        position += 1
        if model_name in tried:
            continue
        tried.add(model_name)
        try:
            if progress:
                progress("Detecting Viral Hooks", f"Analyzing transcript with {model_name}", 48)
            response = client.models.generate_content(model=model_name, contents=prompt)
            return _model_text(response)
        except Exception as error:
            message = str(error)
            errors.append(f"{model_name}: {message}")
            replacements = re.findall(r"(?:models/)?(gemini-[A-Za-z0-9._-]+)", message)
            for replacement in reversed(replacements):
                if replacement not in tried and replacement != model_name:
                    candidates.insert(position, replacement)
    raise RuntimeError(
        "Gemini model router exhausted its candidates. " + " | ".join(errors[-4:])
    )


def _parse_json_response(text: str) -> Any:
    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        stripped = match.group(1)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first >= 0 and last > first:
            try:
                return json.loads(stripped[first : last + 1])
            except json.JSONDecodeError as error:
                raise RuntimeError("Gemini returned invalid JSON for clip suggestions.") from error
        raise RuntimeError("Gemini returned no JSON clip suggestions.")


def _local_hooks(
    transcript: list[dict[str, Any]], duration: float, count: int
) -> list[dict[str, Any]]:
    if not transcript:
        clip_duration = min(45.0, duration)
        return [
            {
                "start": 0.0,
                "end": clip_duration,
                "title": "Source highlight",
                "reason": "Transcript unavailable; preview the source and refine this segment.",
                "score": 50,
                "caption": "",
            }
        ]
    words_per_clip = max(1, round(len(transcript) / max(1, count)))
    stride = max(1, round(words_per_clip * 0.8))
    clips: list[dict[str, Any]] = []
    for index in range(0, len(transcript), stride):
        group = transcript[index : index + words_per_clip]
        if not group:
            continue
        start = max(0.0, float(group[0]["start"]) - 1.0)
        end = min(duration, float(group[-1]["end"]) + 1.0)
        if end - start < 4 and duration > 4:
            end = min(duration, start + 30)
        caption = " ".join(str(item["word"]) for item in group).strip()
        title = " ".join(caption.split()[:7]).strip(" .,!?")
        clips.append(
            {
                "start": start,
                "end": end,
                "title": title or f"Highlight {len(clips) + 1}",
                "reason": "Locally selected transcript segment; add a Gemini API key for AI hook analysis.",
                "score": min(95, 55 + len(clips) * 3),
                "caption": caption,
            }
        )
        if len(clips) >= count:
            break
    return clips


def _detect_hooks(
    transcript: list[dict[str, Any]],
    duration: float,
    count: int,
    progress: ProgressCallback,
) -> list[dict[str, Any]]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        progress(
            "Detecting Viral Hooks",
            "Gemini key not configured; selecting transcript highlights locally",
            45,
        )
        return _local_hooks(transcript, duration, count)
    from google import genai

    text = " ".join(
        f"[{word['start']:.2f}-{word['end']:.2f}] {word['word']}" for word in transcript
    )
    prompt = (
        "Identify strong short-form video hooks in this transcript. Return only JSON "
        'with {"clips":[{"start":number,"end":number,"title":string,'
        '"reason":string,"score":integer,"caption":string}]}. '
        f"Choose up to {count} clips, each 12 to 60 seconds when source length permits. "
        f"Keep all timestamps within 0 and {duration:.2f} seconds. "
        "Prioritize a compelling opening, surprising insight, clear payoff, or emotional moment. "
        "Transcript words include source timestamps:\n"
        + text[:120000]
    )
    response_text = generate_with_fallback(genai.Client(api_key=api_key), prompt, progress)
    parsed = _parse_json_response(response_text)
    suggestions = parsed.get("clips", []) if isinstance(parsed, dict) else parsed
    if not isinstance(suggestions, list):
        raise RuntimeError("Gemini clip suggestions must be a JSON array.")
    clips: list[dict[str, Any]] = []
    for item in suggestions:
        if not isinstance(item, dict):
            continue
        try:
            start = max(0.0, float(item["start"]))
            end = min(duration, float(item["end"]))
        except (KeyError, ValueError, TypeError):
            continue
        if end <= start:
            continue
        clips.append(
            {
                "start": start,
                "end": end,
                "title": str(item.get("title") or f"Hook {len(clips) + 1}")[:120],
                "reason": str(item.get("reason") or "")[:600],
                "score": min(100, max(0, int(item.get("score", 70)))),
                "caption": str(item.get("caption") or ""),
            }
        )
        if len(clips) >= count:
            break
    if not clips:
        progress(
            "Detecting Viral Hooks",
            "Gemini returned no valid timestamps; using local transcript highlights",
            50,
        )
        return _local_hooks(transcript, duration, count)
    return clips


def _caption_words(
    transcript: list[dict[str, Any]], start: float, end: float
) -> list[dict[str, Any]]:
    return [
        {
            "word": str(item["word"]),
            "start": max(0.0, float(item["start"]) - start),
            "end": min(end - start, float(item["end"]) - start),
        }
        for item in transcript
        if float(item["end"]) > start and float(item["start"]) < end
    ]


def _render_clip(
    source: Path,
    clip: dict[str, Any],
    project_id: str,
    clip_number: int,
    framing_mode: str,
    captions_enabled: bool,
    highlight_color: str,
    caption_text: str | None = None,
) -> dict[str, Any]:
    start = float(clip["start"])
    end = float(clip["end"])
    clip_id = str(clip.get("id") or f"clip-{clip_number + 1}")
    output_name = f"{project_id}-{clip_id}-{uuid.uuid4().hex[:6]}.mp4"
    output_path = OUTPUT_DIR / output_name
    ass_path: Path | None = None
    caption = caption_text if caption_text is not None else str(clip.get("caption", ""))
    word_data = clip.get("words") or []
    if caption_text is not None:
        word_data = words_from_caption(caption_text, end - start)
        clip["caption"] = caption_text
        clip["words"] = word_data
    elif not word_data:
        word_data = words_from_caption(caption, end - start)
    if captions_enabled and word_data:
        ass_path = OUTPUT_DIR / f"{project_id}-{clip_id}-{uuid.uuid4().hex[:6]}.ass"
        write_ass(word_data, ass_path, highlight_color)
    result = render_vertical_clip(
        source,
        output_path,
        start,
        end,
        framing_mode=framing_mode,
        caption_file=ass_path,
    )
    clip.update(
        {
            "id": clip_id,
            "file": output_name,
            "preview_url": f"/outputs/{output_name}",
            "duration": round(end - start, 2),
            "framing_mode": result["framing_mode"],
            "captions_enabled": captions_enabled,
            "caption": caption,
            "words": word_data,
        }
    )
    return clip


def process_video(
    source: str | Path,
    progress: ProgressCallback,
    target_count: int = 5,
    framing_mode: str = "face",
    captions_enabled: bool = True,
    highlight_color: str = "yellow",
    source_title: str | None = None,
) -> dict[str, Any]:
    """Ingest, transcribe, identify candidate hooks, and render a new project."""
    progress("Ingesting", "Preparing source media", 0)
    source_text = str(source)
    if source_text.startswith(("http://", "https://")):
        local_source, downloaded_title = _ingest_youtube(source_text, progress)
    else:
        local_source = Path(source_text).resolve()
        downloaded_title = local_source.stem
    if not local_source.is_file():
        raise FileNotFoundError(f"Source media file not found: {local_source}")
    if not local_source.is_relative_to(ROOT):
        adopted_source = UPLOAD_DIR / f"{uuid.uuid4().hex}{local_source.suffix.lower()}"
        shutil.copy2(local_source, adopted_source)
        local_source = adopted_source
    metadata = _ffprobe(local_source)
    if not metadata["has_video"]:
        raise RuntimeError("The selected source does not contain a video stream.")
    title = source_title or metadata.get("title") or downloaded_title
    duration = metadata["duration"]
    progress("Ingesting", "Local source metadata verified", 10)
    project_id = uuid.uuid4().hex

    thumbnail_name = f"{project_id}-source.jpg"
    thumbnail_path = OUTPUT_DIR / thumbnail_name
    thumb_result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{min(1.0, duration / 2):.2f}",
            "-i",
            str(local_source),
            "-frames:v",
            "1",
            "-vf",
            "scale=480:-2",
            str(thumbnail_path),
        ],
        capture_output=True,
        text=True,
    )
    if thumb_result.returncode:
        raise RuntimeError(f"Could not create source thumbnail: {thumb_result.stderr[-1000:]}")

    transcript = _transcribe(local_source, duration, progress)
    progress("Transcribing", f"Captured {len(transcript)} timestamped words", 40)
    progress("Detecting Viral Hooks", "Finding high-retention segments", 42)
    clips = _detect_hooks(
        transcript, duration, min(12, max(1, int(target_count))), progress
    )
    for index, clip in enumerate(clips):
        clip["id"] = f"clip-{index + 1}"
        clip["words"] = _caption_words(transcript, clip["start"], clip["end"])
        if not clip.get("caption"):
            clip["caption"] = " ".join(word["word"] for word in clip["words"])

    project = {
        "id": project_id,
        "title": title,
        "source_title": title,
        "source_path": _relative(local_source),
        "duration": round(duration, 3),
        "thumbnail_url": f"/outputs/{thumbnail_name}",
        "preview_url": f"/outputs/{thumbnail_name}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "transcript": transcript,
        "clips": clips,
    }
    _save_project(project)
    for index, clip in enumerate(project["clips"]):
        percent = 55 + 42 * (index / max(1, len(project["clips"])))
        progress("Face Tracking & Reframing", f"Rendering clip {index + 1}/{len(clips)}", percent)
        _render_clip(
            local_source,
            clip,
            project_id,
            index,
            framing_mode,
            captions_enabled,
            highlight_color,
        )
        _save_project(project)
    progress("Rendering Captions", "Writing final captions and updating project archive", 99)
    _save_project(project)
    return project


def refine_clip(
    project_id: str,
    clip_id: str,
    start: float | None = None,
    end: float | None = None,
    framing_mode: str | None = None,
    captions_enabled: bool | None = None,
    highlight_color: str = "yellow",
    caption_text: str | None = None,
) -> dict[str, Any]:
    project = get_project(project_id)
    clip = next((item for item in project["clips"] if item.get("id") == clip_id), None)
    if clip is None:
        raise KeyError(f"Clip not found: {clip_id}")
    source = (ROOT / project["source_path"]).resolve()
    if not source.is_relative_to(ROOT) or not source.is_file():
        raise RuntimeError("The project's local source file is missing or outside this studio.")
    original_start, original_end = float(clip["start"]), float(clip["end"])
    new_start = original_start if start is None else float(start)
    new_end = original_end if end is None else float(end)
    if new_start < 0 or new_end <= new_start or new_end > project["duration"]:
        raise ValueError("Trim bounds must fit the source duration and end after start.")
    clip["start"], clip["end"] = new_start, new_end
    clip["duration"] = round(new_end - new_start, 2)
    clip["words"] = _caption_words(project.get("transcript", []), new_start, new_end)
    _render_clip(
        source,
        clip,
        project_id,
        project["clips"].index(clip),
        framing_mode or clip.get("framing_mode", "face"),
        clip.get("captions_enabled", True)
        if captions_enabled is None
        else captions_enabled,
        highlight_color,
        caption_text,
    )
    _save_project(project)
    return {"project": project, "clip": clip}


def source_file_from_relative(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("Source path must point to a local file inside this studio.")
    return path
