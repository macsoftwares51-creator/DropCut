"""Loopback-only Flask API for the local video clipping studio."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import queue
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_file, send_from_directory, stream_with_context
from werkzeug.utils import secure_filename
from waitress import serve

import pipeline


ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".mov"}
load_dotenv(ROOT / ".env")

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024 * 1024

_jobs_lock = threading.RLock()
_jobs: dict[str, dict[str, Any]] = {}


def _snapshot(job: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in job.items()
        if key not in {"queue"}
    }


def _update_job(
    job_id: str, stage: str, message: str, percent: float, state: str = "running"
) -> None:
    percent = min(100.0, max(0.0, float(percent)))
    with _jobs_lock:
        job = _jobs[job_id]
        job.update(
            {
                "stage": stage,
                "message": message,
                "progress": round(percent, 1),
                "state": state,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        job["queue"].put(_snapshot(job))


def _run_job(job_id: str, source: str, options: dict[str, Any], source_title: str | None) -> None:
    try:
        project = pipeline.process_video(
            source=source,
            progress=lambda stage, message, value: _update_job(
                job_id, stage, message, value
            ),
            target_count=options.get("target_count", 5),
            framing_mode=options.get("framing_mode", "face"),
            captions_enabled=options.get("captions_enabled", True),
            highlight_color=options.get("highlight_color", "yellow"),
            source_title=source_title,
        )
    except Exception as error:
        _update_job(job_id, "Error", str(error), 100, state="failed")
        return
    with _jobs_lock:
        _jobs[job_id]["project_id"] = project["id"]
    _update_job(job_id, "Complete", "Project is ready in the archive.", 100, state="done")


def _start_job(source: str, options: dict[str, Any], title: str | None) -> str:
    job_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "state": "queued",
            "stage": "Queued",
            "message": "Waiting for the local media worker.",
            "progress": 0.0,
            "created_at": now,
            "updated_at": now,
            "queue": queue.Queue(),
        }
    worker = threading.Thread(
        target=_run_job, args=(job_id, source, options, title), daemon=True
    )
    worker.start()
    return job_id


@app.get("/")
def studio() -> Response:
    return send_file(ROOT / "index.html")


@app.get("/api/health")
def health() -> Response:
    model_path = Path(
        os.environ.get("WHISPER_MODEL_PATH", str(ROOT / "models" / "whisper-small"))
    ).expanduser()
    return jsonify(
        {
            "status": "ok",
            "local_only": True,
            "engines": {
                "whisper_model": model_path.is_dir(),
                "ffmpeg": shutil.which("ffmpeg") is not None,
                "face_tracking": all(
                    importlib.util.find_spec(package) is not None
                    for package in ("mediapipe", "cv2")
                ),
            },
        }
    )


@app.get("/api/projects")
def projects() -> Response:
    try:
        return jsonify({"projects": pipeline.list_projects()})
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 500


@app.get("/api/projects/<project_id>")
def project(project_id: str) -> Response:
    try:
        return jsonify({"project": pipeline.get_project(project_id)})
    except KeyError as error:
        return jsonify({"error": str(error)}), 404
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 500


@app.post("/api/process")
def process() -> Response:
    options: dict[str, Any]
    source_title: str | None = None
    if "file" in request.files:
        uploaded = request.files["file"]
        if not uploaded.filename:
            return jsonify({"error": "Choose a video file to upload."}), 400
        filename = secure_filename(uploaded.filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return jsonify({"error": "Only MP4, MKV, and MOV uploads are supported."}), 400
        source_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        uploaded.save(source_path)
        source = str(source_path)
        source_title = Path(filename).stem
        options = {
            "target_count": request.form.get("target_count", 5),
            "framing_mode": request.form.get("framing_mode", "face"),
            "captions_enabled": request.form.get("captions_enabled", "true").lower()
            in {"true", "1", "yes", "on"},
            "highlight_color": request.form.get("highlight_color", "yellow"),
        }
    else:
        data = request.get_json(silent=True) or {}
        source = str(data.get("url") or data.get("source") or "").strip()
        if not source:
            return jsonify({"error": "Provide a YouTube URL or upload a local video."}), 400
        options = {
            "target_count": data.get("target_count", 5),
            "framing_mode": data.get("framing_mode", "face"),
            "captions_enabled": data.get("captions_enabled", True),
            "highlight_color": data.get("highlight_color", "yellow"),
        }
    try:
        options["target_count"] = min(12, max(1, int(options["target_count"])))
        if options["framing_mode"] not in {"face", "blur", "center"}:
            raise ValueError("Invalid framing mode.")
        if options["highlight_color"] not in {"yellow", "cyan", "green", "white"}:
            raise ValueError("Invalid caption highlight color.")
    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400
    if source.startswith(("http://", "https://")):
        from urllib.parse import urlparse

        hostname = (urlparse(source).hostname or "").lower()
        if hostname not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}:
            return jsonify({"error": "Only YouTube URLs can be processed online."}), 400
    job_id = _start_job(source, options, source_title)
    return jsonify({"job_id": job_id, "progress_url": f"/api/progress/{job_id}"}), 202


@app.post("/api/refine")
def refine() -> Response:
    data = request.get_json(silent=True) or {}
    try:
        result = pipeline.refine_clip(
            project_id=str(data.get("project_id", "")),
            clip_id=str(data.get("clip_id", "")),
            start=float(data["start"]) if data.get("start") is not None else None,
            end=float(data["end"]) if data.get("end") is not None else None,
            framing_mode=data.get("framing_mode"),
            captions_enabled=data.get("captions_enabled"),
            highlight_color=str(data.get("highlight_color", "yellow")),
            caption_text=data.get("caption_text"),
        )
        return jsonify(result)
    except KeyError as error:
        return jsonify({"error": str(error)}), 404
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    except (RuntimeError, OSError) as error:
        return jsonify({"error": str(error)}), 500


@app.get("/api/progress")
def progress_overview() -> Response:
    with _jobs_lock:
        jobs = [_snapshot(job) for job in _jobs.values()]
    return jsonify({"jobs": jobs})


@app.get("/api/progress/<job_id>")
def progress_stream(job_id: str) -> Response:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job ID."}), 404
        updates: queue.Queue[dict[str, Any]] = job["queue"]

    @stream_with_context
    def generate():
        with _jobs_lock:
            current = _jobs.get(job_id)
            if current is None:
                return
            state = _snapshot(current)
            done = state["state"] in {"done", "failed"}
        yield f"data: {json.dumps(state)}\n\n"
        while not done:
            try:
                state = updates.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(state)}\n\n"
            done = state["state"] in {"done", "failed"}

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/outputs/<path:filename>")
def output_file(filename: str) -> Response:
    return send_from_directory(OUTPUT_DIR, filename, conditional=True)


if __name__ == "__main__":
    serve(app, host="127.0.0.1", port=5000, threads=8, ident="Cutline")
