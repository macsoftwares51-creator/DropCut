# Cutline — Local Video Studio

Cutline is a loopback-only desktop studio for turning local videos or YouTube
videos into vertical short clips. Transcription, face tracking, rendering,
caption burning, file serving, and project storage run on this computer.
Network access is limited to YouTube ingestion and optional Gemini hook
analysis.

## Requirements
- Ubuntu Linux, Python 3.12, and FFmpeg with `ffmpeg`/`ffprobe` on `PATH`(if not installed run bash-sudo apt update && sudo apt install ffmpeg)
- A local CTranslate2 `faster-whisper` model directory (the app does not fetch
  model weights at runtime)
- Optional Gemini API key for online viral-hook analysis; without it, Cutline
  selects transcript highlights locally

The Python 3.12 runtime is recommended because PyAV below 13 has no reliable
build against the system Python 3.14/FFmpeg 8 combination. Create the virtual
environment and install dependencies with:

```bash
uv venv --python 3.12 venv
venv/bin/python -m ensurepip --upgrade
venv/bin/python -m pip install -r requirements.txt
```

Put a CTranslate2-compatible Whisper model at `models/whisper-small`, or set
`WHISPER_MODEL_PATH` in `.env` to a different local model directory. Copy
`.env.example` to `.env` and set `GEMINI_API_KEY` only when Gemini analysis is
wanted. All model files stay local; missing Whisper files are reported instead
of being downloaded.

## Launch

```bash
chmod +x run_app.sh
./run_app.sh
```

The launcher binds the Flask application through Waitress to
`127.0.0.1:5000`, loads `.env`, opens the studio in the default browser, and
keeps the server in the foreground until stopped.
Local MP4, MKV, and MOV uploads work without internet. YouTube downloads and
Gemini analysis need internet access.

## Local interfaces

- `GET /api/projects` — saved projects
- `POST /api/process` — YouTube URL JSON or `multipart/form-data` local file
- `GET /api/progress` and `GET /api/progress/<job_id>` — job status and SSE
- `POST /api/refine` — trim, change framing/captions, and re-render a clip
- `GET /outputs/<filename>` — rendered clips and project thumbnails
- `python mcp_server.py` — optional stdio MCP server with process, list, and
  re-render tools

The app stores sources in `uploads/`, renders and thumbnails in `outputs/`,
and project metadata/transcripts in `projects.json`. Do not expose the Flask
port beyond loopback: this local studio intentionally has no account system.
