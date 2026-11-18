#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x "venv/bin/python" ]]; then
  echo "Missing venv. Create it with Python 3.12 and install requirements.txt first." >&2
  exit 1
fi

source venv/bin/activate
set -a
if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi
set +a

url="http://127.0.0.1:5000"
if curl --silent --fail "$url/api/health" >/dev/null; then
  if ! xdg-open "$url" >/dev/null 2>&1; then
    echo "Open $url in your browser." >&2
  fi
  echo "Cutline is already running at $url"
  exit 0
fi

"$VIRTUAL_ENV/bin/python" app.py &
server_pid=$!
cleanup() {
  kill "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 50); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
    exit $?
  fi
  if curl --silent --fail "$url/api/health" >/dev/null; then
    ready=1
    break
  fi
  sleep 0.2
done

if [[ "$ready" -ne 1 ]]; then
  echo "The local server did not become ready at $url." >&2
  exit 1
fi

xdg-open "$url" >/dev/null 2>&1 || echo "Open $url in your browser."
echo "Cutline is running at $url. Press Ctrl+C to stop."
wait "$server_pid"
