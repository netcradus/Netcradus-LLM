"""
Netcradus LLM — Application Launcher.

Ensures the backend HTTP server is running, waits for it to become ready,
then opens the application in the default browser over HTTP.

Behavior:
  1. Probe http://localhost:8000/api/status.
  2. If it is not serving HTTP 200, automatically start the backend:
        python web_server.py --port 8000
  3. Poll /api/status until it returns HTTP 200 (or a timeout).
  4. Open http://localhost:8000/ in the default browser.
  5. The backend stays running in the background; it is NEVER opened via file://.

All navigation from the served HTML uses relative HTTP URLs, so once the
app is opened through the backend, every page (Home / Admin / User / Training)
is loaded over HTTP from the server instead of file://.

Usage:
    python launch.py           # default port 8000
    python launch.py --port 9000
"""

import argparse
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent)
WEB_SERVER = os.path.join(PROJECT_ROOT, "web_server.py")
READY_TIMEOUT = 60        # seconds to wait for /api/status before giving up
POLL_INTERVAL = 0.5       # seconds between readiness checks


def status_url(port: int) -> str:
    return f"http://localhost:{port}/api/status"


def is_backend_ready(port: int) -> bool:
    """Return True if GET /api/status responds with HTTP 200."""
    try:
        with urllib.request.urlopen(status_url(port), timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, ConnectionError, OSError):
        return False


def start_backend(port: int) -> subprocess.Popen:
    """Start the backend server as a detached background process."""
    log_path = os.path.join(PROJECT_ROOT, "server.log")
    log_file = open(log_path, "a", encoding="utf-8")

    # Use the same interpreter that is running this launcher so the command is
    # "python web_server.py --port <port>" regardless of platform.
    python = sys.executable or "python"
    cmd = [python, "-u", WEB_SERVER, "--port", str(port)]

    # start_new_session detaches the child so it survives the launcher exiting
    # and is not tied to this process's terminal/conda session.
    popen_kwargs = dict(
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        cwd=PROJECT_ROOT,
    )
    if os.name == "nt":
        # Windows: don't pop a console window, new process group.
        popen_kwargs["creationflags"] = (
            subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    print(f"[launcher] Started backend (pid={proc.pid}): {' '.join(cmd)}")
    print(f"[launcher] Backend logs: {log_path}")
    return proc


def wait_until_ready(port: int) -> bool:
    """Poll /api/status until it returns HTTP 200 or the timeout expires."""
    deadline = time.time() + READY_TIMEOUT
    while time.time() < deadline:
        if is_backend_ready(port):
            return True
        time.sleep(POLL_INTERVAL)
    return False


def open_browser(port: int) -> None:
    """Open the app in the default browser.

    Runs in a daemon thread so the launcher returns immediately (and never
    blocks on a headless machine where no browser is available).
    """
    url = f"http://localhost:{port}/"
    print(f"[launcher] Opening {url}")

    def _open_blocking() -> None:
        try:
            webbrowser.open(url, new=2)
        except Exception as exc:  # noqa: BLE001 - browser launch must never block the launcher
            print(f"[launcher] Could not open browser automatically ({exc}).")
            print(f"[launcher] Please open {url} manually in your browser.")

    threading.Thread(target=_open_blocking, name="open-browser", daemon=True).start()


def main() -> int:
    parser = argparse.ArgumentParser(description="Netcradus LLM application launcher")
    parser.add_argument("--port", type=int, default=8000, help="Backend port (default: 8000)")
    args = parser.parse_args()
    port = args.port

    print(f"[launcher] Checking backend at {status_url(port)} ...")

    if not is_backend_ready(port):
        print("[launcher] Backend not running. Starting it now ...")
        start_backend(port)
        print(f"[launcher] Waiting for backend to become ready (up to {READY_TIMEOUT}s) ...")
        if not wait_until_ready(port):
            print(f"[launcher] ERROR: backend did not become ready at {status_url(port)} within {READY_TIMEOUT}s.")
            print("[launcher] The backend log is at: server.log")
            return 1

    print(f"[launcher] Backend is ready (HTTP 200 from {status_url(port)}).")
    open_browser(port)
    print("[launcher] Done. The backend continues running in the background.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
