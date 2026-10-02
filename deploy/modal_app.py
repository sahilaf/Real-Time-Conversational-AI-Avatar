"""Deploy the whole Alapon pipeline to Modal.

    modal deploy deploy/modal_app.py

Two pieces, one Modal app:

* ``Pipeline`` - one GPU container running the SyncTalk_2D avatar server and
  the LiveKit agent side by side. The agent reaches the avatar on 127.0.0.1,
  exactly as on the laptop, so the audio -> frames round trip never leaves
  the machine. The agent dials out to LiveKit Cloud; nothing here needs an
  inbound port.
* ``web`` - the Flask site (landing page, /demo, /token) on a small CPU
  container. It wakes the GPU container when someone opens the demo and keeps
  it awake while the page is open, so an idle deployment costs nothing.

LiveKit itself stays on LiveKit Cloud: the media server needs UDP, which
Modal containers cannot accept.

Settings, read when you run ``modal deploy`` (see deploy/README.md):

    ALAPON_GPU         GPU type                       default T4
    ALAPON_KEEP_WARM   1 = keep the GPU up 24/7       default 0
    ALAPON_IDLE_SECS   idle time before GPU sleeps    default 600
    ALAPON_REGION      pin the GPU to a region        default any
"""

import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import modal
import modal.experimental

APP_NAME = "alapon"
SECRET_NAME = "alapon-secrets"

GPU = os.environ.get("ALAPON_GPU", "T4")
KEEP_WARM = int(os.environ.get("ALAPON_KEEP_WARM", "0"))
IDLE_SECS = int(os.environ.get("ALAPON_IDLE_SECS", "600"))
REGION = os.environ.get("ALAPON_REGION") or None

REPO = Path(__file__).resolve().parent.parent
SYNCTALK = REPO / "SyncTalk_2D"
CHECKPOINT = SYNCTALK / "checkpoint" / "alapon"
DATASET = SYNCTALK / "dataset" / "redwan"

# Container layout. The avatar server resolves model/checkpoints relative to
# its working directory and idle_cache relative to its own file, so both sit
# next to the code in /root/synctalk.
SYNCTALK_DIR = "/root/synctalk"
ASSETS_DIR = "/assets"
AGENT_DIR = "/root/agent"
FRONTEND_DIR = "/root/frontend"
AGENT_PYTHON = "/opt/agent/bin/python"

AVATAR_PORT = 5001
AVATAR_BASE = f"http://127.0.0.1:{AVATAR_PORT}"

app = modal.App(APP_NAME)
secret = modal.Secret.from_name(SECRET_NAME)

# ---------------------------------------------------------------------------
# GPU image: avatar server on the system Python, agent in its own venv
# ---------------------------------------------------------------------------

pipeline_image = (
    # Matches the laptop's conda env: Python 3.10, torch 2.2.0 on CUDA 12.1.
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.2.0",
        "torchvision==0.17.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.15.3",
        "librosa==0.10.0",
        "numba==0.63.1",
        "soundfile==0.13.1",
        # 4.10 is the last headless build that still accepts numpy 1.x
        "opencv-python-headless==4.10.0.84",
        "fastapi==0.128.0",
        "uvicorn==0.40.0",
        "wsproto==1.3.2",
    )
    # The agent wants numpy 2 and newer packages throughout, so it gets a
    # separate interpreter rather than a compromise between the two.
    .add_local_file(REPO / "deploy" / "agent-requirements.txt", "/tmp/agent-requirements.txt", copy=True)
    .run_commands(
        "pip install uv",
        "uv venv --python 3.13 /opt/agent",
        f"uv pip install --python {AGENT_PYTHON} -r /tmp/agent-requirements.txt",
    )
    # ~1.4 GB, baked into the image rather than read from a Volume: the server
    # loads source frames on demand while it talks, so they need local-disk
    # latency. Uploaded once; later deploys reuse the cached layer.
    .add_local_file(CHECKPOINT / "59.pth", f"{ASSETS_DIR}/alapon/59.pth", copy=True)
    # Holds mask_version. Without it the server silently falls back to the
    # legacy mouth mask - the one Alapon was trained to replace.
    .add_local_file(CHECKPOINT / "train_config.json", f"{ASSETS_DIR}/alapon/train_config.json", copy=True)
    .add_local_dir(DATASET / "full_body_img", f"{ASSETS_DIR}/redwan/full_body_img", copy=True)
    .add_local_dir(DATASET / "landmarks", f"{ASSETS_DIR}/redwan/landmarks", copy=True)
    .add_local_dir(SYNCTALK / "model" / "checkpoints", f"{SYNCTALK_DIR}/model/checkpoints", copy=True)
    .add_local_dir(SYNCTALK / "idle_cache", f"{SYNCTALK_DIR}/idle_cache", copy=True)
    # Code last and mounted rather than copied, so editing it redeploys in
    # seconds without rebuilding the layers above.
    .add_local_file(SYNCTALK / "avatar_server_ws.py", f"{SYNCTALK_DIR}/avatar_server_ws.py")
    .add_local_file(SYNCTALK / "unet_328.py", f"{SYNCTALK_DIR}/unet_328.py")
    .add_local_file(SYNCTALK / "utils.py", f"{SYNCTALK_DIR}/utils.py")
    .add_local_file(REPO / "agent" / "agent_bangla.py", f"{AGENT_DIR}/agent_bangla.py")
)


def _pump(proc, name, markers=None):
    """Copy a child's output into the container log; set each event in
    ``markers`` ({text: Event}) once its text appears."""
    markers = markers or {}
    for line in proc.stdout:
        print(f"[{name}] {line}", end="", flush=True)
        for text, event in markers.items():
            if text in line:
                event.set()


def _avatar_healthy():
    try:
        with urllib.request.urlopen(f"{AVATAR_BASE}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _warm_frames(root):
    """Read every source frame once so the first sentence isn't served from a
    cold image layer. Runs in the background; the server works without it."""
    started = time.time()
    total = 0
    for path in Path(root).iterdir():
        try:
            total += len(path.read_bytes())
        except OSError:
            pass
    print(f"[pipeline] warmed {total / 1e9:.2f} GB of frames in {time.time() - started:.0f}s", flush=True)


@app.cls(
    image=pipeline_image,
    gpu=GPU,
    cpu=4,
    memory=8192,
    secrets=[secret],
    min_containers=KEEP_WARM,
    # One container, one avatar: sessions and the idle cache live in its
    # memory. Raise this only if you expect parallel conversations - each
    # container carries its own agent + avatar pair, so they scale together.
    max_containers=1,
    scaledown_window=IDLE_SECS,
    # Covers a cold start: model load, idle cache and agent registration.
    timeout=900,
    region=REGION,
)
@modal.concurrent(max_inputs=50)
class Pipeline:
    @modal.enter()
    def start(self):
        self.procs = {}
        threading.Thread(
            target=_warm_frames, args=(f"{ASSETS_DIR}/redwan/full_body_img",), daemon=True
        ).start()

        # 1. Avatar server. Same command as the laptop, with container paths.
        avatar = subprocess.Popen(
            [
                sys.executable, "-u", "avatar_server_ws.py",
                "--checkpoint", f"{ASSETS_DIR}/alapon/59.pth",
                "--dataset", f"{ASSETS_DIR}/redwan",
                "--mode", "ave",
                "--host", "127.0.0.1",
                "--port", str(AVATAR_PORT),
            ],
            cwd=SYNCTALK_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self.procs["avatar"] = avatar
        threading.Thread(target=_pump, args=(avatar, "avatar"), daemon=True).start()

        deadline = time.time() + 600
        while not _avatar_healthy():
            if avatar.poll() is not None:
                raise RuntimeError(f"avatar server exited during startup (code {avatar.returncode})")
            if time.time() > deadline:
                raise RuntimeError("avatar server did not become healthy within 10 minutes")
            time.sleep(1)
        print("[pipeline] avatar server healthy", flush=True)

        # 2. Agent, in production mode, pointed at the local avatar. Ready means
        # registered with LiveKit AND a job process has started: registration
        # alone passed while every job process was crashing on import, which
        # left rooms with no agent and nothing in the health check.
        env = dict(os.environ, AVATAR_BASE=AVATAR_BASE, PYTHONUNBUFFERED="1")
        agent = subprocess.Popen(
            [AGENT_PYTHON, "agent_bangla.py", "start"],
            cwd=AGENT_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self.procs["agent"] = agent
        registered = threading.Event()
        job_ready = threading.Event()
        threading.Thread(
            target=_pump,
            args=(agent, "agent", {"registered worker": registered, "process initialized": job_ready}),
            daemon=True,
        ).start()

        deadline = time.time() + 180
        while not (registered.is_set() and job_ready.is_set()):
            if agent.poll() is not None:
                raise RuntimeError(f"agent exited during startup (code {agent.returncode})")
            if time.time() > deadline:
                missing = "registration with LiveKit" if not registered.is_set() else "a working job process"
                raise RuntimeError(f"agent startup timed out waiting for {missing} - see [agent] log lines")
            time.sleep(1)
        print("[pipeline] agent registered with LiveKit - ready", flush=True)
        self.ready_at = time.time()

    @modal.method()
    def status(self):
        """Health check; calling it also starts the container and keeps it up."""
        dead = {name: p.returncode for name, p in self.procs.items() if p.poll() is not None}
        if dead:
            # A container whose agent or avatar has died cannot serve anyone.
            # Stop taking calls so Modal replaces it with a fresh one.
            try:
                modal.experimental.stop_fetching_inputs()
            except Exception:
                pass
            raise RuntimeError(f"pipeline process exited: {dead}")
        return {"ready": True, "uptime_s": round(time.time() - self.ready_at)}

    @modal.exit()
    def stop(self):
        for proc in self.procs.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in self.procs.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


# ---------------------------------------------------------------------------
# Website: landing page, /demo and the LiveKit token endpoint
# ---------------------------------------------------------------------------

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flask==3.1.2", "flask-cors==6.0.1", "livekit-api==1.2.0", "python-dotenv==1.2.1")
    .add_local_dir(
        REPO / "frontend",
        FRONTEND_DIR,
        # .env stays local - on Modal the keys come from the secret
        ignore=["venv", ".env", "**/__pycache__", "**/node_modules", "**/.claude"],
    )
)


@app.function(image=web_image, secrets=[secret], scaledown_window=300)
@modal.concurrent(max_inputs=100)
@modal.wsgi_app()
def web():
    from flask import jsonify, request

    sys.path.insert(0, FRONTEND_DIR)
    import main  # frontend/main.py - the same Flask app you run locally

    site = main.app

    def wake():
        return Pipeline().status.remote()

    @site.post("/api/wake")
    def api_wake():
        # Blocks through a cold start (a minute or two), then answers fast.
        try:
            return jsonify(wake())
        except Exception as exc:
            return jsonify({"ready": False, "error": str(exc)}), 503

    @site.before_request
    def ensure_pipeline_before_token():
        # No agent is registered while the GPU sleeps, so a room created then
        # would sit empty. Hold the token until the pipeline is up.
        if request.path == "/token" and request.method == "POST":
            try:
                wake()
            except Exception as exc:
                return jsonify({"error": f"Avatar pipeline unavailable: {exc}"}), 503
        return None

    return site
