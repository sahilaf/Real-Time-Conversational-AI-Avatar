# Deploying Alapon to Modal

One command deploys the whole pipeline except LiveKit, which stays on
LiveKit Cloud (its media server needs UDP, which Modal can't accept).

```
browser ──► web (Modal, CPU)         landing page, /demo, /token
   │            │  wakes ▼
   │        Pipeline (Modal, GPU)    avatar server + LiveKit agent, same container
   │            │  dials out ▼
   └──────► LiveKit Cloud ◄──────── agent joins the room, publishes video + audio
```

The GPU container sleeps when nobody is using it. Opening `/demo` wakes it
(a cold start takes a minute or two) and the page keeps it awake while it is
open; about 10 minutes after the last visitor leaves, it shuts down again.

## One-time setup

From the repo root, in PowerShell:

```powershell
pip install modal
modal setup                     # opens the browser to log in
modal secret create alapon-secrets --from-dotenv agent/.env
```

The secret needs `GOOGLE_API_KEY`, `LIVEKIT_URL`, `LIVEKIT_API_KEY` and
`LIVEKIT_API_SECRET` — `agent/.env` already has all four. `.env` files are
never uploaded; the containers read the keys from this secret.

## Deploy

```powershell
modal deploy deploy/modal_app.py
```

The first deploy uploads about 1.4 GB (the Alapon checkpoint and the `redwan`
frames and landmarks) and builds the image, so it takes a while. Later deploys
reuse that layer; code-only changes redeploy in seconds.

The command prints the site URL, `https://<you>--alapon-web.modal.run`. The
landing page is at `/`, the demo at `/demo`.

## Settings

Set these before `modal deploy`; they are read at deploy time.

| Variable | Default | What it does |
| --- | --- | --- |
| `ALAPON_GPU` | `T4` | GPU type, e.g. `L4` or `A10G` for more headroom |
| `ALAPON_KEEP_WARM` | `0` | `1` keeps one GPU container up permanently — no cold start, but billed around the clock |
| `ALAPON_IDLE_SECS` | `600` | How long the GPU stays up after the last visitor |
| `ALAPON_REGION` | any | Pin the GPU near your LiveKit Cloud region to cut latency |

For the defence, keep it warm so the first connection is instant:

```powershell
$env:ALAPON_KEEP_WARM = "1"; modal deploy deploy/modal_app.py
```

and afterwards redeploy without it (or `modal app stop alapon`) so it stops
billing.

## Operating it

```powershell
modal app logs alapon           # live logs: [avatar] and [agent] lines
modal app stop alapon           # take everything down
```

A healthy start logs, in order: `[pipeline] avatar server healthy`, then
`[pipeline] agent registered with LiveKit - ready`.

**Stop your local agent while the Modal one is deployed.** The agent takes
every new room in the LiveKit project, so a laptop agent (`python
agent_bangla.py dev`) and the Modal agent would both try to join the same
conversation. Use a separate LiveKit project for local development if you
want both running.

## What runs where

| | Local | Modal |
| --- | --- | --- |
| Avatar server | `SyncTalk_2D/avatar_server_ws.py`, conda env | same file, Python 3.10 + torch 2.2 (cu121) |
| Agent | `agent/agent_bangla.py dev`, venv | same file in `start` mode, own Python 3.13 venv (`deploy/agent-requirements.txt`) |
| Website | `frontend/main.py` | same Flask app, plus `/api/wake` and a wait before `/token` |

The avatar and agent keep separate interpreters because they need different
numpy majors (1.x under torch 2.2, 2.x for the agent).
