"""
synctalk_agent.py — FIXED (smooth + ordered + no deadlock)

Key fixes:
- Proper prebuffer (store jobs; do NOT skip task_done / do NOT discard jobs)
- SyncTalk concurrency limited (max 1 in-flight request) to avoid GPU overload
- MJPEG decode runs in a thread pool (no asyncio blocking)
- Audio never waits for video; video repeats last frame if missing
"""

import asyncio
import io
import os
import time
import logging
import inspect
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Deque
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import requests
import numpy as np
import soundfile as sf
import cv2

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents import Agent, AgentSession
from livekit.agents.voice import room_io
from livekit.plugins import google

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

# ----------------------------
# CONFIG
# ----------------------------
SYNCTALK_BASE = os.getenv("SYNCTALK_BASE", "http://127.0.0.1:5001")
SYNCTALK_URL = os.getenv("SYNCTALK_URL", f"{SYNCTALK_BASE}/stream_frames")

CHUNK_SECONDS = float(os.getenv("CHUNK_SECONDS", "1.0"))
MIN_SEND_S = float(os.getenv("MIN_SEND_S", "0.35"))
TARGET_FPS = int(os.getenv("TARGET_FPS", "20"))
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "300"))

ASSISTANT_SR = int(os.getenv("ASSISTANT_SR", "24000"))
PUBLISH_SR = ASSISTANT_SR
PUBLISH_CH = 1

SYNC_START_DELAY = float(os.getenv("SYNC_START_DELAY", "1.5"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "75"))

PREBUFFER_JOBS = int(os.getenv("PREBUFFER_JOBS", "3"))
FRAME_WAIT = float(os.getenv("FRAME_WAIT", "0.6"))

RAW_JOB_Q_MAX = int(os.getenv("RAW_JOB_Q_MAX", "100"))
SYNCTALK_JOB_Q_MAX = int(os.getenv("SYNCTALK_JOB_Q_MAX", "4"))
PUBLISH_JOB_Q_MAX = int(os.getenv("PUBLISH_JOB_Q_MAX", "300"))

FRAME_STORE_MAX = int(os.getenv("FRAME_STORE_MAX", "4000"))

VIDEO_TRACK_NAME = os.getenv("VIDEO_TRACK_NAME", "agent_video")
AUDIO_TRACK_NAME = os.getenv("AUDIO_TRACK_NAME", "agent_audio")

# ----------------------------
# QueueAudioOutput import (version-safe)
# ----------------------------
QueueAudioOutput = None
AudioSegmentEnd = None

def import_queue_audio_output():
    global QueueAudioOutput, AudioSegmentEnd
    try:
        from livekit.agents.voice.avatar import QueueAudioOutput as QAO, AudioSegmentEnd as ASE
        QueueAudioOutput, AudioSegmentEnd = QAO, ASE
        return
    except Exception:
        pass
    from livekit.agents.voice.avatar._queue_io import QueueAudioOutput as QAO
    from livekit.agents.voice.avatar._types import AudioSegmentEnd as ASE
    QueueAudioOutput, AudioSegmentEnd = QAO, ASE


# ----------------------------
# Track creation (compat)
# ----------------------------
def create_local_video_track(name: str, source: rtc.VideoSource) -> rtc.LocalVideoTrack:
    if hasattr(rtc.LocalVideoTrack, "create_video_track"):
        return rtc.LocalVideoTrack.create_video_track(name, source)
    return rtc.LocalVideoTrack.create(name, source)

def create_local_audio_track(name: str, source: rtc.AudioSource) -> rtc.LocalAudioTrack:
    if hasattr(rtc.LocalAudioTrack, "create_audio_track"):
        return rtc.LocalAudioTrack.create_audio_track(name, source)
    return rtc.LocalAudioTrack.create(name, source)


# ----------------------------
# Helpers
# ----------------------------
async def capture_maybe_async(source, frame):
    res = source.capture_frame(frame)
    if inspect.isawaitable(res):
        await res

async def sleep_until(t: float):
    now = time.monotonic()
    if t > now:
        await asyncio.sleep(t - now)

def ensure_mono_i16(x: np.ndarray) -> np.ndarray:
    x = x.reshape(-1)
    if x.dtype != np.int16:
        x = x.astype(np.int16, copy=False)
    return x

def wav_bytes_from_pcm16(pcm_i16: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, pcm_i16, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

async def put_drop_oldest(q: asyncio.Queue, item):
    if q.full():
        try:
            _ = q.get_nowait()
        except Exception:
            pass
    q.put_nowait(item)


# ----------------------------
# PCM chunker
# ----------------------------
class PCMChunker:
    def __init__(self, sr: int, chunk_seconds: float):
        self.sr = sr
        self.target = int(sr * chunk_seconds)
        self.buf = np.zeros((0,), dtype=np.int16)

    def push(self, pcm_i16: np.ndarray):
        self.buf = np.concatenate([self.buf, pcm_i16])

    def pop_all_ready(self):
        outs = []
        while self.buf.shape[0] >= self.target:
            outs.append(self.buf[:self.target])
            self.buf = self.buf[self.target:]
        return outs

    def flush_tail(self) -> Optional[np.ndarray]:
        if self.buf.shape[0] == 0:
            return None
        out = self.buf
        self.buf = np.zeros((0,), dtype=np.int16)
        return out


# ----------------------------
# Job + FrameStore
# ----------------------------
@dataclass
class ChunkJob:
    job_id: int
    pcm: np.ndarray
    wav: bytes
    fps: int
    start_at: float


class FrameStore:
    def __init__(self, max_items: int):
        self.max_items = max_items
        self.store: Dict[Tuple[int, int], np.ndarray] = {}
        self.cv = asyncio.Condition()

    async def put(self, job_id: int, frame_idx: int, frame_bgr: np.ndarray):
        async with self.cv:
            if len(self.store) >= self.max_items:
                oldest = next(iter(self.store.keys()))
                self.store.pop(oldest, None)
            self.store[(job_id, frame_idx)] = frame_bgr
            self.cv.notify_all()

    async def get(self, job_id: int, frame_idx: int, timeout_s: float) -> Optional[np.ndarray]:
        end = time.monotonic() + timeout_s
        async with self.cv:
            while True:
                k = (job_id, frame_idx)
                if k in self.store:
                    return self.store.pop(k)
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return None
                try:
                    await asyncio.wait_for(self.cv.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return None


# ----------------------------
# MJPEG parsing
# ----------------------------
def mjpeg_iter_jpegs(resp):
    boundary = b"--frame"
    buf = b""
    for chunk in resp.iter_content(chunk_size=65536):
        if not chunk:
            continue
        buf += chunk
        while True:
            bidx = buf.find(boundary)
            if bidx == -1:
                if len(buf) > 2_000_000:
                    buf = buf[-1_000_000:]
                break

            hdr_end = buf.find(b"\r\n\r\n", bidx)
            if hdr_end == -1:
                break

            header = buf[bidx:hdr_end].decode("latin1", errors="ignore")
            clen = None
            for line in header.split("\r\n"):
                if line.lower().startswith("content-length:"):
                    try:
                        clen = int(line.split(":", 1)[1].strip())
                    except Exception:
                        clen = None

            if clen is None:
                buf = buf[hdr_end + 4:]
                continue

            data_start = hdr_end + 4
            if len(buf) < data_start + clen:
                break

            jpg = buf[data_start:data_start + clen]
            buf = buf[data_start + clen:]
            yield jpg


# ----------------------------
# Stage 1: Gemini PCM -> jobs
# ----------------------------
async def assistant_audio_to_jobs(queue_audio, out_queue: asyncio.Queue):
    logger.info("Collector started (QueueAudioOutput -> ChunkJobs).")
    chunker = PCMChunker(ASSISTANT_SR, CHUNK_SECONDS)

    playhead: Optional[float] = None
    job_id = 0

    async def emit(pcm_i16: np.ndarray):
        nonlocal playhead, job_id
        pcm_i16 = ensure_mono_i16(pcm_i16)
        dur_s = pcm_i16.shape[0] / ASSISTANT_SR
        if dur_s < MIN_SEND_S:
            return

        wav = wav_bytes_from_pcm16(pcm_i16, ASSISTANT_SR)

        if playhead is None:
            playhead = time.monotonic() + SYNC_START_DELAY

        job = ChunkJob(job_id=job_id, pcm=pcm_i16, wav=wav, fps=TARGET_FPS, start_at=playhead)
        job_id += 1
        playhead += dur_s

        if out_queue.full():
            try:
                _ = out_queue.get_nowait()
            except Exception:
                pass
        out_queue.put_nowait(job)

    async for item in queue_audio:
        if AudioSegmentEnd is not None and isinstance(item, AudioSegmentEnd):
            tail = chunker.flush_tail()
            if tail is not None and tail.size > 0:
                await emit(tail)
            continue

        pcm = np.frombuffer(bytes(item.data), dtype=np.int16)
        if pcm.size == 0:
            continue
        chunker.push(pcm)

        for chunk in chunker.pop_all_ready():
            await emit(chunk)


# ----------------------------
# Stage 2: SyncTalk worker (LIMITED CONCURRENCY = 1)
# ----------------------------
def run_synctalk_request(job: ChunkJob) -> list[np.ndarray]:
    """Blocking: fetch MJPEG + decode to BGR frames."""
    sess = requests.Session()
    frames = []
    try:
        files = {"audio": ("chunk.wav", job.wav, "audio/wav")}
        data = {
            "fps": str(job.fps),
            "start_frame": "0",
            "reset": "1",
            "jpeg_quality": str(JPEG_QUALITY),
        }
        resp = sess.post(SYNCTALK_URL, files=files, data=data, timeout=HTTP_TIMEOUT, stream=True)
        resp.raise_for_status()

        first_logged = False
        for jpg in mjpeg_iter_jpegs(resp):
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if not first_logged:
                logger.info(f"✅ First frame arrived for job {job.job_id} (shape={frame.shape})")
                first_logged = True
            frames.append(frame)

        resp.close()
        return frames
    finally:
        try:
            sess.close()
        except Exception:
            pass


async def synctalk_worker(job_queue: asyncio.Queue, frames: FrameStore):
    logger.info(f"SyncTalk worker started: {SYNCTALK_URL}")

    # One worker thread only => stable GPU latency
    pool = ThreadPoolExecutor(max_workers=1)

    idx_cache: Dict[int, int] = {}

    while True:
        job: ChunkJob = await job_queue.get()
        try:
            bgr_frames = await asyncio.get_running_loop().run_in_executor(pool, run_synctalk_request, job)

            for i, frame in enumerate(bgr_frames):
                await frames.put(job.job_id, i, frame)

        except Exception as e:
            logger.error(f"❌ SyncTalk failed for job {job.job_id}: {e}")
        finally:
            job_queue.task_done()


# ----------------------------
# Stage 3: AV publisher (CORRECT PREBUFFER)
# ----------------------------
async def av_publisher(job_queue: asyncio.Queue, frames: FrameStore,
                       video_source: rtc.VideoSource, audio_source: rtc.AudioSource):
    logger.info("AV publisher started (smooth scheduler).")

    audio_packet_samples = int(PUBLISH_SR * 0.02)  # 20ms
    last_rgba = None

    # PREBUFFER: store jobs locally
    pending: Deque[ChunkJob] = deque()

    while True:
        job: ChunkJob = await job_queue.get()
        try:
            pending.append(job)

            # Wait until we have enough buffered jobs
            if len(pending) < PREBUFFER_JOBS:
                continue

            # Play exactly one job per loop iteration (in order)
            job = pending.popleft()

            fps = max(1, job.fps)
            frame_interval = 1.0 / fps

            duration = len(job.pcm) / ASSISTANT_SR
            total_frames = max(1, int(round(duration * fps)))

            await sleep_until(job.start_at)

            audio_pos = 0
            next_audio_t = job.start_at
            next_video_t = job.start_at
            video_idx = 0
            end_t = job.start_at + duration

            while True:
                now = time.monotonic()
                if now >= end_t and audio_pos >= len(job.pcm) and video_idx >= total_frames:
                    break

                # audio priority
                if next_audio_t <= next_video_t:
                    await sleep_until(next_audio_t)
                    next_audio_t += 0.02

                    if audio_pos < len(job.pcm):
                        out = job.pcm[audio_pos: audio_pos + audio_packet_samples]
                        audio_pos += audio_packet_samples
                        if out.shape[0] < audio_packet_samples:
                            out = np.pad(out, (0, audio_packet_samples - out.shape[0]), mode="constant")

                        af = rtc.AudioFrame(
                            data=out.tobytes(),
                            sample_rate=PUBLISH_SR,
                            num_channels=1,
                            samples_per_channel=out.shape[0],
                        )
                        await capture_maybe_async(audio_source, af)

                else:
                    await sleep_until(next_video_t)
                    next_video_t += frame_interval

                    frame_bgr = await frames.get(job.job_id, video_idx, timeout_s=FRAME_WAIT)
                    if frame_bgr is not None:
                        last_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)

                    # hold last frame if missing
                    if last_rgba is not None:
                        vf = rtc.VideoFrame(
                            width=last_rgba.shape[1],
                            height=last_rgba.shape[0],
                            type=rtc.VideoBufferType.RGBA,
                            data=last_rgba.tobytes(),
                        )
                        await capture_maybe_async(video_source, vf)

                    video_idx += 1

        except Exception as e:
            logger.error(f"❌ AV publisher failed: {e}")
        finally:
            job_queue.task_done()


# ----------------------------
# SyncTalk size
# ----------------------------
def fetch_synctalk_size() -> Tuple[int, int]:
    try:
        r = requests.get(f"{SYNCTALK_BASE}/health", timeout=5)
        r.raise_for_status()
        j = r.json()
        sz = j.get("image_size")
        if isinstance(sz, str) and "x" in sz:
            w, h = sz.split("x")
            return int(w), int(h)
    except Exception:
        pass
    return 640, 480


# ----------------------------
# Main
# ----------------------------
async def entrypoint(ctx: JobContext):
    import_queue_audio_output()

    logger.info("Connecting...")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info("Connected.")

    w, h = await asyncio.to_thread(fetch_synctalk_size)
    logger.info(f"SyncTalk size: {w}x{h}")

    video_source = rtc.VideoSource(w, h)
    audio_source = rtc.AudioSource(sample_rate=PUBLISH_SR, num_channels=PUBLISH_CH)

    await ctx.room.local_participant.publish_track(create_local_video_track(VIDEO_TRACK_NAME, video_source))
    await ctx.room.local_participant.publish_track(create_local_audio_track(AUDIO_TRACK_NAME, audio_source))
    logger.info(f"Published tracks: {VIDEO_TRACK_NAME}, {AUDIO_TRACK_NAME}")

    model = google.realtime.RealtimeModel(
        model="gemini-2.5-flash-native-audio-preview-12-2025",
        voice="Puck",
        temperature=0.8,
        instructions="Reply in Bangla. Keep it short."
    )
    agent = Agent(llm=model, instructions="Bangla voice assistant.")
    session = AgentSession()

    queue_audio = QueueAudioOutput(sample_rate=ASSISTANT_SR)
    if hasattr(session, "output") and hasattr(session.output, "audio"):
        session.output.audio = queue_audio
    elif hasattr(session, "output_audio"):
        session.output_audio = queue_audio
    else:
        raise RuntimeError("Cannot attach QueueAudioOutput to AgentSession.")

    opts = room_io.RoomOptions(
        audio_input=True,
        audio_output=False,
        text_input=False,
        text_output=False,
        video_input=False,
    )

    await session.start(room=ctx.room, agent=agent, room_options=opts)
    logger.info("Gemini session started.")
    logger.info(f"Running: chunk={CHUNK_SECONDS}s prebuffer={PREBUFFER_JOBS} frame_wait={FRAME_WAIT}s")

    raw_jobs: asyncio.Queue = asyncio.Queue(maxsize=RAW_JOB_Q_MAX)
    jobs_for_synctalk: asyncio.Queue = asyncio.Queue(maxsize=SYNCTALK_JOB_Q_MAX)
    jobs_for_publish: asyncio.Queue = asyncio.Queue(maxsize=PUBLISH_JOB_Q_MAX)

    frames = FrameStore(max_items=FRAME_STORE_MAX)

    async def fanout():
        while True:
            job = await raw_jobs.get()
            try:
                await put_drop_oldest(jobs_for_synctalk, job)  # can drop
                await jobs_for_publish.put(job)               # never drop
            finally:
                raw_jobs.task_done()

    tasks = [
        asyncio.create_task(assistant_audio_to_jobs(queue_audio, raw_jobs)),
        asyncio.create_task(fanout()),
        asyncio.create_task(synctalk_worker(jobs_for_synctalk, frames)),
        asyncio.create_task(av_publisher(jobs_for_publish, frames, video_source, audio_source)),
    ]

    try:
        while True:
            await asyncio.sleep(1)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
