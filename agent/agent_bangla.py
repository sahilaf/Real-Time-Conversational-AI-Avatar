"""
agent_bangla.py — Bangla avatar agent with audio-master playback.

Pipeline:
    Gemini audio → QueueAudioOutput → WS → avatar server (SyncTalk_2D GPU)
    Server streams back, PER BATCH: the audio first, then frames as generated.
    Client plays AUDIO as the master clock; VIDEO slaves to it.

Why audio-master: the GPU cannot always sustain 25 gen-FPS. If audio waits for
video (the old design), every GPU hiccup becomes an audible gap mid-sentence.
Here audio is pushed on its own paced schedule and never waits; a late video
frame is skipped or the previous frame is held — a briefly frozen mouth instead
of broken speech.

Transitions: idle and speech share one base-frame walk. The client tells the
server where its idle loop is before each utterance ("align"), the server
reports where the walk ended after each utterance ("utterance_end"), and the
client resumes idle there — plus a short crossfade both ways.
"""

import os
import io
import json
import cv2
import time
import asyncio
import inspect
import logging
import numpy as np
import requests
import websockets
import aiohttp
import soundfile as sf
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import deque

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents import Agent, AgentSession
from livekit.agents.voice import room_io
from livekit.plugins import google

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

AVATAR_BASE = os.getenv("AVATAR_BASE", "http://127.0.0.1:5001")
ASSISTANT_SR = int(os.getenv("ASSISTANT_SR", "24000"))
GEMINI_MODEL = os.getenv("GEMINI_REALTIME_MODEL", "gemini-2.5-flash-native-audio-latest")

VIDEO_TRACK_NAME = os.getenv("VIDEO_TRACK_NAME", "agent_video")
AUDIO_TRACK_NAME = os.getenv("AUDIO_TRACK_NAME", "agent_audio")

VIDEO_FPS = 25
FRAME_S = 1.0 / VIDEO_FPS           # 40 ms per video frame

AUDIO_PUSH_S = 0.02                  # 20 ms audio frames to LiveKit
AUDIO_LEAD_S = 0.20                  # how far audio may run ahead of the clock
# The server emits frames in batches (one per audio fold), so the client sees
# bursts, not a steady 25/s. The gate must cover at least one full batch
# interval or playback starts right before the buffer runs dry and every
# batch boundary becomes a hold-then-skip.
GATE_FRAMES = 8                      # video frames buffered before starting
GATE_TIMEOUT_S = 0.45                # ...or start after this long regardless
CROSSFADE_FRAMES = 3                 # blended frames at idle<->speech edges

END_MARKER = 0xFFFFFFFF
FLUSH_SENTINEL = b"__FLUSH__"

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


def create_local_video_track(name: str, source: rtc.VideoSource) -> rtc.LocalVideoTrack:
    if hasattr(rtc.LocalVideoTrack, "create_video_track"):
        return rtc.LocalVideoTrack.create_video_track(name, source)
    return rtc.LocalVideoTrack.create(name, source)


def create_local_audio_track(name: str, source: rtc.AudioSource) -> rtc.LocalAudioTrack:
    if hasattr(rtc.LocalAudioTrack, "create_audio_track"):
        return rtc.LocalAudioTrack.create_audio_track(name, source)
    return rtc.LocalAudioTrack.create(name, source)


async def capture_maybe_async(source, frame):
    res = source.capture_frame(frame)
    if inspect.isawaitable(res):
        await res


async def sleep_until(t: float):
    now = time.monotonic()
    if t > now:
        await asyncio.sleep(t - now)


def create_session() -> str:
    r = requests.post(f"{AVATAR_BASE}/session", timeout=10)
    r.raise_for_status()
    return r.json()["session_id"]


def get_avatar_size() -> tuple[int, int]:
    try:
        r = requests.get(f"{AVATAR_BASE}/health", timeout=5)
        r.raise_for_status()
        w, h = r.json().get("image_size", "0x0").split("x")
        return int(w), int(h)
    except Exception:
        return 450, 450


def pcm16_bytes_to_wav_bytes(pcm_bytes: bytes, sr: int) -> bytes:
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    if pcm.size == 0:
        return b""
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# -----------------------------
# Idle animation cache
# -----------------------------
@dataclass
class IdleFrames:
    frames: List[bytes]                # jpg per cache position
    source_map: List[int]              # cache position -> source frame index

    def pos_for_source(self, source_idx: int) -> int:
        """Cache position showing the frame nearest to source_idx."""
        if not self.source_map:
            return 0
        return min(range(len(self.source_map)),
                   key=lambda i: abs(self.source_map[i] - source_idx))


async def fetch_idle_frames(base_url: str) -> Optional[IdleFrames]:
    try:
        async with aiohttp.ClientSession() as http:
            async with http.get(f"{base_url}/idle/info") as resp:
                if resp.status != 200:
                    logger.warning("Idle cache not available on server")
                    return None
                info = await resp.json()
            if not info.get("ready"):
                logger.warning("Server idle cache not ready")
                return None

            count = info["frame_count"]
            frames = []
            for i in range(count):
                async with http.get(f"{base_url}/idle/frame/{i}") as resp:
                    if resp.status == 200:
                        frames.append(await resp.read())

        if not frames:
            return None
        source_map = info.get("source_map") or list(range(len(frames)))
        logger.info(f"[IDLE] Cached {len(frames)} idle frames "
                    f"(source {source_map[0]}..{source_map[-1]})")
        return IdleFrames(frames=frames, source_map=source_map[:len(frames)])
    except Exception as e:
        logger.error(f"Error fetching idle frames: {e}")
        return None


# -----------------------------
# Stream state
# -----------------------------
@dataclass
class Utterance:
    start: int                          # first global frame number
    nframes: Optional[int] = None       # set by utterance_end
    t0: Optional[float] = None          # wall clock when playback started
    audio_done: bool = False            # all PCM pushed to LiveKit
    samples_pushed: int = 0             # PCM samples already sent to LiveKit


class AvatarStream:
    """Shared state between the receiver, audio pump, and video pump.

    Single event loop, no locks. Global frame numbers are continuous across
    utterances; each utterance records which range it owns.
    """

    def __init__(self, idle: Optional[IdleFrames]):
        self.idle = idle
        self.video_buf: Dict[int, bytes] = {}       # global frame no -> jpg
        self.audio_q: asyncio.Queue = asyncio.Queue()   # np.int16 | None sentinel
        self.seg_base: Dict[int, int] = {}          # segment id -> global base
        self.next_global = 0
        self.utterances: deque[Utterance] = deque()
        self.resume_source_idx: Optional[int] = None
        self.idle_pos = 0                           # position in the idle cache
        self.need_align = True                      # send align before next audio

    # -- receiver side --------------------------------------------------
    def on_audio_packet(self, seg_id: int, nframes: int, pcm: np.ndarray):
        if not self.utterances or self.utterances[-1].nframes is not None:
            self.utterances.append(Utterance(start=self.next_global))
        self.seg_base[seg_id] = self.next_global
        self.next_global += nframes
        self.audio_q.put_nowait(pcm)

    def on_frame_packet(self, seg_id: int, frame_idx: int, jpg: bytes):
        base = self.seg_base.get(seg_id)
        if base is not None:
            self.video_buf[base + frame_idx] = jpg

    def on_utterance_end(self, frames: int, end_source_idx: Optional[int]):
        for utt in self.utterances:
            if utt.nframes is None:
                utt.nframes = frames
                break
        self.resume_source_idx = end_source_idx
        self.audio_q.put_nowait(None)               # delimiter for the pump

    # -- shared ---------------------------------------------------------
    def active_utterance(self) -> Optional[Utterance]:
        while self.utterances:
            utt = self.utterances[0]
            if utt.t0 is None:
                return None                          # not started yet
            if utt.nframes is not None and utt.audio_done and \
               (time.monotonic() - utt.t0) * VIDEO_FPS >= utt.nframes:
                # fully played out; drop it and any stale frames
                for g in [g for g in self.video_buf if g < utt.start + utt.nframes]:
                    self.video_buf.pop(g, None)
                self.utterances.popleft()
                continue
            return utt
        return None

    def speaking(self) -> bool:
        return self.active_utterance() is not None

    def idle_source_idx(self) -> int:
        if self.idle and self.idle.source_map:
            return self.idle.source_map[self.idle_pos % len(self.idle.source_map)]
        return 0

    def make_align_message(self) -> str:
        return json.dumps({"type": "align", "source_idx": self.idle_source_idx()})


# -----------------------------
# Gemini → server
# -----------------------------
async def pcm_fanout(queue_audio, pcm_q_ws: asyncio.Queue, st: AvatarStream):
    """Single consumer of QueueAudioOutput; forwards PCM to the server queue."""
    async for item in queue_audio:
        if AudioSegmentEnd is not None and isinstance(item, AudioSegmentEnd):
            await pcm_q_ws.put(FLUSH_SENTINEL)
            st.need_align = True                    # next utterance re-aligns
            continue
        pcm = np.frombuffer(bytes(item.data), dtype=np.int16)
        if pcm.size:
            await pcm_q_ws.put(pcm.tobytes())


async def ws_send_pcm(pcm_q_ws: asyncio.Queue, ws_audio, st: AvatarStream,
                      sr: int = ASSISTANT_SR, batch_ms: int = 40):
    """Batch PCM into WAV chunks and send to the avatar server. Never drops.

    Before the first audio of each utterance, sends an "align" control message
    so the server starts its base-frame walk where the idle loop currently is.
    """
    buf = np.zeros(0, dtype=np.int16)
    last_send = time.monotonic()
    batch_samples = int(sr * batch_ms / 1000.0)

    while True:
        force_flush = False
        try:
            b = await asyncio.wait_for(pcm_q_ws.get(), timeout=0.05)
            if b == FLUSH_SENTINEL:
                force_flush = True
            else:
                buf = np.concatenate([buf, np.frombuffer(b, dtype=np.int16)])
        except asyncio.TimeoutError:
            pass

        age = time.monotonic() - last_send
        if not force_flush and (buf.size < batch_samples and age < 0.25):
            continue
        if buf.size == 0 and not force_flush:
            continue

        try:
            if buf.size > 0:
                if st.need_align:
                    await ws_audio.send(st.make_align_message())
                    st.need_align = False
                wav = pcm16_bytes_to_wav_bytes(buf.tobytes(), sr)
                if wav:
                    await ws_audio.send(wav)
                buf = np.zeros(0, dtype=np.int16)
                last_send = time.monotonic()
            if force_flush:
                await ws_audio.send(FLUSH_SENTINEL)
        except Exception as e:
            logger.error(f"Error sending audio to WS: {e}")
            buf = np.zeros(0, dtype=np.int16)


# -----------------------------
# Server → client
# -----------------------------
async def receive_stream(ws_video, st: AvatarStream):
    """Parse the server stream into the shared state.

    Binary: 16-byte header [seg_id][frame_idx][total][dur_ms] + payload.
        frame_idx == END_MARKER → payload is the batch PCM (arrives FIRST).
        otherwise               → payload is a jpg frame.
    Text: {"type": "utterance_end", "frames": N, "end_source_idx": i}
    """
    while True:
        msg = await ws_video.recv()
        if isinstance(msg, str):
            try:
                js = json.loads(msg)
            except ValueError:
                continue
            if js.get("type") == "utterance_end":
                st.on_utterance_end(int(js.get("frames", 0)),
                                    js.get("end_source_idx"))
            continue

        if len(msg) < 16:
            continue
        seg_id = int.from_bytes(msg[0:4], "little")
        frame_idx = int.from_bytes(msg[4:8], "little")
        nframes = int.from_bytes(msg[8:12], "little")
        payload = msg[16:]

        if frame_idx == END_MARKER:
            st.on_audio_packet(seg_id, nframes,
                               np.frombuffer(payload, dtype=np.int16))
        else:
            st.on_frame_packet(seg_id, frame_idx, payload)


# -----------------------------
# Audio pump — the master clock
# -----------------------------
async def audio_pump(st: AvatarStream, audio_source: rtc.AudioSource):
    """Push PCM to LiveKit on an absolute schedule. Never waits for video.

    Per utterance: gate briefly so the first video frames exist, anchor the
    clock (utt.t0), then push 20 ms frames capped AUDIO_LEAD_S ahead of the
    clock. A None sentinel delimits utterances.
    """
    push_samples = int(ASSISTANT_SR * AUDIO_PUSH_S)

    while True:
        pcm = await st.audio_q.get()

        if pcm is None:
            # utterance delimiter: audio arrives in order, so this closes the
            # OLDEST open utterance only. Closing all of them (the previous
            # version) let one delimiter swallow a later utterance's audio.
            for utt in st.utterances:
                if not utt.audio_done:
                    utt.audio_done = True
                    break
            continue

        # audio belongs to the oldest utterance still expecting it
        utt = next((u for u in st.utterances if not u.audio_done), None)
        if utt is None:
            continue                                  # stale audio after reset

        if utt.t0 is None:
            # Strictly serial playback: never anchor this utterance's clock
            # while an earlier one is still playing out, or its video timeline
            # starts in the past and the pump skips a burst of frames.
            while st.utterances and st.utterances[0] is not utt:
                st.active_utterance()      # lets finished utterances retire
                await asyncio.sleep(0.05)

            gate_t0 = time.monotonic()
            while (sum(1 for g in st.video_buf if g >= utt.start) < GATE_FRAMES
                   and time.monotonic() - gate_t0 < GATE_TIMEOUT_S):
                await asyncio.sleep(0.02)
            utt.t0 = time.monotonic()
            logger.info(f"▶️ Utterance started (global frame {utt.start})")

        pending = pcm
        while pending.size > 0:
            chunk, pending = pending[:push_samples], pending[push_samples:]
            # cap how far audio runs ahead of the playback clock
            target = utt.t0 + utt.samples_pushed / ASSISTANT_SR - AUDIO_LEAD_S
            await sleep_until(target)
            af = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=ASSISTANT_SR,
                num_channels=1,
                samples_per_channel=len(chunk),
            )
            await capture_maybe_async(audio_source, af)
            utt.samples_pushed += len(chunk)


# -----------------------------
# Video pump — slaved to the audio clock
# -----------------------------
async def video_pump(st: AvatarStream, video_source: rtc.VideoSource):
    """Publish one frame every 40 ms on an absolute schedule.

    Speaking: show the newest buffered frame at or before the audio clock,
    dropping older ones; hold the last frame when the GPU is behind.
    Idle: loop the cached idle animation.
    Transitions crossfade over CROSSFADE_FRAMES frames both ways.
    """
    last_bgr: Optional[np.ndarray] = None
    last_played_global = -1
    crossfade_left = 0

    def decode(jpg: bytes) -> Optional[np.ndarray]:
        arr = np.frombuffer(jpg, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    async def publish(bgr: np.ndarray):
        nonlocal last_bgr, crossfade_left
        if crossfade_left > 0 and last_bgr is not None and \
                last_bgr.shape == bgr.shape:
            alpha = 1.0 - crossfade_left / (CROSSFADE_FRAMES + 1)
            bgr = cv2.addWeighted(bgr, alpha, last_bgr, 1.0 - alpha, 0)
            crossfade_left -= 1
        else:
            crossfade_left = 0
        last_bgr = bgr
        h, w = bgr.shape[:2]
        i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        vf = rtc.VideoFrame(width=w, height=h,
                            type=rtc.VideoBufferType.I420,
                            data=i420.tobytes())
        await capture_maybe_async(video_source, vf)

    was_speaking = False
    next_t = time.monotonic()

    while True:
        next_t += FRAME_S
        await sleep_until(next_t)

        utt = st.active_utterance()

        if utt is not None:
            if not was_speaking:
                was_speaking = True
                crossfade_left = CROSSFADE_FRAMES       # idle -> speech blend
            # frame the audio clock says we should be showing, clamped so a
            # finishing utterance never pulls in the next utterance's frames
            target = utt.start + int((time.monotonic() - utt.t0) * VIDEO_FPS)
            if utt.nframes is not None:
                target = min(target, utt.start + utt.nframes - 1)
            best = None
            for g in st.video_buf:
                if utt.start <= g <= target and (best is None or g > best):
                    best = g
            if best is not None and best > last_played_global:
                jpg = st.video_buf.pop(best)
                # evict anything older; it will never be shown
                for g in [g for g in st.video_buf if g < best]:
                    st.video_buf.pop(g, None)
                bgr = decode(jpg)
                if bgr is not None:
                    last_played_global = best
                    await publish(bgr)
                    continue
            # nothing new in time: hold the last frame
            if last_bgr is not None:
                await publish(last_bgr)
            continue

        # ---- idle ----------------------------------------------------
        if was_speaking:
            was_speaking = False
            crossfade_left = CROSSFADE_FRAMES           # speech -> idle blend
            if st.idle and st.resume_source_idx is not None:
                st.idle_pos = st.idle.pos_for_source(st.resume_source_idx)
                st.resume_source_idx = None

        if st.idle and st.idle.frames:
            bgr = decode(st.idle.frames[st.idle_pos % len(st.idle.frames)])
            st.idle_pos = (st.idle_pos + 1) % len(st.idle.frames)
            if bgr is not None:
                await publish(bgr)
        elif last_bgr is not None:
            await publish(last_bgr)


# -----------------------------
# Entrypoint
# -----------------------------
async def entrypoint(ctx: JobContext):
    import_queue_audio_output()

    logger.info("Connecting LiveKit...")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    w, h = get_avatar_size()
    logger.info(f"Avatar server image_size: {w}x{h}")

    sid = await asyncio.to_thread(create_session)
    logger.info(f"Created avatar session: {sid}")

    idle = await fetch_idle_frames(AVATAR_BASE)
    st = AvatarStream(idle)

    ws_base = AVATAR_BASE.replace("http://", "ws://").replace("https://", "wss://")
    ws_audio_url = f"{ws_base}/ws/audio/{sid}"
    ws_video_url = f"{ws_base}/ws/video/{sid}"

    video_source = rtc.VideoSource(w, h)
    audio_source = rtc.AudioSource(sample_rate=ASSISTANT_SR, num_channels=1)
    video_track = create_local_video_track(VIDEO_TRACK_NAME, video_source)
    audio_track = create_local_audio_track(AUDIO_TRACK_NAME, audio_source)

    await ctx.room.local_participant.publish_track(
        video_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA))
    await ctx.room.local_participant.publish_track(
        audio_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    logger.info(f"Published tracks: {VIDEO_TRACK_NAME}, {AUDIO_TRACK_NAME}")

    model = google.realtime.RealtimeModel(
        model=GEMINI_MODEL,
        voice="Puck",
        temperature=0.7,
        instructions="""You are Redwan, a helpful AI avatar assistant who ONLY speaks Bangla.

YOUR IDENTITY:
- Name: Redwan.
- Role: AI Avatar Assistant.
- Language: You must strictly speak ONLY in Bangla. Do not use English unless specifically asked to translate or explain an English term, but even then, explain in Bangla.

INTERACTION STYLE:
- Be polite, friendly, and helpful.
- Keep responses concise and natural.
- Assist the user with their queries in Bangla.
- If asked about your identity, say you are Redwan, an AI assistant.
""",
    )
    agent = Agent(llm=model,
                  instructions="You are Redwan, a helpful AI avatar assistant. "
                               "Speak ONLY in Bangla. Keep responses concise.")
    session = AgentSession()

    queue_audio = QueueAudioOutput(sample_rate=ASSISTANT_SR)
    if hasattr(session, "output") and hasattr(session.output, "audio"):
        session.output.audio = queue_audio
    elif hasattr(session, "output_audio"):
        session.output_audio = queue_audio
    else:
        raise RuntimeError("Cannot attach QueueAudioOutput for this livekit-agents version.")

    # With text_output enabled, livekit-agents wraps the audio output in a
    # transcript synchronizer. That wrapper's flush() crashes (ChanClosed in
    # transcription/_speaking_rate) after the first response when the output is
    # a custom queue, and after the internal reconnect our queue is no longer
    # the sink - the agent then answers exactly one question. Disable the sync
    # if this livekit-agents version supports it; transcripts still flow, they
    # are just not word-timed to the audio.
    try:
        text_out = room_io.TextOutputOptions(sync_transcription=False)
    except TypeError:
        text_out = True
        logger.warning("This livekit-agents has no sync_transcription option; "
                       "relying on the post-start re-attach instead.")
    try:
        opts = room_io.RoomOptions(
            audio_input=True,
            audio_output=False,
            text_input=True,
            text_output=text_out,
            video_input=False,
            close_on_disconnect=False,   # keep alive if Playground disconnects
        )
    except TypeError:
        opts = room_io.RoomOptions(
            audio_input=True, audio_output=False,
            text_input=True, text_output=True, video_input=False)

    await session.start(room=ctx.room, agent=agent, room_options=opts)

    # Re-attach AFTER start: if session.start wrapped or replaced the audio
    # output (transcript synchronizer, room IO), this restores our queue as
    # the direct sink for every subsequent response.
    if hasattr(session, "output") and hasattr(session.output, "audio"):
        if session.output.audio is not queue_audio:
            logger.info("Audio output was wrapped by session.start; re-attaching queue.")
            session.output.audio = queue_audio

    logger.info(f"Gemini session started (model={GEMINI_MODEL}). Connecting WS...")

    pcm_q_ws = asyncio.Queue(maxsize=2000)

    async with websockets.connect(ws_audio_url, max_size=None) as wsa, \
               websockets.connect(ws_video_url, max_size=None) as wsv:
        logger.info("Connected to GPU avatar server. Audio-master playback active.")

        tasks = [
            asyncio.create_task(pcm_fanout(queue_audio, pcm_q_ws, st), name="pcm_fanout"),
            asyncio.create_task(ws_send_pcm(pcm_q_ws, wsa, st), name="ws_send_pcm"),
            asyncio.create_task(receive_stream(wsv, st), name="receive_stream"),
            asyncio.create_task(audio_pump(st, audio_source), name="audio_pump"),
            asyncio.create_task(video_pump(st, video_source), name="video_pump"),
        ]
        try:
            while True:
                await asyncio.sleep(1)
                for task in tasks:
                    if task.done():
                        try:
                            task.result()
                        except Exception:
                            logger.exception(f"Task {task.get_name()} died; "
                                             "shutting down pipeline")
                            raise
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    required = ["GOOGLE_API_KEY", "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
