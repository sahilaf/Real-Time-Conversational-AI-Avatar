"""
agent.py — SYNCHRONIZED Avatar Playback

Architecture:
  1. Gemini audio → server via WS
  2. Server sends segments: [video frames] + [end-of-segment with PCM audio]
  3. Client collects COMPLETE segments before playback
  4. Synchronized playback: audio + video at exactly 25 FPS

This ensures perfect lip sync by:
  ✅ Waiting for all frames + audio for each segment
  ✅ Playing audio and video together at matched timing
  ✅ Constant 25 FPS output (40ms per frame)
"""

import os
import io
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
from dataclasses import dataclass, field
from typing import List, Optional
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

VIDEO_TRACK_NAME = os.getenv("VIDEO_TRACK_NAME", "agent_video")
AUDIO_TRACK_NAME = os.getenv("AUDIO_TRACK_NAME", "agent_audio")

VIDEO_FPS = 25
FRAME_DURATION_S = 1.0 / VIDEO_FPS  # 40ms per frame
AUDIO_PACKET_S = 0.04  # 40ms audio packets (larger = smoother)

END_MARKER = 0xFFFFFFFF

# Idle animation cache
@dataclass
class IdleFramesCache:
    """Cache of idle animation frames fetched from server"""
    frames: List[bytes]  # List of jpg bytes
    frame_count: int
    img_w: int
    img_h: int
    
    def get_frame(self, index: int) -> bytes:
        """Get frame at index, wrapping around for looping"""
        return self.frames[index % len(self.frames)]

idle_frames_cache: Optional[IdleFramesCache] = None


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
        js = r.json()
        sz = js.get("image_size", "0x0")
        w, h = sz.split("x")
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
# Segment Data Structure
# -----------------------------
@dataclass
class Segment:
    """A complete audio-video segment for synchronized playback"""
    segment_id: int
    total_frames: int
    audio_duration_ms: int
    video_frames: List[bytes] = field(default_factory=list)  # List of jpg bytes
    audio_pcm: Optional[bytes] = None  # PCM16 audio bytes
    
    def is_complete(self) -> bool:
        """Check if segment has all video frames and audio"""
        return (
            len(self.video_frames) == self.total_frames and 
            self.audio_pcm is not None
        )


# -----------------------------
# Audio sender (Gemini → Server)
# -----------------------------
FLUSH_SENTINEL = b"__FLUSH__"

async def pcm_fanout(queue_audio, pcm_q_ws: asyncio.Queue):
    """
    SINGLE consumer of QueueAudioOutput => sends to server WS queue.
    Audio is synced with video - both come back together in segments.
    """
    async for item in queue_audio:
        if AudioSegmentEnd is not None and isinstance(item, AudioSegmentEnd):
            await pcm_q_ws.put(FLUSH_SENTINEL)
            continue
            
        pcm = np.frombuffer(bytes(item.data), dtype=np.int16)
        if pcm.size == 0:
            continue
        b = pcm.tobytes()

        # Send to server for synced video generation
        await pcm_q_ws.put(b)


async def ws_send_pcm(pcm_q_ws: asyncio.Queue, ws_audio, sr: int = ASSISTANT_SR, batch_size_ms: int = 40):
    """
    Send PCM audio to avatar server via WebSocket.
    Batches small chunks together and sends as single WAV file.
    NEVER drops audio. Handles flushing.
    """
    pcm_buffer = np.zeros((0,), dtype=np.int16)
    last_send_time = time.time()
    batch_samples = int(sr * batch_size_ms / 1000.0)
    
    while True:
        force_flush = False
        try:
            # Get audio with timeout
            b = await asyncio.wait_for(pcm_q_ws.get(), timeout=0.05)
            
            if b == FLUSH_SENTINEL:
                force_flush = True
            else:
                pcm = np.frombuffer(b, dtype=np.int16)
                pcm_buffer = np.concatenate([pcm_buffer, pcm])
            
            pcm_q_ws.task_done()
            
        except asyncio.TimeoutError:
            # Send buffered audio if enough time has passed
            if pcm_buffer.size > 0 and (time.time() - last_send_time >= batch_size_ms / 1000.0):
                pass  # Will send below
            else:
                continue
        
        # Check if we should send the buffer
        should_send = False
        if pcm_buffer.size > 0:
            buffer_age = time.time() - last_send_time
            # Send if: buffer is large enough OR old enough OR queue is filling up
            if pcm_buffer.size >= batch_samples or buffer_age >= 0.25 or pcm_q_ws.qsize() > 50:
                should_send = True
        
        if force_flush:
            should_send = True
            
        if not should_send:
            continue
        
        try:
            # NO DROPPING HERE!
            
            # Convert PCM buffer to single WAV file
            if pcm_buffer.size > 0:
                wav_bytes = pcm16_bytes_to_wav_bytes(pcm_buffer.tobytes(), sr)
                if wav_bytes:
                    await ws_audio.send(wav_bytes)
                    logger.debug(f"Sent {len(wav_bytes)} bytes ({pcm_buffer.size / sr:.3f}s) to avatar server")
                
                pcm_buffer = np.zeros((0,), dtype=np.int16)
                last_send_time = time.time()
            
            if force_flush:
                # Send special flush marker to server
                await ws_audio.send(FLUSH_SENTINEL)
                logger.debug("Sent FLUSH signal to server")
            
        except Exception as e:
            logger.error(f"Error sending audio to WS: {e}")
            pcm_buffer = np.zeros((0,), dtype=np.int16)


# -----------------------------
# Idle Frames Fetcher
# -----------------------------
async def fetch_idle_frames(base_url: str) -> Optional[IdleFramesCache]:
    """Fetch idle animation frames from server"""
    global idle_frames_cache
    
    try:
        # Get idle info
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/idle/info") as resp:
                if resp.status != 200:
                    logger.warning("Idle cache not available on server")
                    return None
                info = await resp.json()
        
        if not info.get("ready"):
            logger.warning("Server idle cache not ready")
            return None
        
        frame_count = info["frame_count"]
        img_w = info["img_w"]
        img_h = info["img_h"]
        
        logger.info(f"[IDLE] Fetching {frame_count} idle frames from server...")
        
        # Fetch all frames
        frames = []
        async with aiohttp.ClientSession() as session:
            for i in range(frame_count):
                async with session.get(f"{base_url}/idle/frame/{i}") as resp:
                    if resp.status == 200:
                        frames.append(await resp.read())
                
                if (i + 1) % 25 == 0:
                    logger.info(f"[IDLE] Fetched {i + 1}/{frame_count} frames...")
        
        if len(frames) == 0:
            logger.warning("No idle frames fetched")
            return None
        
        idle_frames_cache = IdleFramesCache(
            frames=frames,
            frame_count=len(frames),
            img_w=img_w,
            img_h=img_h
        )
        
        logger.info(f"[IDLE] ✅ Cached {len(frames)} idle frames")
        return idle_frames_cache
        
    except Exception as e:
        logger.error(f"Error fetching idle frames: {e}")
        return None


# -----------------------------
# Segment Receiver (Server → Client)
# -----------------------------
async def receive_segments(ws_video, segment_q: asyncio.Queue):
    """
    Receive segment packets from server and assemble complete segments.
    
    Packet format (16-byte header):
        [4B segment_id][4B frame_index][4B total_frames][4B audio_duration_ms] + data
        
    When frame_index == 0xFFFFFFFF, it's an end-of-segment marker with PCM audio.
    """
    pending_segments = {}  # segment_id -> Segment
    segments_completed = 0
    
    while True:
        try:
            msg = await ws_video.recv()
            if isinstance(msg, str):
                continue
            if not msg or len(msg) < 16:
                continue

            # Parse 16-byte header
            segment_id = int.from_bytes(msg[0:4], "little", signed=False)
            frame_index = int.from_bytes(msg[4:8], "little", signed=False)
            total_frames = int.from_bytes(msg[8:12], "little", signed=False)
            audio_dur_ms = int.from_bytes(msg[12:16], "little", signed=False)
            data = msg[16:]
            
            # Get or create pending segment
            if segment_id not in pending_segments:
                pending_segments[segment_id] = Segment(
                    segment_id=segment_id,
                    total_frames=total_frames,
                    audio_duration_ms=audio_dur_ms,
                    video_frames=[None] * total_frames  # Pre-allocate
                )
            
            seg = pending_segments[segment_id]
            
            if frame_index == END_MARKER:
                # End-of-segment marker with audio PCM
                seg.audio_pcm = data
                logger.debug(f"[Segment {segment_id}] Received audio PCM ({len(data)} bytes)")
            else:
                # Video frame
                if frame_index < len(seg.video_frames):
                    seg.video_frames[frame_index] = data
                logger.debug(f"[Segment {segment_id}] Received frame {frame_index + 1}/{total_frames}")
            
            # Check if segment is complete
            if seg.is_complete():
                # Verify all frames are present (not None)
                if all(f is not None for f in seg.video_frames):
                    segments_completed += 1
                    logger.info(f"[SYNC] Segment {segment_id} complete: {len(seg.video_frames)} frames for {seg.audio_duration_ms}ms audio")
                    
                    # Send to playback queue
                    await segment_q.put(seg)
                    
                    # Clean up
                    del pending_segments[segment_id]
                else:
                    logger.warning(f"[Segment {segment_id}] Has null frames, waiting for retransmit")
                    
        except Exception as e:
            logger.error(f"Error receiving segment: {e}")
            await asyncio.sleep(0.01)


# -----------------------------
# Synchronized Playback
# -----------------------------
async def synchronized_playback(
    segment_q: asyncio.Queue,
    video_source: rtc.VideoSource,
    audio_source: rtc.AudioSource,
    img_w: int,
    img_h: int
):
    """
    Play audio and video in perfect sync at exactly 25 FPS.
    When buffer is empty, play idle animation for lifelike appearance.
    
    For each complete segment:
    1. Start audio and video playback together
    2. Pace video frames at 40ms intervals (25 FPS)
    3. Pace audio packets to match video timing
    """
    global idle_frames_cache
    
    segments_played = 0
    total_frames_played = 0
    idle_frame_index = 0
    last_frame_time = time.monotonic()
    
    audio_packet_samples = int(ASSISTANT_SR * AUDIO_PACKET_S)  # 960 samples per 40ms
    
    async def publish_video_frame(jpg_bytes: bytes) -> bool:
        """Publish a single video frame"""
        try:
            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                return False
            
            h, w = frame_bgr.shape[:2]
            frame_i420 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420)
            
            vf = rtc.VideoFrame(
                width=w,
                height=h,
                type=rtc.VideoBufferType.I420,
                data=frame_i420.tobytes(),
            )
            await capture_maybe_async(video_source, vf)
            return True
        except Exception as e:
            logger.error(f"Error publishing video frame: {e}")
            return False
    
    async def play_idle_frame() -> bool:
        """Play a single idle frame at 25 FPS"""
        nonlocal idle_frame_index, last_frame_time
        
        if idle_frames_cache is None or len(idle_frames_cache.frames) == 0:
            await asyncio.sleep(FRAME_DURATION_S)
            return False
        
        # Wait for correct frame timing (25 FPS)
        current_time = time.monotonic()
        time_since_last = current_time - last_frame_time
        if time_since_last < FRAME_DURATION_S:
            await asyncio.sleep(FRAME_DURATION_S - time_since_last)
        
        jpg_bytes = idle_frames_cache.get_frame(idle_frame_index)
        await publish_video_frame(jpg_bytes)
        
        idle_frame_index = (idle_frame_index + 1) % idle_frames_cache.frame_count
        last_frame_time = time.monotonic()
        return True
    
    logger.info("[PLAYBACK] Starting Jitter Buffer Mode (Start @ 6 frames)")
    
    playback_q = asyncio.Queue(maxsize=1000)
    
    # Task: Buffer filler (Deque segments -> frames/audio)
    async def buffer_worker():
        while True:
            try:
                # Wait for segment
                seg = await segment_q.get()
                
                nframes = len(seg.video_frames)
                if nframes == 0: continue
                
                pcm_full = np.frombuffer(seg.audio_pcm, dtype=np.int16)
                total_samples = len(pcm_full)
                
                # Split audio per frame
                samples_per_frame = total_samples // nframes
                remainder = total_samples % nframes
                
                offset = 0
                for i in range(nframes):
                    current_samples = samples_per_frame + (1 if i < remainder else 0)
                    chunk = pcm_full[offset : offset + current_samples]
                    offset += current_samples
                    
                    frame_dur = current_samples / float(ASSISTANT_SR)
                    
                    await playback_q.put((seg.video_frames[i], chunk, frame_dur))
                    
            except Exception as e:
                logger.error(f"Buffer worker error: {e}")
                await asyncio.sleep(0.01)

    asyncio.create_task(buffer_worker())
    
    last_frame_bytes = None
    last_activity_time = time.monotonic()
    
    # FIX 5: Adaptive buffering for optimal latency/reliability balance
    MIN_BUFFER = 12  # Initial buffer: 480ms (good balance)
    RESUME_BUFFER = 25  # After underrun: 1s (more conservative)
    currently_underrun = False  # Track if we just had an underrun
    is_playing = False
    
    while True:
        # State 1: Buffering / Idle
        if not is_playing:
            current_qsize = playback_q.qsize()
            
            # Use adaptive threshold based on underrun history
            required_buffer = RESUME_BUFFER if currently_underrun else MIN_BUFFER
            
            if current_qsize >= required_buffer:
                is_playing = True
                currently_underrun = False  # Reset underrun flag
                last_activity_time = time.monotonic()
                logger.info(f"▶️ Playback Started (Buffered {current_qsize} frames)")
            else:
                # Idle Timeout Check
                if time.monotonic() - last_activity_time > 1.5:
                    await play_idle_frame()
                    continue
                else:
                    # Hold last frame (freeze)
                    if last_frame_bytes:
                        await publish_video_frame(last_frame_bytes)
                    else:
                        await play_idle_frame()
                    await asyncio.sleep(0.02)
                    continue

        # State 2: Playing
        if playback_q.empty():
            # Underrun - switch to buffering mode
            is_playing = False
            currently_underrun = True  # Mark that we had an underrun
            logger.warning(f"⏸️ Buffer underrun - switching to buffering (will need {RESUME_BUFFER} frames)")
            continue
            
        try:
            # Play
            frame, audio_chunk, dur = await playback_q.get()
            last_activity_time = time.monotonic()
            last_frame_bytes = frame
            
            # Publish Video
            await publish_video_frame(frame)
            
            # Publish Audio
            if len(audio_chunk) > 0:
                 af = rtc.AudioFrame(
                    data=audio_chunk.tobytes(),
                    sample_rate=ASSISTANT_SR,
                    num_channels=1,
                    samples_per_channel=len(audio_chunk)
                 )
                 await capture_maybe_async(audio_source, af)
            
            # Wait
            await asyncio.sleep(dur)
            
        except Exception as e:
            logger.error(f"Playback error: {e}")
            await asyncio.sleep(0.04)
            



async def entrypoint(ctx: JobContext):
    import_queue_audio_output()

    logger.info("Connecting LiveKit...")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info("Connected LiveKit.")

    w, h = get_avatar_size()
    logger.info(f"Avatar server image_size: {w}x{h}")

    sid = await asyncio.to_thread(create_session)
    logger.info(f"Created avatar session: {sid}")
    
    # Fetch idle animation frames from server
    await fetch_idle_frames(AVATAR_BASE)

    ws_audio_url = AVATAR_BASE.replace("http://", "ws://").replace("https://", "wss://") + f"/ws/audio/{sid}"
    ws_video_url = AVATAR_BASE.replace("http://", "ws://").replace("https://", "wss://") + f"/ws/video/{sid}"

    # Create sources with real dimensions
    video_source = rtc.VideoSource(w, h)
    audio_source = rtc.AudioSource(sample_rate=ASSISTANT_SR, num_channels=1)

    # Create tracks
    video_track = create_local_video_track(VIDEO_TRACK_NAME, video_source)
    audio_track = create_local_audio_track(AUDIO_TRACK_NAME, audio_source)

    # Publish with explicit options
    video_options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_CAMERA,
    )
    audio_options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_MICROPHONE,
    )

    await ctx.room.local_participant.publish_track(video_track, video_options)
    await ctx.room.local_participant.publish_track(audio_track, audio_options)
    logger.info(f"Published tracks: {VIDEO_TRACK_NAME}, {AUDIO_TRACK_NAME}")

    model = google.realtime.RealtimeModel(
        model="gemini-2.5-flash-native-audio-preview-12-2025",
        voice="Puck",
        temperature=0.7,
        instructions="""You are Obama, a helpful AI avatar assistant who ONLY speaks English.

YOUR IDENTITY:
- Name: Obama.
- Role: AI Avatar Assistant.
- Language: You must strictly speak ONLY in English. Do not use any other language. Always respond in English regardless of the language the user speaks in.

INTERACTION STYLE:
- Be polite, friendly, and helpful.
- Keep responses concise and natural.
- Assist the user with their queries in English.
- If asked about your identity, say you are Obama, an AI assistant.
""",
    )
    agent = Agent(llm=model, instructions="You are Obama, a helpful AI avatar assistant. Speak ONLY in English. Keep responses concise.")
    session = AgentSession()

    queue_audio = QueueAudioOutput(sample_rate=ASSISTANT_SR)

    # Attach QueueAudioOutput robustly
    if hasattr(session, "output") and hasattr(session.output, "audio"):
        session.output.audio = queue_audio
    elif hasattr(session, "output_audio"):
        session.output_audio = queue_audio
    else:
        raise RuntimeError("Cannot attach QueueAudioOutput for this livekit-agents version.")

    opts = room_io.RoomOptions(
        audio_input=True,
        audio_output=False,
        text_input=True,  # Enable user speech transcription display
        text_output=True,  # Enable AI response text display
        video_input=False,
    )

    # Keep agent alive if Playground participant disconnects
    try:
        if hasattr(opts, "input") and opts.input and hasattr(opts.input, "close_on_disconnect"):
            opts.input.close_on_disconnect = False
        else:
            opts.input = room_io.RoomInputOptions(close_on_disconnect=False)
    except Exception:
        pass

    await session.start(room=ctx.room, agent=agent, room_options=opts)
    logger.info("Gemini session started. Connecting WS...")

    # Queues for synchronized playback - large enough for complete responses
    # Queues for synchronized playback - large enough for complete responses (NEVER DROP)
    pcm_q_ws = asyncio.Queue(maxsize=2000)  # Gemini audio → server
    segment_q = asyncio.Queue(maxsize=300)  # User requested 300

    async with websockets.connect(ws_audio_url, max_size=None) as wsa, \
               websockets.connect(ws_video_url, max_size=None) as wsv:
        logger.info("Connected to GPU avatar server via WebSockets.")
        logger.info("🎯 SYNCHRONIZED PLAYBACK MODE: Segments will play with perfect lip sync")

        tasks = [
            asyncio.create_task(pcm_fanout(queue_audio, pcm_q_ws), name="pcm_fanout"),
            asyncio.create_task(ws_send_pcm(pcm_q_ws, wsa), name="ws_send_pcm"),
            asyncio.create_task(receive_segments(wsv, segment_q), name="receive_segments"),
            asyncio.create_task(
                synchronized_playback(segment_q, video_source, audio_source, w, h),
                name="synchronized_playback"
            ),
        ]

        try:
            while True:
                await asyncio.sleep(1)
                
                # Check if any task has failed
                for task in tasks:
                    if task.done():
                        try:
                            task.result()
                        except Exception as e:
                            logger.error(f"Task {task.get_name()} failed: {e}")
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