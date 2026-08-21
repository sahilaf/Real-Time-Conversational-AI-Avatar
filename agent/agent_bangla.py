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
import contextlib
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
from google.genai import types as genai_types

try:
    from livekit.plugins import silero
except ImportError:
    silero = None

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

AVATAR_BASE = os.getenv("AVATAR_BASE", "http://127.0.0.1:5001")
ASSISTANT_SR = int(os.getenv("ASSISTANT_SR", "24000"))
# The live-cascade line, NOT gemini-2.5-flash-native-audio-*: the native-audio
# family's server-side processing slows as session audio context accumulates
# (measured: reply latency growing 8 -> 66s over five turns, in a bare agent,
# with a flat send queue proving the audio was already at Google). On this
# model the same conversation measures a constant 0.5s generation per turn.
GEMINI_MODEL = os.getenv("GEMINI_REALTIME_MODEL", "gemini-3.1-flash-live-preview")

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

# Upload cap for the avatar video. The agent and the browser share one uplink
# to LiveKit Cloud, so an uncapped publish can starve the microphone upload.
# 800 kbps is ample for a mostly-static 720x720 talking head.
VIDEO_MAX_BITRATE = int(os.getenv("VIDEO_MAX_BITRATE", "800000"))

# Bisect switches. Plain Gemini answers in ~0.5s; this pipeline lags Gemini
# behind the microphone by ~7s and growing, and lowering the video bitrate made
# it worse, so it is not bandwidth. Turn the pieces off independently and read
# GEMINI LAG to find which one introduces it:
#   DISABLE_VIDEO=1 DISABLE_AVATAR=1  -> plain agent, the control
#   DISABLE_AVATAR=1                  -> video publishing only
#   DISABLE_VIDEO=1                   -> avatar audio path only
DISABLE_VIDEO = os.getenv("DISABLE_VIDEO", "0") == "1"
DISABLE_AVATAR = os.getenv("DISABLE_AVATAR", "0") == "1"
# mic_monitor opens a SECOND AudioStream on the user's track, which a plain
# agent does not do. Off makes the control truly minimal, at the cost of the
# GEMINI LAG line (fall back to comparing "heard you (final)" against when you
# actually spoke).
DISABLE_MIC_MONITOR = os.getenv("DISABLE_MIC_MONITOR", "0") == "1"

# Gemini's own VAD decides when your turn starts and ends; nothing local does
# it, because AgentSession is built without a VAD. Measured with a plain agent:
# utterances of 0.6-1.4s were marked as speaking for 3.3-10.7s, i.e. the turn
# was not CLOSING. So only the end-side knobs are tuned here.
#
# start sensitivity is deliberately left at Gemini's default: setting it LOW
# (an attempt to stop room noise opening a turn) made the model deaf - a clear
# 0.9s utterance at rms 3822 produced no turn at all. "DEFAULT" omits the field.
# OFF BY DEFAULT. Sending any realtime_input_config stopped Gemini responding
# at all - first with START_SENSITIVITY_LOW, then again with only the end-side
# fields set. Until that is understood, stock Gemini behaviour is the default
# so the agent works; set VAD_TUNE=1 to experiment.
VAD_TUNE = os.getenv("VAD_TUNE", "0") == "1"

# Local turn detection. Gemini's server VAD marked 0.6-1.4s utterances as
# speaking for 8-11s and fell further behind every turn, in a bare agent with
# none of this pipeline attached. A local VAD endpoints from audio the agent
# already holds - the mic monitor proves it arrives on time - instead of
# waiting on the signal that is late. LOCAL_VAD=0 reverts to Gemini's VAD.
LOCAL_VAD = os.getenv("LOCAL_VAD", "1") == "1" and silero is not None
# Silence before a turn is considered finished. Bangla speakers pause
# mid-sentence, so too small clips questions in half.
VAD_MIN_SILENCE = float(os.getenv("VAD_MIN_SILENCE", "0.55"))
VAD_ACTIVATION = float(os.getenv("VAD_ACTIVATION", "0.5"))

# Both of the next two were tried against the reply-latency growth described on
# GEMINI_MODEL, and NEITHER fixed it - the curve was unchanged with thinking off
# and with compression firing. The actual cause was the model family. They are
# kept because they are cheap and sane defaults for a long session, not because
# they solved anything; do not read them as the fix.
#
# push_audio() does forward every frame to Gemini including silence, so context
# really does grow at ~32 tokens/s of wall-clock time. That makes bounded
# context worth having regardless.
GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))  # -1 = model default

# Trigger must be small enough to actually fire: 16000 tokens is ~8 minutes of
# continuous audio, which no test session reached, so that first experiment was
# void. 4000/2000 engages within ~2 minutes. 0 disables.
CTX_TRIGGER_TOKENS = int(os.getenv("CTX_TRIGGER_TOKENS", "4000"))
CTX_TARGET_TOKENS = int(os.getenv("CTX_TARGET_TOKENS", "2000"))

# LiveKit delivers room audio at 24 kHz by default (AudioInputOptions.
# sample_rate), but the Gemini realtime plugin's input contract is 16 kHz
# (INPUT_AUDIO_SAMPLE_RATE). A 1.5x rate error makes the model treat one second
# of speech as 1.5 seconds, so it consumes input slower than real time and the
# backlog grows with session length - which is the shape measured in every
# configuration tonight, including a plain agent with none of this pipeline.
INPUT_SAMPLE_RATE = int(os.getenv("INPUT_SAMPLE_RATE", "16000"))

# Pin the transcription language. The plugin otherwise sends an EMPTY
# AudioTranscriptionConfig, which means auto-detect re-run on every utterance -
# and with 0.6-1.8s turns, Bangla/Hindi/Punjabi phonetic overlap, and English
# loanwords in the questions, the same speaker came back as Devanagari,
# Gurmukhi and Roman within one session. This only affects the transcript shown
# in the frontend; comprehension and replies were correct Bangla throughout.
# bn-BD, bn-IN and bn were all accepted by gemini-3.1-flash-live-preview.
INPUT_LANGUAGE = os.getenv("INPUT_LANGUAGE", "bn-BD")   # empty string = auto-detect
# Terms that repeatedly pulled the detector toward English/Hindi.
TRANSCRIPT_PHRASES = ["ChatGPT", "ডিপ লার্নিং", "মেশিন লার্নিং",
                      "নিউরাল নেটওয়ার্ক", "আর্কিটেকচার", "রেডওয়ান"]
VAD_START_SENSITIVITY = os.getenv("VAD_START", "DEFAULT").upper()
VAD_END_SENSITIVITY = os.getenv("VAD_END", "HIGH").upper()
VAD_SILENCE_MS = int(os.getenv("VAD_SILENCE_MS", "500"))
VAD_PREFIX_MS = int(os.getenv("VAD_PREFIX_MS", "300"))

END_MARKER = 0xFFFFFFFF
FLUSH_SENTINEL = b"__FLUSH__"

# Defined once and passed to BOTH RealtimeModel and Agent. They must not
# diverge: gemini-3.1-flash-live-preview reports mutable_instructions=False, so
# a mid-session instruction update is silently dropped and only the
# connect-time copy is guaranteed to reach the model.
#
# Written for speech, not for reading. Every line below exists because of
# something observed in testing: replies ran 10-15s (long render, long wait),
# the model appended "আপনি কি আরও জানতে চান?" to nearly every turn, and input
# transcription frequently garbles Bangla into Hindi or Roman script.
AGENT_INSTRUCTIONS = """You are Redwan, a Bangla-speaking AI assistant presented as a video avatar.

LANGUAGE
- Respond only in Bangla. Use the formal register (আপনি), not the familiar one.
- Keep established technical terms in their usual form when Bangla has no common
  equivalent, spoken naturally inside the Bangla sentence.
- The user speaks Bangla. Their words may reach you transcribed in Hindi, Roman
  or another script, or partly garbled. Interpret the intent and answer in
  Bangla. Never mirror the other script and never switch language.

SPEECH FORMAT
- Your reply is spoken aloud by an avatar. Produce plain spoken sentences only.
- No markdown, bullet points, numbering, emoji, parentheses or symbols.
- Write numbers, dates, units and years as words, not digits.
- Default to two to four sentences. Give a longer answer only when the user asks
  for detail, and then structure it as connected speech.

TONE
- Professional, composed and courteous. Neutral and factual, never effusive.
- Answer directly. No preamble, no restating the question, no flattery.
- Apologise once if genuinely warranted, then move on.
- End when the answer is complete. Do not append a follow-up question to every
  turn; ask one only when you genuinely need something from the user.

ACCURACY
- If you do not know, or are not confident, say so plainly in one short sentence.
- Never invent facts, figures, dates, names or sources.
- If a request is ambiguous or the transcription is too unclear to act on, ask a
  single short clarifying question instead of guessing.

SCOPE
- If asked what you are, say you are Redwan, an AI assistant.
- If a request falls outside what you can help with, say so briefly and, where
  possible, point to what you can do instead."""

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
    notified: bool = False              # playback completion reported upstream

    def played_seconds(self) -> float:
        """Audio actually heard so far, in seconds."""
        if self.t0 is None:
            return 0.0
        played = time.monotonic() - self.t0
        if self.nframes is not None:
            played = min(played, self.nframes / VIDEO_FPS)
        return max(0.0, played)


class AvatarStream:
    """Shared state between the receiver, audio pump, and video pump.

    Single event loop, no locks. Global frame numbers are continuous across
    utterances; each utterance records which range it owns.

    One LiveKit speech maps to exactly one Utterance: the framework calls
    flush() once per speech, which becomes an AudioSegmentEnd, our flush
    sentinel, the server's utterance_end, and finally on_utterance_end here.
    Because of that 1:1 mapping, reporting completion once per Utterance keeps
    the framework's speech bookkeeping exact.
    """

    def __init__(self, idle: Optional[IdleFrames], on_playback_done=None):
        self.idle = idle
        # Called with (playback_position_seconds, interrupted). LiveKit waits
        # on this to learn the agent stopped talking; without it the session
        # believes the agent speaks forever, treats every user turn as an
        # interruption, and stalls each one on a 5 s timeout.
        self.on_playback_done = on_playback_done
        self.video_buf: Dict[int, bytes] = {}       # global frame no -> jpg
        self.audio_q: asyncio.Queue = asyncio.Queue()   # np.int16 | None sentinel
        self.seg_base: Dict[int, int] = {}          # segment id -> global base
        self.next_global = 0
        self.utterances: deque[Utterance] = deque()
        self.resume_source_idx: Optional[int] = None
        self.idle_pos = 0                           # position in the idle cache
        self.need_align = True                      # send align before next audio
        self.awaiting_reset = False                 # drop packets until reset_done

    # -- receiver side --------------------------------------------------
    def on_audio_packet(self, seg_id: int, nframes: int, pcm: np.ndarray):
        if self.awaiting_reset:
            return
        if not self.utterances or self.utterances[-1].nframes is not None:
            self.utterances.append(Utterance(start=self.next_global))
        self.seg_base[seg_id] = self.next_global
        self.next_global += nframes
        self.audio_q.put_nowait(pcm)

    def on_frame_packet(self, seg_id: int, frame_idx: int, jpg: bytes):
        if self.awaiting_reset:
            return
        base = self.seg_base.get(seg_id)
        if base is not None:
            self.video_buf[base + frame_idx] = jpg

    def on_utterance_end(self, frames: int, end_source_idx: Optional[int]):
        if self.awaiting_reset:
            return
        closed = False
        for utt in self.utterances:
            if utt.nframes is None:
                utt.nframes = frames
                closed = True
                break
        if not closed:
            # A speech that produced no audio still owes the framework a
            # completion report. Stand in a finished zero-length utterance so
            # the normal retire path emits one instead of the session waiting
            # on an acknowledgement that never comes.
            self.utterances.append(Utterance(start=self.next_global, nframes=0,
                                             audio_done=True))
            logger.info("Utterance end with no pending audio; reporting empty speech")
        self.resume_source_idx = end_source_idx
        self.audio_q.put_nowait(None)               # delimiter for the pump

    # -- playback reporting ----------------------------------------------
    def _report(self, utt: Utterance, interrupted: bool):
        """Tell the framework this speech finished. Exactly once per utterance."""
        if utt.notified:
            return
        utt.notified = True
        if not self.on_playback_done:
            return
        try:
            self.on_playback_done(utt.played_seconds(), interrupted)
        except Exception:
            logger.exception("playback completion callback failed")

    def _retire(self, utt: Utterance, interrupted: bool):
        self._report(utt, interrupted)
        end = utt.start + (utt.nframes or 0)
        for g in [g for g in self.video_buf if g < end]:
            self.video_buf.pop(g, None)
        self.utterances.popleft()

    def abandon_all(self):
        """Interruption: drop everything pending, reporting what was heard."""
        while self.utterances:
            self._retire(self.utterances[0], interrupted=True)
        self.video_buf.clear()
        self.seg_base.clear()
        while not self.audio_q.empty():
            try:
                self.audio_q.get_nowait()
            except asyncio.QueueEmpty:
                break

    # -- shared ---------------------------------------------------------
    def active_utterance(self) -> Optional[Utterance]:
        while self.utterances:
            utt = self.utterances[0]
            if utt.t0 is None:
                # Never started. Retire only if it is closed and carried no
                # audio, so an empty speech still gets its completion report
                # instead of blocking the queue forever.
                if utt.audio_done and utt.nframes is not None and utt.samples_pushed == 0:
                    self._retire(utt, interrupted=False)
                    continue
                return None
            if utt.nframes is not None and utt.audio_done and \
               (time.monotonic() - utt.t0) * VIDEO_FPS >= utt.nframes:
                self._retire(utt, interrupted=False)
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
async def pcm_fanout(queue_audio, pcm_q_ws: asyncio.Queue, st: AvatarStream,
                     timer: Optional["TurnTimer"] = None):
    """Single consumer of QueueAudioOutput; forwards PCM to the server queue."""
    async for item in queue_audio:
        if AudioSegmentEnd is not None and isinstance(item, AudioSegmentEnd):
            await pcm_q_ws.put(FLUSH_SENTINEL)
            st.need_align = True                    # next utterance re-aligns
            continue
        pcm = np.frombuffer(bytes(item.data), dtype=np.int16)
        if pcm.size:
            if timer:
                timer.note_first_pcm()
            await pcm_q_ws.put(pcm.tobytes())


def install_gemini_send_probe():
    """Measure whether audio is stuck in OUR process waiting to upload.

    The plugin queues every outbound message - audio frames AND the
    end-of-turn signal - through one channel (_msg_ch) drained by a single
    websocket send task. If the upload runs slower than real time, the
    end-of-turn queues behind the backlog and what we log as "generation"
    is actually upload deficit. Sampling the queue depth splits the two
    remaining suspects: backlog grows -> our upload path to Google is too
    slow; backlog stays ~0 -> the audio is already at Google and the delay
    is server-side.

    Read-only: wraps push_audio, samples qsize every few seconds, never
    raises into the caller.
    """
    try:
        from livekit.plugins.google.realtime import realtime_api as ra
    except ImportError:
        return
    orig = ra.RealtimeSession.push_audio
    if getattr(orig, "_lag_probe", False):
        return
    state = {"last": 0.0, "peak": 0}

    def push_audio(self, frame):
        orig(self, frame)
        try:
            n = self._msg_ch.qsize()
            state["peak"] = max(state["peak"], n)
            now = time.monotonic()
            if now - state["last"] >= 5.0:
                state["last"] = now
                # each queued message is ~50 ms of audio
                level = logger.warning if state["peak"] > 20 else logger.info
                level(f"⏱  gemini send queue: now={n} peak={state['peak']} "
                      f"(~{state['peak'] * 0.05:.1f}s of audio waiting to upload)")
                state["peak"] = 0
        except Exception:
            pass

    push_audio._lag_probe = True
    ra.RealtimeSession.push_audio = push_audio
    logger.info("Gemini send-queue probe installed")


async def loop_lag_monitor(interval: float = 0.25, warn_s: float = 0.15):
    """Measure event-loop scheduling delay.

    Everything in this process shares one loop, including livekit-agents'
    forwarding of the user's microphone to Gemini. If our own tasks block it,
    that audio is handed over late and speech is recognised seconds after it
    was spoken. Sleep a known interval and report the overshoot.
    """
    worst = 0.0
    last_report = time.monotonic()
    while True:
        t0 = time.monotonic()
        await asyncio.sleep(interval)
        lag = time.monotonic() - t0 - interval
        worst = max(worst, lag)
        if lag > warn_s:
            logger.warning(f"⏱  EVENT LOOP STALLED {lag * 1000:.0f}ms")
        now = time.monotonic()
        if now - last_report >= 10.0:
            level = logger.warning if worst > warn_s else logger.info
            level(f"⏱  loop lag: worst {worst * 1000:.0f}ms over 10s")
            worst, last_report = 0.0, now


# Ground truth for "is the user actually talking", independent of Gemini's VAD.
MIC_RMS_SPEECH = 350          # int16 RMS treated as speech (~-39 dBFS)
MIC_ONSET_S = 0.12            # sustained level before calling it speech
MIC_HANGOVER_S = 0.60         # silence before calling the utterance over
MIC_REPORT_S = 5.0            # idle level report cadence


async def mic_monitor(room: rtc.Room, timer: Optional["TurnTimer"] = None):
    """Log when the user's microphone actually carries speech.

    Gemini's server VAD drives user_state, so when it stays silent we cannot
    tell whether the audio never arrived or arrived and was not detected. This
    measures the track directly at the agent, upstream of Gemini, and reports
    the level periodically so a dead mic is distinguishable from a quiet one.
    """
    streams: Dict[str, asyncio.Task] = {}

    async def pump(track: rtc.Track, who: str):
        stream = rtc.AudioStream(track)
        speaking = False
        above_since: Optional[float] = None
        below_since: Optional[float] = None
        started: Optional[float] = None
        last_report = time.monotonic()
        peak = 0.0
        try:
            async for ev in stream:
                buf = np.frombuffer(bytes(ev.frame.data), dtype=np.int16)
                if buf.size == 0:
                    continue
                rms = float(np.sqrt(np.mean(buf.astype(np.float32) ** 2)))
                now = time.monotonic()
                peak = max(peak, rms)

                if rms >= MIC_RMS_SPEECH:
                    below_since = None
                    above_since = above_since or now
                    if not speaking and now - above_since >= MIC_ONSET_S:
                        speaking, started = True, above_since
                        logger.info(f"🎙  MIC: {who} started talking (rms {rms:.0f})")
                        if timer:
                            timer.note_mic_speech(above_since)
                else:
                    above_since = None
                    below_since = below_since or now
                    if speaking and now - below_since >= MIC_HANGOVER_S:
                        speaking = False
                        spoke = (below_since - started) if started else 0.0
                        logger.info(f"🎙  MIC: {who} stopped talking "
                                    f"(spoke {spoke:.1f}s)")

                if not speaking and now - last_report >= MIC_REPORT_S:
                    logger.info(f"🎙  MIC idle: peak rms {peak:.0f} over last "
                                f"{MIC_REPORT_S:.0f}s (threshold {MIC_RMS_SPEECH})")
                    last_report, peak = now, 0.0
        except Exception:
            logger.exception("mic monitor stopped")

    def maybe_watch(track, publication, participant):
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if participant.identity in streams:
            return
        logger.info(f"🎙  monitoring microphone of {participant.identity}")
        streams[participant.identity] = asyncio.create_task(
            pump(track, participant.identity))

    room.on("track_subscribed", maybe_watch)
    # tracks already subscribed before this task started
    for p in room.remote_participants.values():
        for pub in p.track_publications.values():
            if pub.track:
                maybe_watch(pub.track, pub, p)

    await asyncio.Event().wait()          # run until cancelled


class TurnTimer:
    """Splits reply latency into the segments that can own it.

    Anchored on "user started speaking", not "stopped": with the avatar playing
    out loud the mic hears it, so the framework keeps the user marked as
    speaking long after they went quiet and "stopped" arrives far too late to
    measure against. When the agent starts replying before the user is even
    marked as stopped, that is reported as an echo warning - it means Gemini is
    being fed the avatar's own voice as user speech.

    This realtime model never enters the "thinking" state (it goes listening ->
    speaking directly), so endpointing and generation cannot be separated here;
    they are reported together as the reply latency.
    """

    def __init__(self):
        self.user_started: Optional[float] = None
        self.user_stopped: Optional[float] = None
        self.agent_speaking: Optional[float] = None
        self.first_pcm: Optional[float] = None
        self.replied_before_stop = False
        self.playback_ended: Optional[float] = None
        self.mic_speech_at: Optional[float] = None

    def _reset(self):
        self.user_stopped = None
        self.agent_speaking = None
        self.first_pcm = None
        self.replied_before_stop = False

    def on_user_state(self, old: str, new: str):
        now = time.monotonic()
        if new == "speaking":
            self.user_started = now
            # With local VAD this fires before the mic monitor trips, so the
            # old "GEMINI LAG vs microphone" number compared against a stale
            # event from the previous turn and read 19-28s. Dropped; the real
            # number is generation time, reported on the agent transition.
            self.mic_speech_at = None
            self._reset()
            # The interval nobody has accounted for yet: avatar goes quiet,
            # then how long until the system registers you talking again? If
            # you spoke promptly and this is large, speech is not being picked
            # up; if it matches how long you took to think, it is not a fault.
            if self.playback_ended is not None:
                logger.info(f"⏱  silence gap: {now - self.playback_ended:.1f}s "
                            "(avatar finished -> you registered as speaking)")
            logger.info("⏱  user started speaking")
        elif old == "speaking":
            self.user_stopped = now
            spoke = now - self.user_started if self.user_started else 0.0
            # Not an echo signal: with server-side VAD the model ends the turn
            # and starts replying while this state lags several seconds behind.
            note = "  (reply began before the turn was marked finished)" \
                if self.replied_before_stop else ""
            logger.info(f"⏱  user stopped speaking (marked speaking for "
                        f"{spoke:.1f}s){note}")

    def note_playback_finished(self):
        self.playback_ended = time.monotonic()

    def note_mic_speech(self, at: float):
        """Microphone carried speech at `at`, measured before Gemini sees it."""
        if self.mic_speech_at is None:
            self.mic_speech_at = at

    def on_agent_state(self, old: str, new: str):
        now = time.monotonic()
        # Every transition, not just "speaking". If the session lingers in
        # "speaking" after playback has been reported finished, it is not
        # listening to the user - which is exactly the window where speech
        # goes unregistered.
        extra = ""
        if new == "listening" and self.playback_ended is not None:
            extra = f"  ({now - self.playback_ended:.1f}s after playback finished)"
        logger.info(f"⏱  agent state: {old} -> {new}{extra}")

        if new != "speaking":
            return
        self.agent_speaking = now
        if self.user_started:
            logger.info(f"⏱  reply latency: {now - self.user_started:.2f}s "
                        "(you started speaking -> first response audio)")
        if self.user_stopped is not None:
            # The number that now dominates: your turn was committed and this
            # is how long Gemini took to produce anything. Local VAD made the
            # commit prompt, so anything large here is generation, not
            # endpointing.
            logger.warning(f"⏱  GEMINI GENERATION: {now - self.user_stopped:.1f}s "
                           "(turn committed -> first response audio)")
        else:
            self.replied_before_stop = True

    def note_first_pcm(self):
        """First response audio handed to us by the framework."""
        if self.first_pcm is not None:
            return
        self.first_pcm = time.monotonic()

    def note_playback_start(self):
        now = time.monotonic()
        if self.first_pcm:
            logger.info(f"⏱  avatar pipeline: {now - self.first_pcm:.2f}s "
                        "(first audio in -> playback start)")
        if self.user_started:
            logger.info(f"⏱  TOTAL you-started-speaking -> avatar-speaks: "
                        f"{now - self.user_started:.2f}s")


async def handle_interruption(st: AvatarStream, pcm_q_ws: asyncio.Queue, ws_audio):
    """Barge-in: abandon the current turn everywhere it is buffered.

    LiveKit calls clear_buffer() on the audio output when the user interrupts;
    it drains its own channel and emits "clear_buffer". Everything downstream
    of that channel is ours to discard: audio still queued for the server, the
    server's own buffers, and the frames and PCM already sent to this client.
    Reporting what was actually heard lets the framework trim the transcript
    to the point of interruption.
    """
    st.abandon_all()
    st.awaiting_reset = True
    st.need_align = True

    while not pcm_q_ws.empty():
        try:
            pcm_q_ws.get_nowait()
        except asyncio.QueueEmpty:
            break

    try:
        await ws_audio.send(json.dumps({"type": "reset"}))
        logger.info("Interrupted: sent reset to avatar server")
    except Exception as err:
        logger.warning(f"Could not send reset: {err}")
        st.awaiting_reset = False     # no ack is coming; do not stall the stream


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
          {"type": "reset_done"}
    """
    while True:
        msg = await ws_video.recv()
        if isinstance(msg, str):
            try:
                js = json.loads(msg)
            except ValueError:
                continue
            kind = js.get("type")
            if kind == "utterance_end":
                st.on_utterance_end(int(js.get("frames", 0)),
                                    js.get("end_source_idx"))
            elif kind == "reset_done":
                # Everything the server had queued before this marker belongs
                # to the interrupted turn; from here the stream is clean.
                st.awaiting_reset = False
                logger.info("Server reset acknowledged; stream resumed")
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
async def audio_pump(st: AvatarStream, audio_source: rtc.AudioSource,
                     timer: Optional["TurnTimer"] = None):
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
            # starts in the past and the pump skips a burst of frames. An
            # interruption can drop utt from the deque while we wait, so leave
            # as soon as it is no longer queued rather than spinning forever.
            while utt in st.utterances and st.utterances[0] is not utt:
                st.active_utterance()      # lets finished utterances retire
                await asyncio.sleep(0.05)
            if utt not in st.utterances:
                continue                              # abandoned mid-wait

            gate_t0 = time.monotonic()
            while (sum(1 for g in st.video_buf if g >= utt.start) < GATE_FRAMES
                   and time.monotonic() - gate_t0 < GATE_TIMEOUT_S):
                await asyncio.sleep(0.02)
            utt.t0 = time.monotonic()
            logger.info(f"▶️ Utterance started (global frame {utt.start})")
            if timer:
                timer.note_playback_start()

        pending = pcm
        while pending.size > 0:
            if utt not in st.utterances:
                break                                 # interrupted; stop pushing
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

    def _to_i420(bgr: np.ndarray, blend_with: Optional[np.ndarray],
                 alpha: float) -> tuple:
        if blend_with is not None:
            bgr = cv2.addWeighted(bgr, alpha, blend_with, 1.0 - alpha, 0)
        h, w = bgr.shape[:2]
        return bgr, w, h, cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).tobytes()

    async def publish(bgr: np.ndarray):
        nonlocal last_bgr, crossfade_left
        blend_with, alpha = None, 1.0
        if crossfade_left > 0 and last_bgr is not None and \
                last_bgr.shape == bgr.shape:
            alpha = 1.0 - crossfade_left / (CROSSFADE_FRAMES + 1)
            blend_with = last_bgr
            crossfade_left -= 1
        else:
            crossfade_left = 0

        bgr, w, h, i420 = await asyncio.to_thread(_to_i420, bgr, blend_with, alpha)
        last_bgr = bgr
        vf = rtc.VideoFrame(width=w, height=h,
                            type=rtc.VideoBufferType.I420, data=i420)
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
    install_gemini_send_probe()

    logger.info("Connecting LiveKit...")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    logger.info(f"MODE: video={'off' if DISABLE_VIDEO else 'on'} "
                f"avatar={'off' if DISABLE_AVATAR else 'on'}")

    w, h, idle, sid = 0, 0, None, None
    ws_audio_url = ws_video_url = None
    video_source = audio_source = None

    if not DISABLE_AVATAR:
        w, h = get_avatar_size()
        logger.info(f"Avatar server image_size: {w}x{h}")
        sid = await asyncio.to_thread(create_session)
        logger.info(f"Created avatar session: {sid}")
        idle = await fetch_idle_frames(AVATAR_BASE)

        ws_base = AVATAR_BASE.replace("http://", "ws://").replace("https://", "wss://")
        ws_audio_url = f"{ws_base}/ws/audio/{sid}"
        ws_video_url = f"{ws_base}/ws/video/{sid}"

        audio_source = rtc.AudioSource(sample_rate=ASSISTANT_SR, num_channels=1)
        await ctx.room.local_participant.publish_track(
            create_local_audio_track(AUDIO_TRACK_NAME, audio_source),
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        logger.info(f"Published audio track: {AUDIO_TRACK_NAME}")

    if not DISABLE_VIDEO:
        if not w:
            w, h = get_avatar_size()
        video_source = rtc.VideoSource(w, h)
        await ctx.room.local_participant.publish_track(
            create_local_video_track(VIDEO_TRACK_NAME, video_source),
            rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_CAMERA,
                video_encoding=rtc.VideoEncoding(
                    max_bitrate=VIDEO_MAX_BITRATE, max_framerate=VIDEO_FPS),
            ))
        logger.info(f"Published video track: {VIDEO_TRACK_NAME} "
                    f"(capped {VIDEO_MAX_BITRATE / 1000:.0f} kbps / {VIDEO_FPS} fps)")

    realtime_input = None
    if VAD_TUNE:
        # Each field is only set when explicitly asked for; anything left as
        # DEFAULT keeps Gemini's own behaviour, so a bad value cannot silence
        # the model the way START_SENSITIVITY_LOW did.
        detect = genai_types.AutomaticActivityDetection(
            prefix_padding_ms=VAD_PREFIX_MS,
            silence_duration_ms=VAD_SILENCE_MS,
        )
        if VAD_START_SENSITIVITY != "DEFAULT":
            detect.start_of_speech_sensitivity = getattr(
                genai_types.StartSensitivity,
                f"START_SENSITIVITY_{VAD_START_SENSITIVITY}")
        if VAD_END_SENSITIVITY != "DEFAULT":
            detect.end_of_speech_sensitivity = getattr(
                genai_types.EndSensitivity,
                f"END_SENSITIVITY_{VAD_END_SENSITIVITY}")
        realtime_input = genai_types.RealtimeInputConfig(
            automatic_activity_detection=detect)
        logger.info(f"Gemini VAD: start={VAD_START_SENSITIVITY} "
                    f"end={VAD_END_SENSITIVITY} silence={VAD_SILENCE_MS}ms "
                    f"prefix={VAD_PREFIX_MS}ms")

    # Only pass realtime_input_config when we actually have one: the plugin
    # dereferences it without a None check, so an explicit None crashes on
    # construction.
    model_kwargs = {}
    if realtime_input is not None:
        model_kwargs["realtime_input_config"] = realtime_input
    if INPUT_LANGUAGE:
        model_kwargs["input_audio_transcription"] = (
            genai_types.AudioTranscriptionConfig(
                language_codes=[INPUT_LANGUAGE],
                adaptation_phrases=TRANSCRIPT_PHRASES,
            ))
        logger.info(f"Input transcription pinned to {INPUT_LANGUAGE} "
                    f"({len(TRANSCRIPT_PHRASES)} adaptation phrases)")
    else:
        logger.info("Input transcription: auto-detect (may switch script per turn)")
    if CTX_TRIGGER_TOKENS > 0:
        model_kwargs["context_window_compression"] = (
            genai_types.ContextWindowCompressionConfig(
                trigger_tokens=CTX_TRIGGER_TOKENS,
                sliding_window=genai_types.SlidingWindow(
                    target_tokens=CTX_TARGET_TOKENS),
            ))
        logger.info(f"Context compression: trigger={CTX_TRIGGER_TOKENS} "
                    f"target={CTX_TARGET_TOKENS} tokens "
                    f"(fires after ~{CTX_TRIGGER_TOKENS // 32}s of audio)")
    if GEMINI_THINKING_BUDGET >= 0:
        model_kwargs["thinking_config"] = genai_types.ThinkingConfig(
            thinking_budget=GEMINI_THINKING_BUDGET)
        logger.info(f"Gemini thinking budget: {GEMINI_THINKING_BUDGET}"
                    f"{' (thinking disabled)' if GEMINI_THINKING_BUDGET == 0 else ''}")

    model = google.realtime.RealtimeModel(
        model=GEMINI_MODEL,
        voice="Puck",
        temperature=0.7,
        **model_kwargs,
        instructions=AGENT_INSTRUCTIONS,
    )
    agent = Agent(llm=model, instructions=AGENT_INSTRUCTIONS)
    # proc.userdata["vad"] is loaded once per worker by prewarm; falling back to
    # a direct load keeps this working if the entrypoint is driven some other way.
    vad = None
    if LOCAL_VAD:
        vad = getattr(ctx.proc, "userdata", {}).get("vad") if hasattr(ctx, "proc") else None
        if vad is None:
            vad = silero.VAD.load(min_silence_duration=VAD_MIN_SILENCE,
                                  activation_threshold=VAD_ACTIVATION)
        logger.info(f"Turn detection: LOCAL silero VAD "
                    f"(min_silence={VAD_MIN_SILENCE}s activation={VAD_ACTIVATION})")
    else:
        logger.info("Turn detection: Gemini server VAD")

    session = AgentSession(vad=vad) if vad is not None else AgentSession()

    queue_audio = QueueAudioOutput(sample_rate=ASSISTANT_SR)

    # The framework learns the agent stopped speaking only when the audio sink
    # reports it. QueueAudioOutput exposes notify_playback_finished for exactly
    # this; the avatar owns playback timing, so the report has to come from us.
    timer = TurnTimer()

    def report_playback_done(position: float, interrupted: bool):
        logger.info(f"Playback finished: {position:.2f}s "
                    f"{'(interrupted)' if interrupted else ''}".rstrip())
        timer.note_playback_finished()
        queue_audio.notify_playback_finished(playback_position=position,
                                             interrupted=interrupted)

    st = AvatarStream(idle, on_playback_done=report_playback_done)

    session.on("user_state_changed",
               lambda e: timer.on_user_state(e.old_state, e.new_state))
    session.on("agent_state_changed",
               lambda e: timer.on_agent_state(e.old_state, e.new_state))
    session.on("user_input_transcribed",
               lambda e: logger.info(f"⏱  heard you{' (final)' if e.is_final else ''}: "
                                     f"{e.transcript[:60]}") if e.is_final else None)

    if not DISABLE_AVATAR:
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
    # With the avatar off the session publishes its own audio to the room, so
    # the control run behaves like a plain Gemini agent.
    try:
        audio_in = room_io.AudioInputOptions(sample_rate=INPUT_SAMPLE_RATE)
        logger.info(f"Input audio: {INPUT_SAMPLE_RATE} Hz "
                    "(Gemini realtime expects 16000)")
    except Exception:
        audio_in = True
        logger.warning("AudioInputOptions unavailable; using the default rate")

    try:
        opts = room_io.RoomOptions(
            audio_input=audio_in,
            audio_output=DISABLE_AVATAR,
            text_input=True,
            text_output=text_out,
            video_input=False,
            close_on_disconnect=False,   # keep alive if Playground disconnects
        )
    except TypeError:
        opts = room_io.RoomOptions(
            audio_input=True, audio_output=DISABLE_AVATAR,
            text_input=True, text_output=True, video_input=False)

    await session.start(room=ctx.room, agent=agent, room_options=opts)

    # Re-attach AFTER start: if session.start wrapped or replaced the audio
    # output (transcript synchronizer, room IO), this restores our queue as
    # the direct sink for every subsequent response. Only when the avatar is
    # driving playback - with DISABLE_AVATAR the session must keep RoomIO,
    # otherwise its audio goes into a queue nobody drains and every speech
    # stalls on the 5 s playback timeout.
    if not DISABLE_AVATAR and hasattr(session, "output") and hasattr(session.output, "audio"):
        if session.output.audio is not queue_audio:
            logger.info("Audio output was wrapped by session.start; re-attaching queue.")
            session.output.audio = queue_audio

    logger.info(f"Gemini session started (model={GEMINI_MODEL}).")

    pcm_q_ws = asyncio.Queue(maxsize=2000)
    # Measurement always runs; it is what the bisect is read from.
    tasks = [asyncio.create_task(loop_lag_monitor(), name="loop_lag_monitor")]
    if not DISABLE_MIC_MONITOR:
        tasks.append(asyncio.create_task(mic_monitor(ctx.room, timer),
                                         name="mic_monitor"))
    stack = contextlib.AsyncExitStack()

    async def supervise():
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

    async with stack:
        if not DISABLE_AVATAR:
            wsa = await stack.enter_async_context(
                websockets.connect(ws_audio_url, max_size=None))
            wsv = await stack.enter_async_context(
                websockets.connect(ws_video_url, max_size=None))
            logger.info("Connected to GPU avatar server. Audio-master playback active.")

            # clear_buffer fires on barge-in. The handler is sync, so hand the
            # async teardown to the loop and keep a reference so it is not
            # garbage-collected mid-flight.
            pending_interrupts: set = set()

            def on_clear_buffer():
                task = asyncio.create_task(handle_interruption(st, pcm_q_ws, wsa))
                pending_interrupts.add(task)
                task.add_done_callback(pending_interrupts.discard)

            queue_audio.on("clear_buffer", on_clear_buffer)

            tasks += [
                asyncio.create_task(pcm_fanout(queue_audio, pcm_q_ws, st, timer),
                                    name="pcm_fanout"),
                asyncio.create_task(ws_send_pcm(pcm_q_ws, wsa, st), name="ws_send_pcm"),
                asyncio.create_task(receive_stream(wsv, st), name="receive_stream"),
                asyncio.create_task(audio_pump(st, audio_source, timer), name="audio_pump"),
            ]

        if not DISABLE_VIDEO:
            tasks.append(asyncio.create_task(video_pump(st, video_source),
                                             name="video_pump"))

        try:
            await supervise()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    required = ["GOOGLE_API_KEY", "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")

    def prewarm(proc):
        """Load the VAD model once per worker, not once per conversation."""
        if LOCAL_VAD:
            proc.userdata["vad"] = silero.VAD.load(
                min_silence_duration=VAD_MIN_SILENCE,
                activation_threshold=VAD_ACTIVATION)

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
