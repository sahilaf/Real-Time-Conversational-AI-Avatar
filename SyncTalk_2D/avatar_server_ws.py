"""
avatar_server_ws.py — WS Avatar Server for SyncTalk_2D (SYNCHRONIZED)

Protocol:
  POST /session -> {"session_id": "..."}
  WS /ws/audio/{sid} : client sends WAV bytes (one chunk per message)
  WS /ws/video/{sid} : server sends binary frames:
        [4B segment_id][4B frame_index][4B total_frames][4B audio_duration_ms] + [jpg bytes]
        
        Special end-of-segment marker:
        [4B segment_id][4B 0xFFFFFFFF][4B total_frames][4B audio_duration_ms] + [audio_pcm_bytes]

Key features:
✅ Segment-based processing for perfect lip sync
✅ Client knows exactly how many frames per segment
✅ Audio PCM sent with end-of-segment for synchronized playback
✅ Constant 25 FPS output when played back
"""

import os
import io
import time
import uuid
import asyncio
import argparse
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional, List

import cv2
import numpy as np
import torch
import soundfile as sf
from torch.utils.data import DataLoader

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
import uvicorn

from unet_328 import Model
from scipy.signal import resample_poly

from utils import (AudioEncoder, AudDataset, get_audio_features as _get_audio_features,
                   apply_mouth_mask, read_mask_version, blend_bottom_edge,
                   melspectrogram)


def log(*args):
    """Timestamped, unbuffered server log.

    The agent logs wall-clock times too; without stamps here the two streams
    cannot be lined up, which is what made the first latency hunt guesswork.
    """
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


# -----------------------------
# App / globals (models)
# -----------------------------
app = FastAPI()

audio_encoder: Optional[AudioEncoder] = None
synctalk_model: Optional[Model] = None
device = None
mode = "ave"

dataset_dir = None
img_dir = None
lms_dir = None
len_img = 0
img_h = 0
img_w = 0

# -----------------------------
# Idle Cache for lifelike animation
# -----------------------------
import pickle
import json
import hashlib

IDLE_CACHE_DIR = "idle_cache"
IDLE_DURATION_SECONDS = 4.0  # 4 seconds of idle animation
IDLE_FPS = 25

# Bumped whenever idle generation changes, so stale caches are rebuilt instead
# of silently reused.
#   v2: real silence features instead of zeros, closed-mouth reference frames
#   v3: reference frames must be CONTIGUOUS (v2 scattered them and the head
#       appeared to skip between unrelated poses)
#   v4: idle uses the ORIGINAL frames, no model inference at all
IDLE_CACHE_VERSION = 4

# Idle needs no lip-sync, and the source video already contains real footage of
# this person sitting quietly. Running the generator over it can only
# approximate what is already there - and this generator was trained without
# working sync supervision, so its mouth is imperfect for any input, silence
# included. Playing the real frames is exact by construction.
#
# The crop/resize round-trip is kept identical to the speech path so the only
# difference between idle and generated frames is the mouth pixels themselves;
# that keeps the transition from popping.
IDLE_USE_RAW_FRAMES = True

# Inner-lip landmarks in this 110-point layout. 90-101 is the outer lip ring,
# 102-109 the inner one; the gap between the inner rows is mouth aperture.
INNER_LIP_UPPER = [103, 104, 105]
INNER_LIP_LOWER = [107, 108, 109]

# Populated at startup by initialize_idle_inputs()
silence_feats: Optional[np.ndarray] = None      # encoded TRUE silence, not zeros
idle_ref_indices: List[int] = []                # a contiguous closed-mouth run

# --- Reference-frame leakage -------------------------------------------------
# Training feeds channels 0-2 a RANDOM DIFFERENT frame (datasetsss_328.py picks
# ex_int at random) and channels 3-5 the current frame with its mouth blacked
# out. Inference instead put the CURRENT frame in channels 0-2, mouth visible.
#
# So at inference the reference channels carry the source video's real mouth,
# and because ping-pong walks through footage of the person talking, those
# channels change every frame. The generated mouth follows them - which is why
# the avatar appears to speak even when the audio is silent.
#
# Holding the reference fixed restores the training distribution (a frame that
# is not the current one) and leaves audio as the only varying driver.
USE_FIXED_REFERENCE = True

# Set from the checkpoint at startup; must match how the model was trained.
MASK_VERSION = "legacy"

# Rows over which the generated crop fades back to the source at its bottom
# edge. The v2 mask leaves no unmasked strip there, so a hard paste puts the
# invented jaw straight against untouched source. 0 restores the hard paste.
FEATHER_ROWS = 16
fixed_ref_tensor: Optional[torch.Tensor] = None   # [3,320,320], set at startup
fixed_ref_index: Optional[int] = None             # which source frame it came from

# End-of-batch marker in the frame_index header field (also carries the PCM).
END_MARKER = 0xFFFFFFFF

# Source frames the speech walk may use as base footage. Set to the idle run at
# startup so idle and speech share the same footage and transitions stay
# pose-continuous; falls back to the whole video if no run was found.
speech_window: List[int] = []

# Idle cache position -> source frame index, so the client can hand the walk
# position back and forth across idle/speech transitions.
idle_source_map: List[int] = []

# Decoded source frames for the speech window. The window is ~100-160 frames,
# so this trades a few hundred MB of RAM for removing JPEG decode (~10 ms) from
# every generated frame. Populated lazily; capped.
FRAME_CACHE_MAX = 170
_frame_cache: Dict[int, tuple] = {}   # ref_index -> (img_bgr, lms)

# Output frames are resized to this before JPEG encoding (longest side).
# 0 = native. Encoding 1080px costs ~10 ms/frame of a 40 ms budget; 720px is
# indistinguishable in the browser at typical layout sizes.
OUT_SIZE = 720

@dataclass
class IdleCache:
    """Pre-generated idle animation frames for smooth playback during processing"""
    frames: List[bytes]  # List of jpg bytes
    frame_count: int
    img_w: int
    img_h: int
    
    def get_frame(self, index: int) -> bytes:
        """Get frame at index, wrapping around for looping"""
        return self.frames[index % len(self.frames)]

idle_cache: Optional[IdleCache] = None


# -----------------------------
# Session state
# -----------------------------
@dataclass
class SessionState:
    sid: str
    audio_q: asyncio.Queue         # wav bytes chunks
    frame_q: asyncio.Queue         # segment packets (header + data)
    closed: asyncio.Event
    img_idx: int = 0
    step_stride: int = 1
    last_audio_time: float = 0.0
    frames_generated: int = 0
    segment_id: int = 0            # Counter for audio segments
    audio_connected: bool = False
    video_connected: bool = False
    reset_requested: bool = False  # set by a client "reset" control message


sessions: Dict[str, SessionState] = {}


# -----------------------------
# SyncTalk helpers
# -----------------------------
def _pingpong_next(sess: SessionState) -> int:
    """Advance the walk one step and return the SOURCE frame index.

    sess.img_idx is a position within speech_window, not a source index.
    Must only be called from the (single, sequential) generation path - it
    mutates shared state.
    """
    n = len(speech_window)
    if n == 0:
        return 0
    if n == 1:
        return speech_window[0]

    if sess.img_idx >= n - 1:
        sess.step_stride = -1
    if sess.img_idx <= 0:
        sess.step_stride = 1

    sess.img_idx = max(0, min(sess.img_idx + sess.step_stride, n - 1))
    return speech_window[sess.img_idx]


def _align_walk(sess: SessionState, source_idx: int):
    """Position the walk at the window frame nearest to source_idx.

    Called when the client reports where its idle loop currently is, so the
    first speech frame continues from the same footage instead of jumping.
    """
    if not speech_window:
        return
    pos = min(range(len(speech_window)),
              key=lambda i: abs(speech_window[i] - source_idx))
    sess.img_idx = pos
    sess.step_stride = 1 if pos < len(speech_window) - 1 else -1


def _read_source_frame(ref_index: int):
    """Decoded frame + landmarks, cached for the speech window."""
    hit = _frame_cache.get(ref_index)
    if hit is not None:
        return hit

    img = cv2.imread(os.path.join(img_dir, f"{ref_index}.jpg"))
    if img is None:
        raise RuntimeError(f"Failed to read image: {ref_index}.jpg")
    lms = _load_landmarks(os.path.join(lms_dir, f"{ref_index}.lms"))

    if len(_frame_cache) < FRAME_CACHE_MAX:
        _frame_cache[ref_index] = (img, lms)
    return img, lms


def _load_landmarks(path: str) -> np.ndarray:
    pts = []
    with open(path, "r") as f:
        for line in f.read().splitlines():
            pts.append(np.array(line.split(" "), dtype=np.float32))
    return np.array(pts, dtype=np.int32)


def _process_audio_to_features(wav_path: str) -> np.ndarray:
    """
    Process audio file to features using AudDataset -> DataLoader -> AudioEncoder
    """
    ds = AudDataset(wav_path)
    try:
        if len(ds) == 0:
            return np.zeros((3, 8192), dtype=np.float32)
    except Exception:
        return np.zeros((3, 8192), dtype=np.float32)

    loader = DataLoader(ds, batch_size=64, shuffle=False)

    outs = []
    for mel in loader:
        if mel is None:
            continue
        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel)
        if mel.numel() == 0:
            continue

        mel = mel.to(device)
        with torch.no_grad():
            out = audio_encoder(mel)
        outs.append(out.cpu())

    if not outs:
        return np.zeros((3, 8192), dtype=np.float32)

    outputs = torch.cat(outs, dim=0)
    first_frame, last_frame = outputs[:1], outputs[-1:]
    audio_feats = torch.cat([first_frame, outputs, last_frame], dim=0)
    return audio_feats.numpy()


class StreamingFeatureExtractor:
    """Incremental AVE features that match offline extraction.

    Offline (training and inference_328): resample to 16 kHz, mel spectrogram
    (hop 200, win 800, centred), one 16-mel window per video frame starting at
    mel int(80*j/25), AudioEncoder, then [first | feats | last] edge
    duplication - a convention baked into the aud_ave.npy files the model was
    trained on. A streaming path that ignores any of this shifts or corrupts
    the audio the generator sees, and the mouth stops matching the speech.

    This produces the same numbers incrementally:
      - resampling runs over the whole utterance each time with a fixed
        phase, so chunk boundaries cannot drift the timeline;
      - mel is recomputed from a mel-hop-aligned tail with CTX_MEL context
        frames, so window values never depend on where a chunk started;
      - a feature is only produced once all of its audio exists (window end
        m*200 + 400 <= len16), so no edge-padded mel is ever consumed.

    raw[j] here corresponds to offline outputs[j]; the caller applies the
    [first | raw | last] convention when building windows.
    """
    SR24 = 24000
    SR16 = 16000
    MEL_HOP = 200                 # samples per mel frame at 16 kHz
    MEL_HALF = 400                # centred window: mel m spans m*200 +/- 400
    WIN = 16                      # mel frames per feature window
    CTX_MEL = 8                   # context mels absorbing chunk-edge effects

    def __init__(self):
        self.pcm24 = np.zeros(0, dtype=np.int16)
        self.raw: Optional[np.ndarray] = None      # [n, 512] feature frames

    @property
    def n_raw(self) -> int:
        return 0 if self.raw is None else len(self.raw)

    @property
    def n_audio_frames(self) -> int:
        """Complete 40 ms video frames of audio received."""
        return len(self.pcm24) // (self.SR24 // 25)

    def add(self, pcm24: np.ndarray):
        self.pcm24 = np.concatenate([self.pcm24, pcm24])

    def extract(self):
        """Compute every feature frame the buffered audio now supports."""
        n16 = int(len(self.pcm24) * 2) // 3
        m_valid = (n16 - self.MEL_HALF) // self.MEL_HOP + 1 if n16 >= self.MEL_HALF else 0

        r0, r_new = self.n_raw, self.n_raw
        while int(80 * r_new / 25) + self.WIN <= m_valid:
            r_new += 1
        if r_new == r0:
            return

        wav16 = resample_poly(self.pcm24.astype(np.float32) / 32768.0, 2, 3)

        # mel-hop-aligned start, with context so reflection padding and the
        # pre-emphasis transient decay before any window in use
        s0 = max(0, int(80 * r0 / 25) - self.CTX_MEL)
        mel = melspectrogram(wav16[s0 * self.MEL_HOP:]).T    # [frames, 80]

        wins = []
        for j in range(r0, r_new):
            a = int(80 * j / 25) - s0
            wins.append(mel[a:a + self.WIN].T)               # [80, 16]
        batch = torch.from_numpy(np.stack(wins).astype(np.float32)).unsqueeze(1)
        with torch.no_grad():
            out = audio_encoder(batch.to(device)).cpu().numpy()
        self.raw = out if self.raw is None else np.concatenate([self.raw, out])

    def finalize(self):
        """Compute the tail windows with AudDataset's end-clamping.

        Offline extraction produces int((m-16)/80*25)+2 windows for m mel
        frames, the last few re-using the final 16 mels (crop_audio_window
        slides the window back at the end). Reproducing that here makes the
        flush tail bit-consistent with offline inference too.
        """
        n16 = int(len(self.pcm24) * 2) // 3
        if n16 < self.MEL_HOP:
            return
        m_total = n16 // self.MEL_HOP + 1        # librosa centred frame count
        data_len = int((m_total - 16) / 80.0 * 25.0) + 2
        if data_len <= self.n_raw:
            return

        r0 = self.n_raw
        s0 = max(0, min(int(80 * r0 / 25), m_total - self.WIN) - self.CTX_MEL)
        wav16 = resample_poly(self.pcm24.astype(np.float32) / 32768.0, 2, 3)
        mel = melspectrogram(wav16[s0 * self.MEL_HOP:]).T

        wins = []
        for j in range(r0, data_len):
            a = min(int(80 * j / 25), m_total - self.WIN) - s0
            wins.append(mel[a:a + self.WIN].T)
        batch = torch.from_numpy(np.stack(wins).astype(np.float32)).unsqueeze(1)
        with torch.no_grad():
            out = audio_encoder(batch.to(device)).cpu().numpy()
        self.raw = out if self.raw is None else np.concatenate([self.raw, out])

    def features_for_emit(self, final: bool) -> Optional[np.ndarray]:
        """Feature array in the trained [first | raw | last] convention.

        Mid-stream the tail is extended by repeating the newest frame, so the
        outer slots of the last windows hold the latest real audio instead of
        the zeros get_audio_features would insert; on the final call the array
        matches offline extraction exactly.
        """
        if self.raw is None:
            return None
        if final:
            return np.concatenate([self.raw[:1], self.raw, self.raw[-1:]])
        tail = np.repeat(self.raw[-1:], 8, axis=0)
        return np.concatenate([self.raw[:1], self.raw, tail])


def _idle_walk_indices(refs: List[int], n: int) -> List[int]:
    """The source index sequence the idle cache walks - cache pos -> source.

    Must match the generation loop in initialize_idle_cache exactly; the client
    uses this mapping to hand the walk position across idle/speech transitions.
    """
    out, pos, step = [], 0, 1
    for _ in range(n):
        if pos >= len(refs) - 1:
            step = -1
        if pos <= 0:
            step = 1
        pos += step
        out.append(refs[max(0, min(pos, len(refs) - 1))])
    return out


def _reshape_audio_feat(a, mode_str: str) -> torch.Tensor:
    """
    Conv2D expects [B, C, H, W]. Keep SyncTalk modes:
      ave   -> (32,16,16) = 8192
      hubert-> (32,32,32)
      wenet -> (256,16,32)
    """
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    elif not torch.is_tensor(a):
        a = torch.tensor(a)

    a = a.contiguous().view(-1)

    if mode_str == "hubert":
        C, H, W = 32, 32, 32
    elif mode_str == "wenet":
        C, H, W = 256, 16, 32
    else:
        C, H, W = 32, 16, 16

    need = C * H * W
    if a.numel() < need:
        a = torch.cat([a, torch.zeros((need - a.numel(),), dtype=a.dtype)], dim=0)
    elif a.numel() > need:
        a = a[:need]

    return a.view(1, C, H, W)


def _generate_frame(sess: SessionState, audio_feats: np.ndarray, frame_idx: int) -> np.ndarray:
    """Generate one lip-synced frame.

    frame_idx selects the audio feature window; the base frame comes from the
    walk. Calls must be strictly sequential - the walk mutates shared state.
    """
    ref_index = _pingpong_next(sess)

    src, lms = _read_source_frame(ref_index)
    img = src.copy()   # the composite below writes into it; never mutate the cache

    xmin = int(lms[1][0])
    ymin = int(lms[52][1])
    xmax = int(lms[31][0])
    width = xmax - xmin
    ymax = ymin + width

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img.shape[1] - 1, xmax)
    ymax = min(img.shape[0] - 1, ymax)

    crop = img[ymin:ymax, xmin:xmax]
    if crop.size == 0:
        return cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)

    h, w = crop.shape[:2]
    crop_img = cv2.resize(crop, (328, 328), interpolation=cv2.INTER_CUBIC)
    crop_img_ori = crop_img.copy()

    # Extract face region (320x320 from 328x328)
    img_real_ex = crop_img[4:324, 4:324].copy()
    img_real_ex_ori = img_real_ex.copy()

    # CRITICAL: Create masked version - black out mouth region
    # The working version uses tuple (5, 5, 310, 305) which OpenCV interprets as (x, y, w, h)
    img_masked = apply_mouth_mask(img_real_ex_ori, MASK_VERSION)
    
    # Transpose to CHW format for model
    img_masked = img_masked.transpose(2, 0, 1).astype(np.float32)
    img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32)

    # Convert to tensors. Channels 0-2 are the APPEARANCE reference; training
    # always supplied a different frame there, so keeping it fixed both matches
    # training and stops the reference's own mouth from driving the output.
    if USE_FIXED_REFERENCE and fixed_ref_tensor is not None:
        img_real_ex_T = fixed_ref_tensor
    else:
        img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
    img_masked_T = torch.from_numpy(img_masked / 255.0)
    img_concat_T = torch.cat([img_real_ex_T, img_masked_T], dim=0)[None].to(device)

    # Get audio features
    a = _get_audio_features(audio_feats, frame_idx)
    a = _reshape_audio_feat(a, mode).to(device)

    # Generate prediction
    with torch.no_grad():
        pred = synctalk_model(img_concat_T, a)[0]

    pred = (pred.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # Composite back. img_real_ex_ori is the untouched crop, so the fade at the
    # bottom edge lands on real pixels rather than on the masked input.
    pred = blend_bottom_edge(pred, img_real_ex_ori, FEATHER_ROWS)
    crop_img_ori[4:324, 4:324] = pred
    crop_img_ori = cv2.resize(crop_img_ori, (w, h), interpolation=cv2.INTER_CUBIC)
    img[ymin:ymax, xmin:xmax] = crop_img_ori

    return cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)


def initialize_models(checkpoint_path: str, dataset_path: str, asr_mode: str):
    global audio_encoder, synctalk_model, device, mode
    global dataset_dir, img_dir, lms_dir, len_img, img_h, img_w
    global MASK_VERSION

    mode = asr_mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[SyncTalk] Using device: {device}")

    # Input shapes never change, so let cuDNN pick the fastest kernels once.
    torch.backends.cudnn.benchmark = True

    # Feeding a mask the model was not trained on leaves the jaw strip visible
    # as real pixels, and the generator's output there disagrees with the rest
    # of the crop - a rectangle outlining the mask appears on every frame.
    MASK_VERSION = read_mask_version(checkpoint_path)
    log(f"[SyncTalk] Mouth mask: {MASK_VERSION}")

    log("[SyncTalk] Loading audio encoder...")
    ae = AudioEncoder().to(device).eval()
    ckpt_path = os.path.join(".", "model", "checkpoints", "audio_visual_encoder.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    ae.load_state_dict({f"audio_encoder.{k}": v for k, v in ckpt.items()})

    log(f"[SyncTalk] Loading SyncTalk model from {checkpoint_path}...")
    m = Model(6, mode).to(device)
    m.load_state_dict(torch.load(checkpoint_path, map_location=device))
    m.eval()

    dataset_dir = dataset_path
    img_dir = os.path.join(dataset_dir, "full_body_img")
    lms_dir = os.path.join(dataset_dir, "landmarks")

    if not os.path.isdir(img_dir):
        raise RuntimeError(f"full_body_img not found: {img_dir}")
    if not os.path.isdir(lms_dir):
        raise RuntimeError(f"landmarks not found: {lms_dir}")

    # Count landmarks, not images. The closed-mouth scan in
    # initialize_idle_inputs walks range(len_img) reading .lms files, while
    # images are only ever read for the frames that scan selects. Deriving the
    # count from landmarks therefore lets a deployment ship all landmarks
    # (a few MB) with only the images it will actually use, instead of the
    # whole 1.3 GB of source frames.
    lms_files = [f for f in os.listdir(lms_dir) if f.endswith(".lms")]
    if len(lms_files) < 2:
        raise RuntimeError(f"Not enough .lms landmarks in {lms_dir}")
    len_img = len(lms_files)

    jpgs = sorted((f for f in os.listdir(img_dir) if f.endswith(".jpg")),
                  key=lambda f: int(os.path.splitext(f)[0]))
    if not jpgs:
        raise RuntimeError(f"No .jpg frames in {img_dir}")

    exm = cv2.imread(os.path.join(img_dir, jpgs[0]))
    if exm is None:
        raise RuntimeError(f"Could not read {jpgs[0]}")
    img_h, img_w = exm.shape[:2]

    # Output resolution: everything internal stays native; only the final
    # resize before JPEG encoding changes. Idle cache metadata keys on
    # img_w/img_h, so a resolution change rebuilds it automatically.
    if OUT_SIZE > 0 and max(img_h, img_w) > OUT_SIZE:
        scale = OUT_SIZE / max(img_h, img_w)
        img_w, img_h = int(round(img_w * scale)), int(round(img_h * scale))
        log(f"[SyncTalk] Output scaled to {img_w}x{img_h} (native {exm.shape[1]}x{exm.shape[0]})")

    # Until initialize_idle_inputs finds the closed-mouth run, the walk may use
    # any frame.
    global speech_window
    speech_window = list(range(len_img))

    audio_encoder = ae
    synctalk_model = m

    log(f"[SyncTalk] ✅ Ready. Image: {img_w}x{img_h} landmarks={len_img} "
          f"images={len(jpgs)} mode={mode}")

    initialize_idle_inputs()


def _mouth_aperture(lms: np.ndarray) -> float:
    """Vertical gap between the inner lips. ~0 when the mouth is shut."""
    return float(lms[INNER_LIP_LOWER, 1].mean() - lms[INNER_LIP_UPPER, 1].mean())


def initialize_idle_inputs():
    """Prepare the two things idle frames need: silence features and quiet frames.

    Previously idle fed an all-zero audio tensor, described as "silence". It is
    not. The AVE encoder is post-ReLU, so real features are non-negative with
    mean ~0.36, and true silence encodes to a rich vector of similar magnitude.
    An all-zero tensor sits ~2.5x further from the feature centroid than real
    silence does - it is out of distribution, and the UNet emits an arbitrary
    mouth for it. That is the "mouth moves during idle" artefact.

    The second cause is the reference frame. Channels 0-2 of the model input
    carry a real frame including its mouth, and idle used to ping-pong across
    the whole video - mostly frames where the speaker was mid-sentence. So the
    generated mouth tracked whatever the reference happened to be doing. We now
    only use frames whose mouth is actually closed.
    """
    global silence_feats, idle_ref_indices

    # --- 1. encode real silence -------------------------------------------
    try:
        import tempfile
        import soundfile as sf
        sil_path = os.path.join(tempfile.gettempdir(), "synctalk_silence.wav")
        if not os.path.exists(sil_path):
            sf.write(sil_path, np.zeros(16000 * 2, dtype=np.float32), 16000)
        silence_feats = _process_audio_to_features(sil_path)
        log(f"[Idle] Encoded true silence: {silence_feats.shape} "
              f"mean={silence_feats.mean():+.4f} (zeros would be 0.0000)")
    except Exception as e:
        silence_feats = None
        log(f"[Idle] WARNING: could not encode silence ({e}); falling back to zeros. "
              "Idle mouth may look wrong.")

    # --- 2. find a CONTIGUOUS run of closed-mouth frames -------------------
    # Contiguity matters as much as closure. Picking the globally-quietest
    # frames gives a set scattered across the whole video, so consecutive idle
    # frames jump between unrelated head poses and the avatar looks like it is
    # skipping. A single continuous stretch keeps natural head motion.
    ap = np.full(len_img, np.nan, dtype=np.float32)
    for i in range(len_img):
        try:
            ap[i] = _mouth_aperture(_load_landmarks(os.path.join(lms_dir, f"{i}.lms")))
        except Exception:
            continue

    valid = ~np.isnan(ap)
    if not valid.any():
        idle_ref_indices = list(range(min(len_img, 100)))
        log("[Idle] WARNING: no landmarks readable; idle will use the first frames.")
        return

    def longest_run(mask):
        best_start, best_len, start = 0, 0, None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start > best_len:
                    best_start, best_len = start, i - start
                start = None
        if start is not None and len(mask) - start > best_len:
            best_start, best_len = start, len(mask) - start
        return best_start, best_len

    # Ping-pong over L frames yields 2L-2 distinct frames before repeating, so
    # L = N/2 + 2 is the minimum. But when idle plays REAL frames we ask for the
    # full N, which means playback never has to reverse - and time-reversed
    # motion (a blink running backwards) is one of the things that reads as
    # unnatural. Real frames cost nothing to "generate", so the longer run is
    # free; it comes at slightly higher aperture, which does not matter when the
    # mouth is genuine footage rather than something the model invented.
    n_idle = int(IDLE_DURATION_SECONDS * IDLE_FPS)
    needed = n_idle if IDLE_USE_RAW_FRAMES else n_idle // 2 + 2
    v = ap[valid]
    log(f"[Idle] Mouth aperture across {int(valid.sum())} frames: "
          f"min={v.min():.1f} median={float(np.median(v)):.1f} max={v.max():.1f}")

    chosen = None
    for q in (10, 15, 20, 25, 30, 35, 45, 60, 100):
        thr = float(np.percentile(v, q))
        start, length = longest_run(valid & (ap <= thr))
        if length >= needed:
            chosen = (start, length, thr, q)
            break

    if chosen is None:
        # Nothing long enough anywhere; take the quietest stretch we can get.
        thr = float(np.percentile(v, 60))
        start, length = longest_run(valid & (ap <= thr))
        chosen = (start, max(length, 1), thr, 60)
        log("[Idle] WARNING: no long closed-mouth run found; using the best available.")

    start, length, thr, q = chosen
    idle_ref_indices = list(range(start, start + length))
    seg = ap[start:start + length]
    log(f"[Idle] Idle reference run: frames {start}..{start + length - 1} "
          f"({length} frames, {length / IDLE_FPS:.1f}s) at <={thr:.1f} ({q}th pct)")
    log(f"[Idle] Aperture in run: mean={np.nanmean(seg):.2f} max={np.nanmax(seg):.2f} "
          f"(video median {float(np.median(v)):.1f})")

    # Speech base frames come from the same run: idle plays this footage, so
    # starting speech inside it means the transition cannot jump to an
    # unrelated pose, and the small working set stays hot in the page cache.
    global speech_window
    if length >= 25:
        speech_window = list(idle_ref_indices)
        log(f"[Walk] Speech base frames constrained to the idle run "
              f"({len(speech_window)} frames)")

    # --- 3. build the fixed reference frame for channels 0-2 ---------------
    if USE_FIXED_REFERENCE:
        try:
            quietest = int(np.nanargmin(np.where(valid, ap, np.nan)))
            _set_fixed_reference(quietest)
            log(f"[Ref] Fixed reference frame = {quietest} (aperture {ap[quietest]:.2f}). "
                  "Channels 0-2 no longer track the current frame.")
        except Exception as e:
            log(f"[Ref] WARNING: could not build fixed reference ({e}); "
                  "falling back to per-frame reference (mouth may move on silence).")


def _set_fixed_reference(frame_idx: int):
    """Cache one closed-mouth frame as the appearance reference (channels 0-2)."""
    global fixed_ref_tensor, fixed_ref_index

    img = cv2.imread(os.path.join(img_dir, f"{frame_idx}.jpg"))
    if img is None:
        raise RuntimeError(f"could not read reference frame {frame_idx}")
    lms = _load_landmarks(os.path.join(lms_dir, f"{frame_idx}.lms"))

    xmin, ymin = max(0, int(lms[1][0])), max(0, int(lms[52][1]))
    xmax = min(img.shape[1] - 1, int(lms[31][0]))
    ymax = min(img.shape[0] - 1, ymin + (xmax - xmin))

    crop = img[ymin:ymax, xmin:xmax]
    if crop.size == 0:
        raise RuntimeError("empty crop for reference frame")
    crop = cv2.resize(crop, (328, 328), interpolation=cv2.INTER_CUBIC)
    ref = crop[4:324, 4:324].transpose(2, 0, 1).astype(np.float32) / 255.0
    fixed_ref_tensor = torch.from_numpy(ref)


def _generate_idle_frame(img_idx: int, mode_str: str) -> np.ndarray:
    """Generate a single idle frame with neutral/closed mouth (silence audio features)"""
    global img_dir, lms_dir, img_h, img_w, len_img, synctalk_model, device
    
    # Wrap index for ping-pong
    ref_index = img_idx % max(1, len_img)
    
    img_path = os.path.join(img_dir, f"{ref_index}.jpg")
    lms_path = os.path.join(lms_dir, f"{ref_index}.lms")
    
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    
    lms = _load_landmarks(lms_path)
    
    xmin = int(lms[1][0])
    ymin = int(lms[52][1])
    xmax = int(lms[31][0])
    width = xmax - xmin
    ymax = ymin + width
    
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img.shape[1] - 1, xmax)
    ymax = min(img.shape[0] - 1, ymax)
    
    crop = img[ymin:ymax, xmin:xmax]
    if crop.size == 0:
        return cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)
    
    h, w = crop.shape[:2]
    crop_img = cv2.resize(crop, (328, 328), interpolation=cv2.INTER_CUBIC)
    crop_img_ori = crop_img.copy()
    
    if IDLE_USE_RAW_FRAMES:
        # crop_img_ori still holds the untouched crop, so the same resize path
        # below simply returns the real frame with its real mouth.
        crop_img_ori = cv2.resize(crop_img_ori, (w, h), interpolation=cv2.INTER_CUBIC)
        img[ymin:ymax, xmin:xmax] = crop_img_ori
        return cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)

    img_real_ex = crop_img[4:324, 4:324].copy()
    img_real_ex_ori = img_real_ex.copy()

    img_masked = apply_mouth_mask(img_real_ex_ori, MASK_VERSION)
    
    img_masked = img_masked.transpose(2, 0, 1).astype(np.float32)
    img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32)
    
    img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
    img_masked_T = torch.from_numpy(img_masked / 255.0)
    img_concat_T = torch.cat([img_real_ex_T, img_masked_T], dim=0)[None].to(device)
    
    # Feed features of REAL silence, not zeros. Zeros are out of distribution
    # for this post-ReLU encoder and make the UNet emit an arbitrary mouth.
    if silence_feats is not None and len(silence_feats) > 20:
        mid = len(silence_feats) // 2
        window = _get_audio_features(silence_feats, mid)          # [16, D]
        silent_audio = _reshape_audio_feat(window, mode_str).to(device)
    else:
        if mode_str == "hubert":
            C, H, W = 32, 32, 32
        elif mode_str == "wenet":
            C, H, W = 256, 16, 32
        elif mode_str == "ssl":
            C, H, W = 16, 32, 32
        else:  # ave
            C, H, W = 32, 16, 16
        silent_audio = torch.zeros((1, C, H, W), dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = synctalk_model(img_concat_T, silent_audio)[0]
    
    pred = (pred.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    
    crop_img_ori[4:324, 4:324] = pred
    crop_img_ori = cv2.resize(crop_img_ori, (w, h), interpolation=cv2.INTER_CUBIC)
    img[ymin:ymax, xmin:xmax] = crop_img_ori
    
    return cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)


def initialize_idle_cache():
    """Generate or load cached idle animation frames"""
    global idle_cache, img_w, img_h, mode, idle_source_map

    cache_dir = os.path.join(os.path.dirname(__file__), IDLE_CACHE_DIR)
    cache_file = os.path.join(cache_dir, "idle_frames.pkl")
    meta_file = os.path.join(cache_dir, "metadata.json")

    num_frames = int(IDLE_DURATION_SECONDS * IDLE_FPS)  # 100 frames for 4 seconds

    # cache position -> source frame; deterministic, so valid for a loaded
    # cache too (the generation loop below walks the identical sequence).
    refs_for_map = idle_ref_indices if idle_ref_indices else list(range(len_img))
    idle_source_map = _idle_walk_indices(refs_for_map, num_frames)
    
    # Check if cache exists and is valid
    if os.path.exists(cache_file) and os.path.exists(meta_file):
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
            
            if (meta.get("frame_count") == num_frames and
                meta.get("img_w") == img_w and
                meta.get("img_h") == img_h and
                meta.get("mode") == mode and
                meta.get("version") == IDLE_CACHE_VERSION):
                
                log(f"[IdleCache] Loading {num_frames} cached idle frames...")
                with open(cache_file, "rb") as f:
                    frames = pickle.load(f)
                
                idle_cache = IdleCache(
                    frames=frames,
                    frame_count=len(frames),
                    img_w=img_w,
                    img_h=img_h
                )
                log(f"[IdleCache] ✅ Loaded {len(frames)} idle frames from cache")
                return
        except Exception as e:
            log(f"[IdleCache] Cache invalid or corrupted: {e}")
    
    # Generate new idle frames
    log(f"[IdleCache] Generating {num_frames} idle frames ({IDLE_DURATION_SECONDS}s at {IDLE_FPS}fps)...")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    frames = []

    # Ping-pong through CLOSED-MOUTH frames only. Walking the whole video meant
    # most reference frames caught the speaker mid-sentence, and the generated
    # mouth partly copied whatever the reference was doing. The sequence comes
    # from _idle_walk_indices so idle_source_map stays exact by construction.
    for i, img_idx in enumerate(idle_source_map):
        try:
            frame = _generate_idle_frame(img_idx, mode)
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                frames.append(jpg.tobytes())
        except Exception as e:
            # Fail on the first frame rather than logging the same error 100
            # times and then caching an empty animation.
            if not frames:
                raise RuntimeError(
                    f"Idle frame generation failed on the first frame: {e}"
                ) from e
            log(f"[IdleCache] Error generating frame {i}: {e}")

        if (i + 1) % 25 == 0:
            log(f"[IdleCache] Generated {i + 1}/{num_frames} frames...")
    
    # Save to cache
    with open(cache_file, "wb") as f:
        pickle.dump(frames, f)
    
    with open(meta_file, "w") as f:
        json.dump({
            "frame_count": len(frames),
            "img_w": img_w,
            "img_h": img_h,
            "mode": mode,
            "duration_seconds": IDLE_DURATION_SECONDS,
            "fps": IDLE_FPS,
            "version": IDLE_CACHE_VERSION,
            "closed_mouth_refs": len(idle_ref_indices),
            "silence_features": silence_feats is not None,
        }, f)
    
    idle_cache = IdleCache(
        frames=frames,
        frame_count=len(frames),
        img_w=img_w,
        img_h=img_h
    )
    
    log(f"[IdleCache] ✅ Generated and cached {len(frames)} idle frames")


# -----------------------------
# Background worker per session
# -----------------------------
async def session_worker(sess: SessionState, fps: int = 25):
    """Stream lip-synced frames for incoming audio, audio-first.

    Protocol per emitted batch (client is the audio-master; see agent):
        1. AUDIO packet:  [seg_id][END_MARKER][n_frames][dur_ms] + int16 PCM.
           Sent BEFORE any frame so the client can keep speech continuous
           regardless of how fast the GPU renders.
        2. n_frames FRAME packets, streamed as each is generated:
           [seg_id][frame_idx][n_frames][dur_ms] + jpg.
        3. On flush (end of utterance): a text JSON
           {"type": "utterance_end", "frames": N, "end_source_idx": i}
           so the client knows the total and where to resume its idle loop.

    Design notes:
      - Generation is strictly sequential: the base-frame walk mutates shared
        state, and the GPU serialises forwards anyway. The old gather() version
        raced the walk and shuffled head poses.
      - Features come from StreamingFeatureExtractor, which reproduces the
        offline extraction (and its training-time edge-duplication convention)
        incrementally. A frame is emitted once the centre of its feature
        window is real audio; the tail of the window uses the newest frame.
      - If rendering falls behind real time by more than CATCHUP_BEHIND frames,
        the previous JPEG is re-sent instead of rendered. The client would drop
        a late frame anyway; skipping lets the GPU catch up.
    """
    SR = 24000                     # PCM rate of the Gemini stream
    SPF = SR // fps                # samples per video frame (960)
    SEGMENT_S = 0.20               # audio accumulated before a feature batch;
                                   # frames reach the client in one burst per
                                   # batch, so shorter batches = smoother flow
    CATCHUP_BEHIND = 10            # frames behind real time before skipping
    CLIENT_GATE_S = 0.30           # client's prebuffer, used to anchor "behind"
    EMIT_CHUNK = 12                # frames rendered before yielding to the loop

    extractor = StreamingFeatureExtractor()
    sent = 0                                   # frames emitted this utterance
    batch = np.zeros(0, dtype=np.int16)        # PCM since last extraction
    batch_t0: Optional[float] = None
    utt_anchor: Optional[float] = None         # wall clock of client playback start
    last_jpg: Optional[bytes] = None

    def make_header(segment_id: int, frame_idx: int, total_frames: int, audio_dur_ms: int) -> bytes:
        return (
            segment_id.to_bytes(4, "little", signed=False) +
            frame_idx.to_bytes(4, "little", signed=False) +
            total_frames.to_bytes(4, "little", signed=False) +
            audio_dur_ms.to_bytes(4, "little", signed=False)
        )

    def render(feats: np.ndarray, fi: int) -> Optional[bytes]:
        """Generate frame fi of the utterance and JPEG-encode it. Thread-safe
        to run in a worker thread only because calls are strictly sequential."""
        frame = _generate_frame(sess, feats, fi)
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        return jpg.tobytes() if ok else None

    async def emit(n: int, feats: np.ndarray):
        """Send audio for frames [sent, sent+n), then stream those frames."""
        nonlocal sent, utt_anchor, last_jpg
        if n <= 0:
            return
        sess.segment_id += 1
        seg = sess.segment_id
        dur_ms = int(n * 1000 / fps)

        audio = extractor.pcm24[sent * SPF: (sent + n) * SPF]
        if len(audio) < n * SPF:   # only possible on the flush tail
            audio = np.concatenate(
                [audio, np.zeros(n * SPF - len(audio), dtype=np.int16)])
        await sess.frame_q.put(
            make_header(seg, END_MARKER, n, dur_ms) + audio.tobytes())

        if utt_anchor is None:
            utt_anchor = time.monotonic() + CLIENT_GATE_S

        for i in range(n):
            fi = sent + i
            behind = (time.monotonic() - utt_anchor) * fps - fi
            if behind > CATCHUP_BEHIND and last_jpg is not None:
                jpg = last_jpg                      # skip render, catch up
            else:
                jpg = await asyncio.to_thread(render, feats, fi)
                if jpg is None:
                    jpg = last_jpg
                if jpg is None:
                    continue                        # nothing renderable yet
            last_jpg = jpg
            await sess.frame_q.put(make_header(seg, i, n, dur_ms) + jpg)
            sess.frames_generated += 1
        sent += n

    def reset_utterance():
        nonlocal extractor, sent, utt_anchor
        extractor = StreamingFeatureExtractor()
        sent = 0
        utt_anchor = None

    log(f"[Session {sess.sid}] Worker started (streaming, segment={SEGMENT_S}s, "
          f"window={len(speech_window)}f)")

    try:
        while not sess.closed.is_set():
            if sess.reset_requested:
                sess.reset_requested = False
                reset_utterance()
                batch = np.zeros(0, dtype=np.int16)
                batch_t0 = None
                for q in (sess.audio_q, sess.frame_q):
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                # Queued after the flush, so it is the first thing the client
                # receives on the clean stream: everything before it belongs to
                # the interrupted turn and can be discarded without guessing.
                await sess.frame_q.put(json.dumps({"type": "reset_done"}))
                log(f"[Session {sess.sid}] Reset: queues cleared")

            # ---- 1. drain incoming audio --------------------------------
            flush_now = False
            stop = False
            try:
                items = [await asyncio.wait_for(sess.audio_q.get(), timeout=0.05)]
                while not sess.audio_q.empty():
                    try:
                        items.append(sess.audio_q.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                for it in items:
                    if it is None:
                        stop = True
                        break
                    if it == b"__FLUSH__":
                        flush_now = True
                        continue
                    try:
                        with io.BytesIO(it) as f:
                            data, sr_in = sf.read(f, dtype="int16")
                        if data.ndim > 1:
                            data = data[:, 0]
                        if sr_in != SR:
                            n_out = int(len(data) * SR / sr_in)
                            data = np.interp(
                                np.linspace(0, len(data), n_out, endpoint=False),
                                np.arange(len(data)), data).astype(np.int16)
                        batch = np.concatenate([batch, data])
                        if batch_t0 is None:
                            batch_t0 = time.monotonic()
                        sess.last_audio_time = time.time()
                    except Exception as e:
                        log(f"[Session {sess.sid}] Bad WAV chunk: {e}")
                if stop:
                    break
            except asyncio.TimeoutError:
                pass

            # ---- 2. fold the batch into the utterance -------------------
            if batch.size and (flush_now or
                               (batch_t0 and time.monotonic() - batch_t0 >= SEGMENT_S)):
                extractor.add(batch)
                batch = np.zeros(0, dtype=np.int16)
                batch_t0 = None
                await asyncio.to_thread(extractor.extract)

            # ---- 3. emit what has enough context ------------------------
            # Gemini streams far faster than real time, so a whole answer can
            # be waiting at once. Emitting it in one call rendered hundreds of
            # frames without returning here, the audio queue climbed past 200
            # chunks, and neither a flush nor a reset could be seen until the
            # batch finished. EMIT_CHUNK bounds one pass; the loop comes
            # straight back for the rest.
            if flush_now:
                await asyncio.to_thread(extractor.finalize)
                if extractor.n_raw > 0:
                    feats = extractor.features_for_emit(final=True)
                    target = max(sent, max(1, -(-len(extractor.pcm24) // SPF)))
                    while sent < target and not sess.reset_requested:
                        await emit(min(EMIT_CHUNK, target - sent), feats)
                # utterance_end goes out for EVERY flush, including one that
                # carried too little audio to yield a feature frame. The client
                # turns it into the playback-completion report LiveKit waits
                # on; skipping it leaves that speech unacknowledged forever.
                end_idx = speech_window[sess.img_idx] if speech_window else 0
                await sess.frame_q.put(json.dumps({
                    "type": "utterance_end",
                    "frames": sent,
                    "end_source_idx": int(end_idx),
                }))
                log(f"[Session {sess.sid}] Utterance done: {sent} frames, "
                    f"{len(extractor.pcm24) / SR:.2f}s audio, walk at {end_idx}")
                reset_utterance()
            elif extractor.n_raw > 0:
                # Frame fi's feature window reaches raw[fi+6], so emitting at
                # n_raw-7 keeps every mid-stream window identical to offline
                # extraction (verified by the parity test). Costs 280 ms of
                # holdback; emitting earlier trades lip accuracy for latency.
                n_avail = min(extractor.n_audio_frames,
                              extractor.n_raw - 7) - sent
                if n_avail > 0:
                    await emit(min(n_avail, EMIT_CHUNK),
                               extractor.features_for_emit(final=False))

    except Exception as e:
        log(f"[Session {sess.sid}] Worker fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        log(f"[Session {sess.sid}] Worker stopped. Total frames: {sess.frames_generated}")


# -----------------------------
# HTTP
# -----------------------------
@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "models_loaded": (audio_encoder is not None and synctalk_model is not None),
        "device": str(device),
        "mode": mode,
        "image_size": f"{img_w}x{img_h}",
        "frames_available": len_img,
        "sessions": len(sessions),
        "idle_cache": {
            "ready": idle_cache is not None,
            "frame_count": idle_cache.frame_count if idle_cache else 0,
            "duration_seconds": IDLE_DURATION_SECONDS if idle_cache else 0
        },
        "active_sessions": [
            {
                "sid": s.sid,
                "frames_generated": s.frames_generated,
                "audio_queue_size": s.audio_q.qsize(),
                "frame_queue_size": s.frame_q.qsize(),
            }
            for s in sessions.values() if not s.closed.is_set()
        ]
    })


@app.get("/idle/info")
async def idle_info():
    """Get information about idle cache"""
    if idle_cache is None:
        return JSONResponse({"ready": False, "frame_count": 0})
    return JSONResponse({
        "ready": True,
        "frame_count": idle_cache.frame_count,
        "img_w": idle_cache.img_w,
        "img_h": idle_cache.img_h,
        "duration_seconds": IDLE_DURATION_SECONDS,
        "fps": IDLE_FPS,
        # cache position -> source frame index, for walk handoff at
        # idle/speech transitions
        "source_map": idle_source_map,
    })


@app.get("/idle/frame/{index}")
async def get_idle_frame(index: int):
    """Get a single idle frame by index (wraps around)"""
    if idle_cache is None:
        return JSONResponse({"error": "Idle cache not ready"}, status_code=503)
    
    frame_bytes = idle_cache.get_frame(index)
    return Response(content=frame_bytes, media_type="image/jpeg")


@app.post("/session")
async def create_session():
    sid = uuid.uuid4().hex[:12]
    sess = SessionState(
        sid=sid,
        audio_q=asyncio.Queue(maxsize=2000),  # Large for complete responses
        frame_q=asyncio.Queue(maxsize=8000),  # Increased buffer (User requested 8000)
        closed=asyncio.Event(),
    )
    sessions[sid] = sess
    asyncio.create_task(session_worker(sess))
    log(f"[Session {sid}] Created")
    return {"session_id": sid, "idle_cache_ready": idle_cache is not None}


# -----------------------------
# WebSockets
# -----------------------------
@app.websocket("/ws/audio/{sid}")
async def ws_audio(ws: WebSocket, sid: str):
    await ws.accept()
    sess = sessions.get(sid)
    if not sess:
        log(f"[WS Audio] Session {sid} not found")
        await ws.close(code=1008)
        return

    sess.audio_connected = True
    log(f"[Session {sid}] Audio WebSocket connected")
    chunk_count = 0

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            # Text frames are control messages; binary frames are audio.
            text = msg.get("text")
            if text is not None:
                try:
                    js = json.loads(text)
                except (ValueError, TypeError):
                    continue
                kind = js.get("type")
                if kind == "align":
                    # Client reports where its idle loop is so speech starts
                    # from the same footage.
                    _align_walk(sess, int(js.get("source_idx", 0)))
                    log(f"[Session {sid}] Walk aligned to source {js.get('source_idx')}")
                elif kind == "reset":
                    sess.reset_requested = True
                continue

            data = msg.get("bytes")
            if not data:
                continue
            chunk_count += 1

            # NEVER drop audio - queue all chunks (accept latency for sync)
            # FLUSH sentinel (b"__FLUSH__") is also passed through here
            await sess.audio_q.put(data)

            if chunk_count % 50 == 0:
                log(f"[Session {sid}] Received {chunk_count} audio chunks, queue size: {sess.audio_q.qsize()}")

    except WebSocketDisconnect:
        log(f"[Session {sid}] Audio WebSocket disconnected after {chunk_count} chunks")
    except Exception as e:
        log(f"[Session {sid}] Audio WebSocket error: {e}")
    finally:
        sess.audio_connected = False
        if sess and not sess.video_connected:
            log(f"[Session {sid}] All WS disconnected, stopping worker")
            sess.closed.set()


@app.websocket("/ws/video/{sid}")
async def ws_video(ws: WebSocket, sid: str):
    """
    Send segment packets to client.
    
    Packet format (already assembled by session_worker):
        Video frame: [16B header] + jpg_bytes
        End-of-segment: [16B header] + pcm_bytes (frame_index = 0xFFFFFFFF)
    """
    await ws.accept()
    sess = sessions.get(sid)
    if not sess:
        log(f"[WS Video] Session {sid} not found")
        await ws.close(code=1008)
        return

    sess.video_connected = True
    log(f"[Session {sid}] Video WebSocket connected (segment protocol)")
    packet_count = 0
    segment_count = 0
    start_time = time.time()
    last_log_time = start_time
    bytes_sent = 0
    END_MARKER = 0xFFFFFFFF
    
    try:
        while not sess.closed.is_set():
            try:
                # Get packet with small timeout to check closed status
                packet = await asyncio.wait_for(sess.frame_q.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            # Text packets are control messages (e.g. utterance_end)
            if isinstance(packet, str):
                await ws.send_text(packet)
                continue

            # Packet is already complete (header + data), send directly
            await ws.send_bytes(packet)
            
            packet_count += 1
            bytes_sent += len(packet)
            
            # Check if this is an end-of-segment marker
            if len(packet) >= 16:
                frame_idx = int.from_bytes(packet[4:8], "little", signed=False)
                if frame_idx == END_MARKER:
                    segment_count += 1
            
            # Log stats every 2 seconds
            now = time.time()
            if now - last_log_time >= 2.0:
                elapsed_total = now - start_time
                mbps = (bytes_sent * 8 / 1_000_000) / elapsed_total if elapsed_total > 0 else 0
                log(f"[Session {sid}] Sent {packet_count} packets ({segment_count} segments), {mbps:.2f} Mbps, queue: {sess.frame_q.qsize()}")
                last_log_time = now
                
    except WebSocketDisconnect:
        log(f"[Session {sid}] Video WebSocket disconnected after {packet_count} packets ({segment_count} segments)")
    except Exception as e:
        log(f"[Session {sid}] Video WebSocket error: {e}")
    finally:
        sess.video_connected = False
        if sess and not sess.audio_connected:
             log(f"[Session {sid}] All WS disconnected, stopping worker")
             sess.closed.set()
        log(f"[Session {sid}] Video WS closed. Total: {packet_count} packets, {segment_count} segments, {bytes_sent / 1_000_000:.2f} MB")


# -----------------------------
# Main
# -----------------------------
def main():
    global OUT_SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", type=str, default="ave", choices=["ave", "hubert", "wenet"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--out_size", type=int, default=OUT_SIZE,
                        help="Longest side of streamed frames; 0 = native. "
                             "Smaller is faster to encode, send, and decode.")
    args = parser.parse_args()
    OUT_SIZE = args.out_size

    initialize_models(args.checkpoint, args.dataset, args.mode)
    
    # Generate or load idle animation cache
    initialize_idle_cache()

    print(f"\n✨ WS Avatar Server running on http://{args.host}:{args.port}")
    print("  GET  /health")
    print("  GET  /idle/info         (idle cache status)")
    print("  GET  /idle/frame/{idx}  (get idle frame)")
    print("  POST /session -> {session_id}")
    print("  WS   /ws/audio/{session_id}  (send WAV bytes)")
    print("  WS   /ws/video/{session_id}  (recv: [pts_ms uint64] + jpg bytes)")


    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        ws="wsproto",
        ws_ping_interval=0,
        ws_ping_timeout=0,
    )


if __name__ == "__main__":
    main()