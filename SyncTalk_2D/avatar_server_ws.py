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
from utils import (AudioEncoder, AudDataset, get_audio_features as _get_audio_features,
                   apply_mouth_mask, read_mask_version)


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
fixed_ref_tensor: Optional[torch.Tensor] = None   # [3,320,320], set at startup

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


sessions: Dict[str, SessionState] = {}


# -----------------------------
# SyncTalk helpers
# -----------------------------
def _reset_pingpong(sess: SessionState):
    sess.img_idx = 0
    sess.step_stride = 1


def _pingpong_next(sess: SessionState) -> int:
    global len_img
    if len_img <= 1:
        return 0

    if sess.img_idx >= (len_img - 1):
        sess.step_stride = -1
    if sess.img_idx <= 0:
        sess.step_stride = 1

    sess.img_idx += sess.step_stride
    sess.img_idx = max(0, min(sess.img_idx, len_img - 1))
    return sess.img_idx


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


def _generate_frame(sess: SessionState, audio_feats: np.ndarray, frame_idx: int, start_frame: int = 0) -> np.ndarray:
    """
    Generate a single frame using SyncTalk model.
    EXACT copy of working stable version's mask logic.
    """
    idx = _pingpong_next(sess)
    ref_index = (idx + start_frame) % max(1, len_img)

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

    # Composite back
    crop_img_ori[4:324, 4:324] = pred
    crop_img_ori = cv2.resize(crop_img_ori, (w, h), interpolation=cv2.INTER_CUBIC)
    img[ymin:ymax, xmin:xmax] = crop_img_ori

    return cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)


def initialize_models(checkpoint_path: str, dataset_path: str, asr_mode: str):
    global audio_encoder, synctalk_model, device, mode
    global dataset_dir, img_dir, lms_dir, len_img, img_h, img_w

    mode = asr_mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SyncTalk] Using device: {device}")

    print("[SyncTalk] Loading audio encoder...")
    ae = AudioEncoder().to(device).eval()
    ckpt_path = os.path.join(".", "model", "checkpoints", "audio_visual_encoder.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    ae.load_state_dict({f"audio_encoder.{k}": v for k, v in ckpt.items()})

    print(f"[SyncTalk] Loading SyncTalk model from {checkpoint_path}...")
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

    jpgs = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    if len(jpgs) < 2:
        raise RuntimeError(f"Not enough .jpg frames in {img_dir}")
    len_img = len(jpgs)

    exm = cv2.imread(os.path.join(img_dir, "0.jpg"))
    if exm is None:
        raise RuntimeError("Could not read 0.jpg")
    img_h, img_w = exm.shape[:2]

    audio_encoder = ae
    synctalk_model = m

    print(f"[SyncTalk] ✅ Ready. Image: {img_w}x{img_h} frames_available={len_img} mode={mode}")

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
        print(f"[Idle] Encoded true silence: {silence_feats.shape} "
              f"mean={silence_feats.mean():+.4f} (zeros would be 0.0000)")
    except Exception as e:
        silence_feats = None
        print(f"[Idle] WARNING: could not encode silence ({e}); falling back to zeros. "
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
        print("[Idle] WARNING: no landmarks readable; idle will use the first frames.")
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
    print(f"[Idle] Mouth aperture across {int(valid.sum())} frames: "
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
        print("[Idle] WARNING: no long closed-mouth run found; using the best available.")

    start, length, thr, q = chosen
    idle_ref_indices = list(range(start, start + length))

    # --- 3. build the fixed reference frame for channels 0-2 ---------------
    if USE_FIXED_REFERENCE:
        try:
            quietest = int(np.nanargmin(np.where(valid, ap, np.nan)))
            _set_fixed_reference(quietest)
            print(f"[Ref] Fixed reference frame = {quietest} (aperture {ap[quietest]:.2f}). "
                  "Channels 0-2 no longer track the current frame.")
        except Exception as e:
            print(f"[Ref] WARNING: could not build fixed reference ({e}); "
                  "falling back to per-frame reference (mouth may move on silence).")


def _set_fixed_reference(frame_idx: int):
    """Cache one closed-mouth frame as the appearance reference (channels 0-2)."""
    global fixed_ref_tensor

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
    seg = ap[start:start + length]
    print(f"[Idle] Idle reference run: frames {start}..{start + length - 1} "
          f"({length} frames, {length / IDLE_FPS:.1f}s) at <={thr:.1f} ({q}th pct)")
    print(f"[Idle] Aperture in run: mean={np.nanmean(seg):.2f} max={np.nanmax(seg):.2f} "
          f"(video median {float(np.median(v)):.1f})")


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
    global idle_cache, img_w, img_h, mode
    
    cache_dir = os.path.join(os.path.dirname(__file__), IDLE_CACHE_DIR)
    cache_file = os.path.join(cache_dir, "idle_frames.pkl")
    meta_file = os.path.join(cache_dir, "metadata.json")
    
    num_frames = int(IDLE_DURATION_SECONDS * IDLE_FPS)  # 100 frames for 4 seconds
    
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
                
                print(f"[IdleCache] Loading {num_frames} cached idle frames...")
                with open(cache_file, "rb") as f:
                    frames = pickle.load(f)
                
                idle_cache = IdleCache(
                    frames=frames,
                    frame_count=len(frames),
                    img_w=img_w,
                    img_h=img_h
                )
                print(f"[IdleCache] ✅ Loaded {len(frames)} idle frames from cache")
                return
        except Exception as e:
            print(f"[IdleCache] Cache invalid or corrupted: {e}")
    
    # Generate new idle frames
    print(f"[IdleCache] Generating {num_frames} idle frames ({IDLE_DURATION_SECONDS}s at {IDLE_FPS}fps)...")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    frames = []

    # Ping-pong through CLOSED-MOUTH frames only. Walking the whole video meant
    # most reference frames caught the speaker mid-sentence, and the generated
    # mouth partly copied whatever the reference was doing.
    refs = idle_ref_indices if idle_ref_indices else list(range(len_img))
    pos, step = 0, 1

    for i in range(num_frames):
        if pos >= len(refs) - 1:
            step = -1
        if pos <= 0:
            step = 1
        pos += step
        img_idx = refs[max(0, min(pos, len(refs) - 1))]

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
            print(f"[IdleCache] Error generating frame {i}: {e}")

        if (i + 1) % 25 == 0:
            print(f"[IdleCache] Generated {i + 1}/{num_frames} frames...")
    
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
    
    print(f"[IdleCache] ✅ Generated and cached {len(frames)} idle frames")


# -----------------------------
# Background worker per session
# -----------------------------
async def session_worker(sess: SessionState, fps: int = 25, start_frame: int = 0, batch_duration: float = 0.02):
    """
    Process audio chunks and generate video frames with SEGMENT-BASED protocol.
    
    IMPORTANT: Maintains audio context overlap between segments for accurate lip sync.
    get_audio_features() needs ±8 frames of context, so we keep overlap audio.
    
    Each segment includes:
    - Video frames with segment_id, frame_index, total_frames
    - End-of-segment marker with audio PCM for synchronized playback
    
    Frame header format (16 bytes):
        [4B segment_id][4B frame_index][4B total_frames][4B audio_duration_ms] + jpg
    
    End-of-segment marker:
        [4B segment_id][4B 0xFFFFFFFF][4B total_frames][4B audio_duration_ms] + pcm_bytes
    """
    print(f"[Session {sess.sid}] Worker started (segment-based, batch={batch_duration}s, with audio context overlap)")
    
    pcm_buffer = np.zeros((0,), dtype=np.int16)
    buffer_start_time = None
    SAMPLE_RATE = 24000
    MAX_SEGMENT_DUR_S = 0.7 
    END_MARKER = 0xFFFFFFFF
    
    # Audio context overlap for better lip sync
    # Reduced to minimal overlap for lower latency
    # Keep last segment's audio features to prepend to next segment
    CONTEXT_FRAMES = 2  # Minimal context for smooth transitions (80ms)
    prev_audio_feats: Optional[np.ndarray] = None  # Previous segment's audio features
    
    def make_header(segment_id: int, frame_idx: int, total_frames: int, audio_dur_ms: int) -> bytes:
        """Create 16-byte header for frame or end-of-segment marker"""
        return (
            segment_id.to_bytes(4, "little", signed=False) +
            frame_idx.to_bytes(4, "little", signed=False) +
            total_frames.to_bytes(4, "little", signed=False) +
            audio_dur_ms.to_bytes(4, "little", signed=False)
        )
    
    async def generate_frame_bytes(feats, frame_idx: int, start_frm: int, offset: int = 0) -> Optional[bytes]:
        """Generate a single frame and return jpg bytes.
        
        Args:
            feats: Audio features array
            frame_idx: Frame index within the segment (0-based for output)
            start_frm: Starting frame offset for ping-pong animation
            offset: Offset to add to frame_idx for audio feature lookup (for context)
        """
        try:
            # Use frame_idx + offset for audio features lookup
            frame = await asyncio.to_thread(_generate_frame, sess, feats, frame_idx + offset, start_frm)
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                return None
            return jpg.tobytes()
        except Exception as e:
            print(f"[Session {sess.sid}] Error generating frame {frame_idx}: {e}")
            return None
    
    try:
        while not sess.closed.is_set():
            try:
                # 1. Collect all available audio chunks to batch them
                chunks = []
                # Wait for at least one chunk
                first = await asyncio.wait_for(sess.audio_q.get(), timeout=0.05)
                chunks.append(first)
                # Drain the rest immediately
                while not sess.audio_q.empty():
                    try:
                        chunks.append(sess.audio_q.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                
                # 2. Add all to PCM buffer
                should_flush_now = False
                for wav_bytes in chunks:
                    if wav_bytes is None:
                        chunks = [None] # Signal break
                        break
                    
                    if wav_bytes == b"__FLUSH__":
                        should_flush_now = True
                        continue
                    
                    try:
                        with io.BytesIO(wav_bytes) as wav_io:
                            data, sr = sf.read(wav_io, dtype='int16')
                            if data.ndim > 1: data = data[:, 0]
                            pcm_buffer = np.concatenate([pcm_buffer, data])
                            if buffer_start_time is None: buffer_start_time = time.time()
                    except Exception as e:
                        print(f"[Session {sess.sid}] Failed to read WAV chunk: {e}")
                
                if chunks and chunks[0] is None:
                    break

            except asyncio.TimeoutError:
                # Process buffered audio if enough time has passed
                if pcm_buffer.size > 0 and buffer_start_time and (time.time() - buffer_start_time >= batch_duration):
                    pass  # Will process below
                else:
                    continue
            
            # Check if we should process the buffer
            should_process = False
            if pcm_buffer.size > 0:
                buffer_age = time.time() - buffer_start_time if buffer_start_time else 0
                # Process if: buffer is old enough OR queue was backed up OR flush requested
                if buffer_age >= batch_duration or sess.audio_q.qsize() > 10 or should_flush_now:
                    should_process = True
            
            if not should_process:
                continue
            
            # Check duration - DO NOT DROP, just continue buffering if too short (unless flush)
            dur = pcm_buffer.size / float(SAMPLE_RATE)
            if dur < 0.04 and not should_flush_now:
                continue
            
            # Save combined PCM as WAV file for feature extraction
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()

            try:
                # Keep a copy of PCM for sending to client
                segment_pcm = pcm_buffer.copy()
                
                # Write combined PCM buffer as WAV
                sf.write(tmp_path, pcm_buffer, SAMPLE_RATE, format="WAV", subtype="PCM_16")
                
                audio_dur_ms = int(dur * 1000)

                # Increment segment ID
                sess.segment_id += 1
                segment_id = sess.segment_id
                
                sess.last_audio_time = time.time()

                # Extract audio features
                current_feats = await asyncio.to_thread(_process_audio_to_features, tmp_path)
                
                # Combine with previous segment's features for context
                # get_audio_features needs ±8 frames, so prepend last CONTEXT_FRAMES from prev
                # Only concatenate if dimensions match (same feature size per frame)
                if (prev_audio_feats is not None and 
                    len(prev_audio_feats) >= CONTEXT_FRAMES and
                    prev_audio_feats.shape[1:] == current_feats.shape[1:]):  # Check dimensions match
                    # Take last CONTEXT_FRAMES from previous segment
                    context = prev_audio_feats[-CONTEXT_FRAMES:]
                    feats = np.concatenate([context, current_feats], axis=0)
                    offset = CONTEXT_FRAMES  # Offset for frame generation
                    print(f"[Session {sess.sid}] [Segment {segment_id}] Using {CONTEXT_FRAMES} frames of audio context from prev segment")
                else:
                    feats = current_feats
                    offset = 0
                    if prev_audio_feats is not None and prev_audio_feats.shape[1:] != current_feats.shape[1:]:
                        print(f"[Session {sess.sid}] [Segment {segment_id}] Skipping context (shape mismatch: {prev_audio_feats.shape} vs {current_feats.shape})")
                
                # Save current features for next segment's context
                prev_audio_feats = current_feats
                
                nframes = max(1, int(round(dur * fps)))

                print(f"[Session {sess.sid}] [Segment {segment_id}] Generating {nframes} frames for {dur:.2f}s audio (feats: {len(feats)}, offset: {offset})")

                t0 = time.time()
                
                # Generate all frames for this segment with offset for audio lookup
                frame_bytes_list = []
                last_valid_frame = None
                if idle_cache and idle_cache.frame_count > 0:
                    last_valid_frame = idle_cache.get_frame(0)

                # FIX 1: Aggressive parallelization - generate ALL frames at once for small segments
                if nframes <= 25:  # 1 second or less - full parallel generation
                    tasks = [
                        generate_frame_bytes(feats, i, start_frame, offset)
                        for i in range(nframes)
                    ]
                    results = await asyncio.gather(*tasks)
                    
                    # Fix 4: Ensure integrity (replace failures)
                    for res in results:
                        if res is not None:
                            frame_bytes_list.append(res)
                            last_valid_frame = res
                        else:
                            if last_valid_frame:
                                frame_bytes_list.append(last_valid_frame)
                            else:
                                frame_bytes_list.append(b'')
                else:
                    # For longer segments, use larger batches
                    batch_size = 25  # Process 1 second at a time
                    for batch_start in range(0, nframes, batch_size):
                        batch_end = min(batch_start + batch_size, nframes)
                        tasks = [
                            generate_frame_bytes(feats, i, start_frame, offset)
                            for i in range(batch_start, batch_end)
                        ]
                        results = await asyncio.gather(*tasks)
                        
                        # Fix 4: Ensure integrity (replace failures)
                        for res in results:
                            if res is not None:
                                frame_bytes_list.append(res)
                                last_valid_frame = res
                            else:
                                if last_valid_frame:
                                    frame_bytes_list.append(last_valid_frame)
                                else:
                                    frame_bytes_list.append(b'')

                # Ensure non-empty frames check
                actual_nframes = len(frame_bytes_list)
                
                # Check for empty bytes from fallback failure
                if last_valid_frame:
                    frame_bytes_list = [fb if fb != b'' else last_valid_frame for fb in frame_bytes_list]
                
                # Queue all video frames with headers
                for frame_idx, jpg_bytes in enumerate(frame_bytes_list):
                    # NEVER DROP FRAMES - rely on large queue
                    header = make_header(segment_id, frame_idx, actual_nframes, audio_dur_ms)
                    packet = header + jpg_bytes
                    await sess.frame_q.put(packet)
                    sess.frames_generated += 1
                
                # Send end-of-segment marker with audio PCM
                END_MARKER = 0xFFFFFFFF
                end_header = make_header(segment_id, END_MARKER, actual_nframes, audio_dur_ms)
                end_packet = end_header + segment_pcm.tobytes()
                # NEVER DROP
                await sess.frame_q.put(end_packet)

                elapsed = time.time() - t0
                actual_fps = actual_nframes / elapsed if elapsed > 0 else 0
                print(f"[Session {sess.sid}] [Segment {segment_id}] ✅ Sent {actual_nframes} frames + audio ({elapsed:.2f}s, {actual_fps:.1f} gen-FPS)")
                
                # Reset buffer
                pcm_buffer = np.zeros((0,), dtype=np.int16)
                buffer_start_time = None

            except Exception as e:
                print(f"[Session {sess.sid}] worker error: {e}")
                import traceback
                traceback.print_exc()
                pcm_buffer = np.zeros((0,), dtype=np.int16)
                buffer_start_time = None
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                    
    except Exception as e:
        print(f"[Session {sess.sid}] Worker fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"[Session {sess.sid}] Worker stopped. Total frames: {sess.frames_generated}")


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
        "fps": IDLE_FPS
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
    print(f"[Session {sid}] Created")
    return {"session_id": sid, "idle_cache_ready": idle_cache is not None}


# -----------------------------
# WebSockets
# -----------------------------
@app.websocket("/ws/audio/{sid}")
async def ws_audio(ws: WebSocket, sid: str):
    await ws.accept()
    sess = sessions.get(sid)
    if not sess:
        print(f"[WS Audio] Session {sid} not found")
        await ws.close(code=1008)
        return

    sess.audio_connected = True
    print(f"[Session {sid}] Audio WebSocket connected")
    chunk_count = 0
    
    try:
        while True:
            data = await ws.receive_bytes()
            chunk_count += 1
            
            # NEVER drop audio - queue all chunks (accept latency for sync)
            # FLUSH sentinel (b"__FLUSH__") is also passed through here
            await sess.audio_q.put(data)
            
            if chunk_count % 10 == 0:
                print(f"[Session {sid}] Received {chunk_count} audio chunks, queue size: {sess.audio_q.qsize()}")
                
    except WebSocketDisconnect:
        print(f"[Session {sid}] Audio WebSocket disconnected after {chunk_count} chunks")
    except Exception as e:
        print(f"[Session {sid}] Audio WebSocket error: {e}")
    finally:
        sess.audio_connected = False
        if sess and not sess.video_connected:
            print(f"[Session {sid}] All WS disconnected, stopping worker")
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
        print(f"[WS Video] Session {sid} not found")
        await ws.close(code=1008)
        return

    sess.video_connected = True
    print(f"[Session {sid}] Video WebSocket connected (segment protocol)")
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
                print(f"[Session {sid}] Sent {packet_count} packets ({segment_count} segments), {mbps:.2f} Mbps, queue: {sess.frame_q.qsize()}")
                last_log_time = now
                
    except WebSocketDisconnect:
        print(f"[Session {sid}] Video WebSocket disconnected after {packet_count} packets ({segment_count} segments)")
    except Exception as e:
        # print(f"[Session {sid}] Video WebSocket error: {e}")
        # import traceback
        # traceback.print_exc()
        pass
    finally:
        sess.video_connected = False
        if sess and not sess.audio_connected:
             print(f"[Session {sid}] All WS disconnected, stopping worker")
             sess.closed.set()
        print(f"[Session {sid}] Video WS closed. Total: {packet_count} packets, {segment_count} segments, {bytes_sent / 1_000_000:.2f} MB")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", type=str, default="ave", choices=["ave", "hubert", "wenet"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

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