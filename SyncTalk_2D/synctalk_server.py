"""
synctalk_server.py — MJPEG STREAM (NO MOUTH DISTORTION) + SAFE AUDIO FEATURES

Fixes:
✅ Mouth distortion fixed: uses SAME OpenCV rectangle masking style as your old working code:
   cv2.rectangle(img, (5,5,310,305), ...)  # (x,y,w,h) overload, NOT (pt1,pt2)
✅ Safe short-audio handling (no empty torch.cat)
✅ Ensures audio feature tensor is [1,C,H,W] for Conv2D model
✅ Streams MJPEG frames with correct boundary + Content-Length
✅ /health returns image_size for your agent

Run:
python synctalk_server.py --checkpoint checkpoint/mydata/4.pth --dataset dataset/mydata --mode ave --host 0.0.0.0 --port 5001
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

import os
import cv2
import torch
import numpy as np
import tempfile
import soundfile as sf
from torch.utils.data import DataLoader

from unet_328 import Model
from utils import AudioEncoder, AudDataset, get_audio_features

app = Flask(__name__)
CORS(app)

audio_encoder = None
synctalk_model = None

dataset_dir = None
img_dir = None
lms_dir = None
len_img = None
img_h = None
img_w = None

mode = "ave"
device = None

# Ping-pong frame index like repo behavior
img_idx = 0
step_stride = 1


# ----------------------------
# Init
# ----------------------------
def initialize_models(checkpoint_path: str, dataset_path: str, asr_mode: str = "ave"):
    global audio_encoder, synctalk_model
    global dataset_dir, img_dir, lms_dir, len_img, img_h, img_w
    global mode, device

    mode = asr_mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SyncTalk] Using device: {device}")

    print("[SyncTalk] Loading audio encoder...")
    ae = AudioEncoder().to(device).eval()
    ckpt_path = os.path.join(".", "model", "checkpoints", "audio_visual_encoder.pth")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Repo uses keys that expect prefix audio_encoder.*
    ae.load_state_dict({f"audio_encoder.{k}": v for k, v in ckpt.items()})

    print(f"[SyncTalk] Loading SyncTalk model from {checkpoint_path}...")
    m = Model(6, mode).to(device)
    m.load_state_dict(torch.load(checkpoint_path, map_location=device))
    m.eval()

    dataset_dir = dataset_path
    img_dir = os.path.join(dataset_dir, "full_body_img")
    lms_dir = os.path.join(dataset_dir, "landmarks")

    if not os.path.isdir(img_dir):
        raise RuntimeError(f"full_body_img dir not found: {img_dir}")
    if not os.path.isdir(lms_dir):
        raise RuntimeError(f"landmarks dir not found: {lms_dir}")

    jpgs = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    if len(jpgs) < 2:
        raise RuntimeError(f"Not enough .jpg frames in {img_dir}")

    # IMPORTANT: your old code uses len_img = count - 1
    # to avoid going out of range in ping-pong logic.
    len_img_local = len(jpgs) - 1
    if len_img_local < 1:
        len_img_local = len(jpgs)

    exm = cv2.imread(os.path.join(img_dir, "0.jpg"))
    if exm is None:
        raise RuntimeError("Could not read 0.jpg")

    img_h_local, img_w_local = exm.shape[:2]

    audio_encoder = ae
    synctalk_model = m

    # Save globals
    globals()["len_img"] = len_img_local
    globals()["img_h"] = img_h_local
    globals()["img_w"] = img_w_local

    print(f"[SyncTalk] ✅ Ready. Image: {img_w}x{img_h}  frames_available={len_img}  mode={mode}")


def reset_frame_tracking():
    global img_idx, step_stride
    img_idx = 0
    step_stride = 1


def _ping_pong_next_index():
    """
    Match your old behavior:
    - bounce between frames
    - avoid negative
    """
    global img_idx, step_stride, len_img
    if img_idx > len_img - 1:
        step_stride = -1
    if img_idx < 1:
        step_stride = 1
    img_idx += step_stride
    return img_idx


# ----------------------------
# Audio -> Features (SAFE)
# ----------------------------
def process_audio_to_features(audio_path: str) -> np.ndarray:
    """
    Returns numpy array of features.
    Never crashes on short audio (no empty torch.cat).
    """
    info = sf.info(audio_path)
    dur = float(info.duration or 0.0)

    # Too short => return dummy (enough for ave 32*16*16 = 8192)
    if dur < 0.12:
        return np.zeros((3, 8192), dtype=np.float32)

    dataset = AudDataset(audio_path)
    try:
        if len(dataset) == 0:
            return np.zeros((3, 8192), dtype=np.float32)
    except Exception:
        pass

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
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
        outs.append(out.detach().cpu())

    if len(outs) == 0:
        return np.zeros((3, 8192), dtype=np.float32)

    outputs = torch.cat(outs, dim=0)
    first_frame, last_frame = outputs[:1], outputs[-1:]
    audio_feats = torch.cat([first_frame, outputs, last_frame], dim=0)
    return audio_feats.numpy()


def _reshape_audio_feat_to_4d(a, mode_str: str) -> torch.Tensor:
    """
    Conv2D expects [B,C,H,W].
    get_audio_features sometimes returns weird 2D shapes -> flatten -> reshape.
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
        pad = torch.zeros((need - a.numel(),), dtype=a.dtype)
        a = torch.cat([a, pad], dim=0)
    elif a.numel() > need:
        a = a[:need]

    return a.view(1, C, H, W)


# ----------------------------
# Frame Generation (FIXED MASK)
# ----------------------------
def generate_frame(audio_feats: np.ndarray, frame_idx: int, start_frame: int = 0) -> np.ndarray:
    global img_dir, lms_dir, mode, img_w, img_h

    idx = _ping_pong_next_index()
    # Match old code style: direct indexing with start_frame
    ref_index = idx + start_frame

    # Clamp to valid range
    if ref_index < 0:
        ref_index = 0
    if ref_index > len_img - 1:
        ref_index = len_img - 1

    img_path = os.path.join(img_dir, f"{ref_index}.jpg")
    lms_path = os.path.join(lms_dir, f"{ref_index}.lms")

    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    # Load landmarks
    lms_list = []
    with open(lms_path, "r") as f:
        for line in f.read().splitlines():
            arr = np.array(line.split(" "), dtype=np.float32)
            lms_list.append(arr)
    lms = np.array(lms_list, dtype=np.int32)

    # SAME crop logic as your old code
    xmin = int(lms[1][0])
    ymin = int(lms[52][1])
    xmax = int(lms[31][0])
    width = xmax - xmin
    ymax = ymin + width

    # Clamp bounds
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img.shape[1], xmax)
    ymax = min(img.shape[0], ymax)

    crop_img = img[ymin:ymax, xmin:xmax]
    if crop_img.size == 0:
        # fallback: return original
        return img

    h, w = crop_img.shape[:2]
    crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
    crop_img_ori = crop_img.copy()

    img_real_ex = crop_img[4:324, 4:324].copy()
    img_real_ex_ori = img_real_ex.copy()

    # ✅ CRITICAL: match your old code EXACTLY
    # Use (x,y,w,h) overload, not pt1/pt2
    img_masked = cv2.rectangle(img_real_ex_ori, (5, 5, 310, 305), (0, 0, 0), -1)

    img_masked = img_masked.transpose(2, 0, 1).astype(np.float32)
    img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32)

    img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
    img_masked_T = torch.from_numpy(img_masked / 255.0)
    img_concat_T = torch.cat([img_real_ex_T, img_masked_T], dim=0)[None].to(device)

    # Audio features
    a = get_audio_features(audio_feats, frame_idx)
    a = _reshape_audio_feat_to_4d(a, mode).to(device)

    with torch.no_grad():
        pred = synctalk_model(img_concat_T, a)[0]

    pred = (pred.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # Composite result
    crop_img_ori[4:324, 4:324] = pred
    crop_img_ori = cv2.resize(crop_img_ori, (w, h), interpolation=cv2.INTER_CUBIC)
    img[ymin:ymax, xmin:xmax] = crop_img_ori

    return img


# ----------------------------
# API
# ----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": synctalk_model is not None and audio_encoder is not None,
        "device": str(device),
        "mode": mode,
        "image_size": f"{img_w}x{img_h}" if img_w else None,
        "frames_available": len_img,
    })


@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "image_width": img_w,
        "image_height": img_h,
        "total_frames": len_img,
        "mode": mode,
        "device": str(device),
    })


@app.route("/stream_frames", methods=["POST"])
def stream_frames():
    if synctalk_model is None or audio_encoder is None:
        return jsonify({"error": "Models not loaded"}), 503

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    fps = int(request.form.get("fps", 25))
    fps = max(1, min(60, fps))

    start_frame = int(request.form.get("start_frame", 0))
    reset = request.form.get("reset", "1")

    jpeg_quality = int(request.form.get("jpeg_quality", 80))
    jpeg_quality = max(30, min(95, jpeg_quality))

    if reset == "1":
        reset_frame_tracking()

    audio_file = request.files["audio"]

    tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_audio_path = tmp_audio.name
    tmp_audio.close()

    audio_file.save(tmp_audio_path)

    try:
        info = sf.info(tmp_audio_path)
        duration = float(info.duration or 0.0)
        if duration < 0.12:
            return jsonify({"error": f"audio too short: {duration:.3f}s"}), 400

        num_frames = max(1, int(round(duration * fps)))

        # Compute features once per request
        audio_feats = process_audio_to_features(tmp_audio_path)

        def gen():
            try:
                for i in range(num_frames):
                    frame_bgr = generate_frame(audio_feats, i, start_frame=start_frame)
                    ok, jpg = cv2.imencode(
                        ".jpg",
                        frame_bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                    )
                    if not ok:
                        continue
                    data = jpg.tobytes()

                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(data)}\r\n".encode("utf-8")
                        + b"\r\n"
                        + data
                        + b"\r\n"
                    )
                yield b"--frame--\r\n"
            except Exception as e:
                print("[SyncTalk] Stream gen error:", e)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    except Exception as e:
        import traceback
        print("[SyncTalk] ❌ Error:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            os.remove(tmp_audio_path)
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", type=str, default="ave", choices=["ave", "hubert", "wenet"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    initialize_models(args.checkpoint, args.dataset, args.mode)

    print(f"\n✨ SyncTalk RAW FRAMES (MJPEG) running on http://{args.host}:{args.port}")
    print("  GET  /health")
    print("  GET  /info")
    print("  POST /stream_frames (audio=<wav>, fps=?, start_frame=?, reset=1/0, jpeg_quality=?)")
    app.run(host=args.host, port=args.port, threaded=True)
