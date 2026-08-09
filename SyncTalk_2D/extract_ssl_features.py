"""Extract Bangla SSL audio features for the `ssl` audio mode.

Replaces the English mel-CNN (`utils.py:AudioEncoder`, the `ave` mode) with a
self-supervised speech encoder pretrained on multilingual/Indic audio.

Output mirrors the `aud_ave.npy` convention exactly so the rest of the pipeline
is unchanged:
  * one feature row per video frame at 25 fps,
  * a copy of the first row prepended and a copy of the last row appended
    (see `evaluation/scripts/eval_sync_328.py:41`),
  * float32.

Frame-count drift between audio features and video frames is the classic silent
bug in this pipeline, so when writing a dataset the output length is locked to
the existing `aud_ave.npy` for that clip. That guarantees a drop-in swap.

Typical use:
    python extract_ssl_features.py --wav dataset/redwan/aud.wav \
                                   --out dataset/redwan/aud_ssl.npy --fp16

`inference_328.py` imports `extract()` directly for `--asr ssl`.

The encoder runs once, offline. It is NOT part of the training loop, so it costs
no training VRAM.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# AI4Bharat's IndicWav2Vec models are gated on HuggingFace (manual approval).
# XLS-R-300m is the plan's designated fallback: open, multilingual (128 langs
# incl. Bengali), and 1024-dim like every other candidate, so the [16,32,32]
# reshape geometry is unaffected by which one you choose.
DEFAULT_MODEL = "facebook/wav2vec2-xls-r-300m"
GATED_HINT = ("ai4bharat/* models need access approval at huggingface.co, then "
              "`huggingface-cli login` or set HF_TOKEN. Open alternatives: "
              "facebook/wav2vec2-xls-r-300m, facebook/mms-300m, "
              "arijitx/wav2vec2-xls-r-300m-bengali")

VIDEO_FPS = 25
# wav2vec2-family convolutional stride is 320 samples @16 kHz -> 20 ms -> 50 Hz.
SSL_HZ = 50
SAMPLE_RATE = 16000

# Layer 24 (the last) of a contrastively-pretrained XLS-R is collapsed: measured
# per-dim std 0.028 and frame-to-frame variation 0.39x the ave baseline, i.e.
# almost no temporal signal for SyncNet to key on. Middle layers carry ~2x MORE
# variation than ave. Measured on redwan (adj-delta/magnitude vs ave):
#   layer  3: 2.05x   9: 1.98x   12: 1.96x   18: 1.95x   24: 0.39x
# 12 is the conventional choice for phonetic content in wav2vec2-family models.
DEFAULT_LAYER = 12

_CACHE = {}


def load_audio(path):
    """Load as 16 kHz mono float32."""
    import librosa
    wav, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return wav.astype(np.float32)


def _load_weights_manually(model_name, device):
    """Load a .bin checkpoint without transformers' torch>=2.6 gate.

    transformers >=4.56 refuses torch.load on .bin files unless torch>=2.6
    (CVE-2025-32434). None of the wav2vec2/XLS-R repos ship safetensors, and
    upgrading torch would break the pinned CUDA build, so load it ourselves
    with weights_only=True and then VERIFY the weights actually landed - a
    silently random encoder would look like "SSL is worse than ave".
    """
    from transformers import AutoConfig, AutoModel
    from huggingface_hub import hf_hub_download

    cfg = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_config(cfg)

    bin_path = hf_hub_download(model_name, "pytorch_model.bin")
    sd = torch.load(bin_path, map_location="cpu", weights_only=True)

    # Pretraining checkpoints nest the backbone under `wav2vec2.`; AutoModel
    # (Wav2Vec2Model) expects the bare keys.
    target_keys = set(model.state_dict().keys())
    if not (set(sd.keys()) & target_keys):
        stripped = {k.split(".", 1)[1]: v for k, v in sd.items() if "." in k}
        if set(stripped.keys()) & target_keys:
            sd = stripped

    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_total = len(target_keys)
    n_loaded = n_total - len(missing)
    frac = n_loaded / max(n_total, 1)
    print(f"weights    : loaded {n_loaded}/{n_total} tensors ({frac:.1%}) from {Path(bin_path).name}")
    if frac < 0.90:
        raise RuntimeError(
            f"Only {frac:.1%} of encoder weights loaded - the checkpoint does not match "
            f"the model class. Refusing to continue with a partly random encoder.\n"
            f"First few missing: {list(missing)[:5]}"
        )
    return model.to(device).eval()


def load_encoder(model_name, device):
    """Load and cache the SSL encoder + its feature extractor."""
    key = (model_name, str(device))
    if key in _CACHE:
        return _CACHE[key]
    from transformers import AutoModel, AutoFeatureExtractor
    try:
        fe = AutoFeatureExtractor.from_pretrained(model_name)
    except OSError as e:
        if "gated" in str(e).lower() or "403" in str(e):
            raise SystemExit(f"\n{model_name} is a gated repo.\n{GATED_HINT}\n")
        raise
    try:
        model = AutoModel.from_pretrained(model_name).to(device).eval()
    except ValueError as e:
        if "torch.load" in str(e) or "CVE" in str(e) or "2.6" in str(e):
            model = _load_weights_manually(model_name, device)
        else:
            raise
    _CACHE[key] = (model, fe)
    return model, fe


def _encode(wav, model, feature_extractor, device, layer, chunk_sec, overlap_sec, fp16):
    """Run the SSL encoder, chunked so long files do not exhaust memory.

    Chunks overlap and the overlap is trimmed at internal boundaries, so the
    result is close to a single forward pass without holding a whole 2-hour
    file in VRAM.
    """
    chunk = int(chunk_sec * SAMPLE_RATE)
    overlap = int(overlap_sec * SAMPLE_RATE)
    trim_frames = int((overlap_sec / 2) * SSL_HZ)
    step = chunk - overlap
    if step <= 0:
        raise ValueError("chunk_sec must be larger than overlap_sec")

    pieces = []
    starts = list(range(0, max(len(wav) - overlap, 1), step))
    for n, start in enumerate(starts):
        seg = wav[start:start + chunk]
        if len(seg) < SAMPLE_RATE // 10:      # <0.1 s tail, nothing useful
            continue
        inputs = feature_extractor(seg, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        values = inputs.input_values.to(device)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=(fp16 and device.type == "cuda")):
                out = model(values, output_hidden_states=True)
            h = (out.last_hidden_state if layer == -1
                 else out.hidden_states[layer]).float()[0]     # [T, D]

        # Drop the half-overlap on each internal edge.
        lo = 0 if n == 0 else trim_frames
        hi = h.shape[0] if start + chunk >= len(wav) else h.shape[0] - trim_frames
        if hi > lo:
            pieces.append(h[lo:hi].cpu())

    if not pieces:
        raise RuntimeError("Encoder produced no frames - is the wav empty?")
    return torch.cat(pieces, dim=0)                            # [T_ssl, D]


def to_video_rate(feats, target_len):
    """Resample the ~50 Hz feature sequence onto `target_len` video frames."""
    x = feats.t().unsqueeze(0)                                 # [1, D, T]
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return x.squeeze(0).t().contiguous()                       # [target_len, D]


def extract(wav_path, model_name=DEFAULT_MODEL, layer=DEFAULT_LAYER, device=None, fp16=False,
            target_core=None, chunk_sec=30.0, overlap_sec=2.0, verbose=True,
            standardize=True, stats=None, return_stats=False):
    """Return float32 [N, D] features at 25 fps, first/last rows duplicated.

    target_core: number of real video frames (before the 2 pad rows). Defaults
    to round(duration * 25), which is what you want for fresh inference audio.

    standardize: per-dimension zero-mean/unit-variance. Raw hidden-state scale
    varies ~100x across layers, so without this the reshape geometry and the
    downstream BatchNorm behave very differently depending on --layer.
    stats: (mean, std) to reuse instead of computing. Training saves its stats
    so inference applies the SAME normalisation - otherwise train and test see
    differently-scaled features.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                              else ("cpu" if device == "auto" else device))

    wav = load_audio(wav_path)
    duration = len(wav) / SAMPLE_RATE
    if target_core is None:
        target_core = int(round(duration * VIDEO_FPS))
    if target_core < 1:
        raise RuntimeError(f"Computed a target length of {target_core}; audio too short.")

    if verbose:
        print(f"audio      : {wav_path}  ({duration:.2f}s)")
        print(f"encoder    : {model_name} (layer {layer}) on {device}")

    model, fe = load_encoder(model_name, device)

    # output_hidden_states=True retains all 25 layers, so peak memory is far
    # higher than a plain forward. On a 4 GB card 30 s chunks do not fit; halve
    # until they do rather than failing outright.
    feats = None
    cs, ov = chunk_sec, overlap_sec
    while feats is None:
        try:
            feats = _encode(wav, model, fe, device, layer, cs, ov, fp16)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if cs > 5.0:
                cs, ov = cs / 2, max(ov / 2, 0.5)
                if verbose:
                    print(f"  OOM - retrying with {cs:.1f}s chunks (overlap {ov:.1f}s)")
                continue
            # A 300M encoder in fp32 plus 25 retained hidden states does not fit
            # in 4 GB next to the CUDA context. This is a one-time offline job,
            # so fall back to CPU rather than failing.
            if verbose:
                print("  OOM at minimum chunk size - falling back to CPU "
                      "(slower, but this only runs once per clip)")
            device = torch.device("cpu")
            model, fe = load_encoder(model_name, device)
            cs, ov = chunk_sec, overlap_sec
            fp16 = False
    if verbose:
        print(f"raw        : {tuple(feats.shape)}  (~{feats.shape[0]/max(duration,1e-9):.1f} Hz)")

    core = to_video_rate(feats, target_core).numpy().astype(np.float32)

    used_stats = None
    if standardize:
        if stats is not None:
            mean, std = stats
        else:
            mean = core.mean(0)
            std = core.std(0)
        std = np.where(std < 1e-6, 1.0, std)      # dead dims stay zero, no blow-up
        core = (core - mean) / std
        used_stats = (mean.astype(np.float32), std.astype(np.float32))
        if verbose:
            print(f"standardised: per-dim std {core.std(0).mean():.4f} "
                  f"(was {std.mean():.4f})")

    padded = np.concatenate([core[:1], core, core[-1:]], axis=0).astype(np.float32)
    return (padded, used_stats) if return_stats else padded


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", required=True, help="Input audio (any rate; resampled to 16 kHz mono).")
    ap.add_argument("--out", required=True, help="Output .npy, e.g. dataset/redwan/aud_ssl.npy")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"HuggingFace encoder id. Default: {DEFAULT_MODEL}")
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                    help=f"Hidden layer to take (default {DEFAULT_LAYER}). Do NOT use -1: the "
                         "final layer of a pretrained-only XLS-R is collapsed (~100x less "
                         "per-dim variance than middle layers) and SyncNet cannot learn from it.")
    ap.add_argument("--no-standardize", action="store_true",
                    help="Skip per-dim normalisation. Not recommended - raw scale varies "
                         "~100x by layer.")
    ap.add_argument("--match", default=None,
                    help="Lock output length to this .npy (default: sibling aud_ave.npy if "
                         "present). Guarantees drop-in compatibility.")
    ap.add_argument("--frames-dir", default=None,
                    help="full_body_img dir to sanity-check against (default: sibling).")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fp16", action="store_true", help="Half precision encoding. Faster, ~same features.")
    ap.add_argument("--chunk-sec", type=float, default=30.0)
    ap.add_argument("--overlap-sec", type=float, default=2.0)
    args = ap.parse_args()

    out_path = Path(args.out)

    # --- decide the target length -------------------------------------------
    match_path = Path(args.match) if args.match else out_path.parent / "aud_ave.npy"
    if match_path.exists():
        target_total = int(np.load(match_path).shape[0])
        target_core = target_total - 2          # the 2 pad rows are added by extract()
        print(f"length     : matching {match_path.name} -> {target_total} rows")
    else:
        target_core = None
        target_total = None
        print(f"length     : {match_path.name} not found, deriving from duration")

    arr, stats = extract(args.wav, model_name=args.model, layer=args.layer, device=args.device,
                         fp16=args.fp16, target_core=target_core,
                         chunk_sec=args.chunk_sec, overlap_sec=args.overlap_sec,
                         standardize=not args.no_standardize, return_stats=True)

    # --- verify --------------------------------------------------------------
    if target_total is not None and arr.shape[0] != target_total:
        raise RuntimeError(f"Length check failed: got {arr.shape[0]}, expected {target_total}.")

    frames_dir = Path(args.frames_dir) if args.frames_dir else out_path.parent / "full_body_img"
    if frames_dir.exists():
        n_img = len(list(frames_dir.glob("*.jpg")))
        drift = n_img - arr.shape[0]
        status = "ok" if abs(drift) <= 3 else "SUSPICIOUS"
        print(f"frames     : {n_img} images vs {arr.shape[0]} feature rows (drift {drift:+d}) [{status}]")
        if abs(drift) > 3:
            raise RuntimeError(
                f"Feature rows and image count differ by {drift}. Expected within +-3. "
                "Training with misaligned audio silently produces a broken model - fix this first."
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    if stats is not None:
        # Inference must apply the SAME normalisation the model trained on.
        stats_path = out_path.with_name(out_path.stem + "_stats.npz")
        np.savez(stats_path, mean=stats[0], std=stats[1], layer=args.layer, model=args.model)
        print(f"stats      : {stats_path.name}  (reused at inference time)")
    print(f"saved      : {out_path}  shape={arr.shape} dtype={arr.dtype}")
    print(f"per-frame dim = {arr.shape[1]}  ->  window of 16 frames = {16 * arr.shape[1]} values")
    if arr.shape[1] != 1024:
        print(f"NOTE: reshape geometry assumes dim 1024 ([16,32,32]). This model gives "
              f"{arr.shape[1]} - the reshape in datasetsss_328.py/unet_328.py must match.")


if __name__ == "__main__":
    main()
