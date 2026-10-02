"""Split a dataset with the held-out frames in the MIDDLE of the talking.

create_manifest.py holds out the end of the video. For a recording with idle
footage at both ends, that puts the test set on idle frames and keeps the
closing idle out of training. This finds the talking region from the audio,
takes the test split from its middle and validation right after it, and
trains on everything else - both idle sections included, since they teach
the model that silence means a closed mouth.

Splits are separated by --gap frames, wider than the model's +-8-frame audio
window, so no training sample's audio reaches into a held-out frame.

    python evaluation/scripts/create_holdout_manifest.py \\
        --dataset-dir dataset/alapon_v2 --name alapon_v2
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf

FPS = 25


def speech_frames(wav_path, n_frames):
    """Per-video-frame speech flag from audio energy."""
    pcm, sr = sf.read(wav_path, dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm.mean(1)
    hop = sr / FPS
    rms = np.array([np.sqrt(np.mean(pcm[int(i * hop):int((i + 1) * hop)] ** 2) + 1e-12)
                    for i in range(n_frames)])
    db = 20 * np.log10(rms)
    floor, peak = np.percentile(db, 10), np.percentile(db, 95)
    speech = db > floor + 0.35 * (peak - floor)
    # close short pauses between words (0.6 s) so the region is one stretch
    k = int(0.6 * FPS)
    padded = np.convolve(speech.astype(float), np.ones(k), mode="same") > 0
    return padded, db


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="ave")
    ap.add_argument("--test-seconds", type=float, default=30.88,
                    help="772 frames, the size of the previous test split")
    ap.add_argument("--val-seconds", type=float, default=30.84)
    ap.add_argument("--gap", type=int, default=25)
    ap.add_argument("--output", default=None)
    a = ap.parse_args()

    d = a.dataset_dir
    frames = len([f for f in os.listdir(os.path.join(d, "full_body_img")) if f.endswith(".jpg")])
    lms = len([f for f in os.listdir(os.path.join(d, "landmarks")) if f.endswith(".lms")])
    aud = np.load(os.path.join(d, f"aud_{a.mode}.npy"), mmap_mode="r").shape[0]
    last = min(frames, lms, aud - 1) - 1          # same bound as MyDataset

    speech, _ = speech_frames(os.path.join(d, "aud.wav"), last + 1)
    talk = np.flatnonzero(speech)
    if len(talk) == 0:
        raise SystemExit("no speech found in aud.wav")
    ts, te = int(talk[0]), int(talk[-1])

    n_test, n_val, g = int(round(a.test_seconds * FPS)), int(round(a.val_seconds * FPS)), a.gap
    need = n_test + g + n_val
    if te - ts + 1 < need + 2 * g:
        raise SystemExit(f"talking region {ts}..{te} too short for test+val ({need} frames)")
    test_start = (ts + te) // 2 - need // 2
    test = (test_start, test_start + n_test - 1)
    val = (test[1] + 1 + g, test[1] + g + n_val)
    train = [[0, test[0] - g - 1], [val[1] + g + 1, last]]

    out = {
        "dataset_name": a.name,
        "dataset_dir": os.path.abspath(d),
        "mode": a.mode,
        "fps": FPS,
        "created_by": "evaluation/scripts/create_holdout_manifest.py",
        "counts": {"frames": frames, "landmarks": lms, "audio_features": aud,
                   "usable_aligned_frames": last + 1},
        "splits": {
            # MyDataset/train_328 read "ranges"; start/end span the whole
            # thing and include held-out frames, so never use them for train.
            "train": {"ranges": train, "count": sum(e - s + 1 for s, e in train),
                      "start": train[0][0], "end": train[-1][1]},
            "val": {"start": val[0], "end": val[1], "count": n_val},
            "test": {"start": test[0], "end": test[1], "count": n_test},
        },
        "talking": {"start": ts, "end": te},
        "idle": {"leading": [0, ts - 1] if ts > 0 else None,
                 "trailing": [te + 1, last] if te < last else None},
    }
    path = a.output or os.path.join(os.path.dirname(__file__), "..", "manifests", f"{a.name}_splits.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    sec = lambda f: f"{f / FPS:.1f}s"
    print(f"usable frames 0..{last} ({sec(last + 1)})")
    print(f"idle: leading {out['idle']['leading']}  trailing {out['idle']['trailing']}")
    print(f"talking {ts}..{te}")
    print(f"train {train} ({out['splits']['train']['count']} frames)")
    print(f"test  {test[0]}..{test[1]} ({sec(n_test)})   val {val[0]}..{val[1]} ({sec(n_val)})")
    print(f"wrote {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
