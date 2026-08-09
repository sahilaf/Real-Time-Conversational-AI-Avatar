"""Shape check for the `ssl` audio mode. Run this BEFORE any ssl training.

A mismatch between the audio branch and the visual bottleneck does not fail
loudly - it fails at the torch.cat fusion, or worse, trains something subtly
wrong. Catching it here costs seconds; catching it later costs a full retrain.

    python tests/test_audio_dims.py        # from SyncTalk_2D/

Checks, for every mode:
  1. the audio branch produces the same spatial size as the reference `ave` mode
  2. Model(...) returns [B, 3, 320, 320]
  3. SyncNet embeddings for audio and face have matching width (cosine needs it)
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]      # SyncTalk_2D/
sys.path.insert(0, str(REPO))

from unet_328 import Model, AudioConvSSL, AudioConvAve, AudioConvHubert  # noqa: E402
from syncnet_328 import SyncNet_color                                    # noqa: E402

# mode -> the [C, H, W] the dataset reshape produces for a 16-frame window
SHAPES = {
    "ave":    (32, 16, 16),   # 16 frames x  512 dims =  8192
    "hubert": (32, 32, 32),   # 16 frames x 2048 dims = 32768
    "ssl":    (16, 32, 32),   # 16 frames x 1024 dims = 16384
}
BRANCHES = {"ave": AudioConvAve, "hubert": AudioConvHubert, "ssl": AudioConvSSL}

B = 2
failures = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  -> ' + detail) if detail else ''}")
    if not cond:
        failures.append(label)


print("=" * 68)
print("1. Audio branch output shapes (must all agree for fusion to work)")
print("=" * 68)
branch_out = {}
for mode, shape in SHAPES.items():
    c, h, w = shape
    assert 16 * (c * h * w // 16) == c * h * w
    per_frame = c * h * w // 16
    x = torch.randn(B, c, h, w)
    y = BRANCHES[mode]()(x)
    branch_out[mode] = tuple(y.shape[1:])
    print(f"  {mode:<7} in {tuple(x.shape)}  (per-frame dim {per_frame:>5})  ->  out {tuple(y.shape)}")

ref = branch_out["ave"]
check("ssl audio branch matches ave spatial size", branch_out["ssl"] == ref,
      f"ssl {branch_out['ssl']} vs ave {ref}")
check("hubert audio branch matches ave spatial size", branch_out["hubert"] == ref,
      f"hubert {branch_out['hubert']} vs ave {ref}")

print()
print("=" * 68)
print("2. Full generator forward (fusion + decoder)")
print("=" * 68)
img = torch.randn(B, 6, 320, 320)
for mode, shape in SHAPES.items():
    try:
        out = Model(6, mode=mode)(img, torch.randn(B, *shape))
        ok = tuple(out.shape) == (B, 3, 320, 320)
        check(f"Model(mode={mode!r}) -> [B,3,320,320]", ok, str(tuple(out.shape)))
    except Exception as e:
        check(f"Model(mode={mode!r}) forward", False, f"{type(e).__name__}: {e}")

print()
print("=" * 68)
print("3. SyncNet embedding widths (cosine similarity needs them equal)")
print("=" * 68)
face = torch.randn(B, 3, 320, 320)
for mode, shape in SHAPES.items():
    try:
        a, f = SyncNet_color(mode)(face, torch.randn(B, *shape))
        check(f"SyncNet(mode={mode!r}) audio/face widths match",
              a.shape == f.shape, f"audio {tuple(a.shape)} vs face {tuple(f.shape)}")
    except Exception as e:
        check(f"SyncNet(mode={mode!r}) forward", False, f"{type(e).__name__}: {e}")

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED - fix before training:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All shape checks passed. Safe to train `ssl`.")
