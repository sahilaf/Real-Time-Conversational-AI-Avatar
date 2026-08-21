"""The acceptance test: does the SyncNet actually measure lip-sync?

A falling validation loss is not evidence of that. A model can separate
positives from negatives using anything that happens to correlate - overall
energy, speaker identity, a recording artefact - and still supply no gradient
about whether the mouth matches the sound. That failure has already cost this
project one full training round, which is why the rule in CHECKLIST.md is that
the offset curve must peak before any result is believed.

The test shifts the audio against the video by a fixed number of frames and
scores TRUE pairs only. A SyncNet that has learned sync must score highest at
offset 0 and fall away on both sides. A flat curve means it is reading
something else.

Run on HELD-OUT SPEAKERS. Scored on the training voices it would look sharp
whether or not the mapping generalises, which is the thing being tested.

    python eval_sync_offset.py --ckpt syncnet_ckpt/universal_bn/best_val.pth
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader

from syncnet_328 import SyncNet_color
from syncnet_corpus import CorpusDataset, split_speakers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--corpus", default="../research/corpus/processed")
    p.add_argument("--manifest", default="../research/corpus/manifest.json")
    p.add_argument("--asr", default="ave")
    p.add_argument("--max_offset", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--samples", type=int, default=4000,
                   help="frames per offset; the whole val set is unnecessary")
    p.add_argument("--val_speakers", type=int, default=4)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--cache_dir", default="")
    p.add_argument("--on_train_speakers", action="store_true",
                   help="score the TRAINING voices instead - for contrast only")
    a = p.parse_args()

    man = json.loads(open(a.manifest).read())
    train_spk, val_spk, _ = split_speakers(man, a.val_speakers)
    spk = train_spk if a.on_train_speakers else val_spk
    print(f"scoring {'TRAIN' if a.on_train_speakers else 'HELD-OUT'} "
          f"speakers: {', '.join(sorted(spk))}\n")

    model = SyncNet_color(a.asr).cuda()
    model.load_state_dict(torch.load(a.ckpt, map_location="cuda"))
    model.eval()

    offsets = list(range(-a.max_offset, a.max_offset + 1))
    scores = []
    for off in offsets:
        ds = CorpusDataset(a.corpus, man, spk, a.asr, stride=a.stride,
                           neg_prob=0.0,          # true pairs only
                           cache_dir=a.cache_dir or None, audio_offset=off)
        n = min(a.samples, len(ds))
        # a fixed stride subset, not a random one, so every offset is scored on
        # exactly the same frames and the curve compares like with like
        sub = torch.utils.data.Subset(ds, list(range(0, len(ds), max(1, len(ds) // n)))[:n])
        dl = DataLoader(sub, batch_size=a.batch_size, num_workers=a.num_workers)
        sims = []
        with torch.no_grad():
            for face, aud, _ in dl:
                ae, fe = model(face.cuda(), aud.cuda())
                sims += cosine_similarity(ae, fe).tolist()
        scores.append(float(np.mean(sims)))
        print(f"  offset {off:+3d}   mean similarity {scores[-1]:.4f}")

    best = int(np.argmax(scores))
    peak = offsets[best]
    at0 = scores[offsets.index(0)]
    edges = [scores[0], scores[-1]]
    contrast = at0 - float(np.mean(edges))

    print(f"\n{'peak at offset':<28}{peak:>+8d}")
    print(f"{'score at 0':<28}{at0:>8.4f}")
    print(f"{'mean at +/-max':<28}{float(np.mean(edges)):>8.4f}")
    print(f"{'contrast (0 - edges)':<28}{contrast:>8.4f}")

    print()
    if peak != 0:
        print(f"FAIL: the curve peaks at {peak:+d}, not 0. The audio and video are")
        print("      misaligned, or the model is reading something other than sync.")
    elif contrast < 0.05:
        print("FAIL: the curve is flat. It separates pairs using something that is")
        print("      not lip-sync, so it cannot supply a useful gradient to the")
        print("      generator. Do not train against this checkpoint.")
    else:
        print("PASS: peaks at 0 with real contrast - it is measuring sync, and it")
        print("      does so on speakers it has never heard.")
    print("\nCompare with --on_train_speakers. A sharp curve on training voices")
    print("beside a flat one here is exactly the speaker entanglement this")
    print("corpus was built to remove.")


if __name__ == "__main__":
    main()
