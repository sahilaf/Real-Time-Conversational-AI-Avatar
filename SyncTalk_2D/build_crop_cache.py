"""Decode every crop once into a memmap, so training never decodes a JPEG.

The training job is bottlenecked on JPEG decode, not on the GPU. On an A100
box that means an 80 GB card sits mostly idle while eight CPU workers unpack
250k images over and over - once per epoch, thirty times.

This decodes each frame exactly once into a flat uint8 memmap. After that a
sample is a memcpy out of the page cache instead of a full JPEG decode plus
resize, which is roughly two orders of magnitude cheaper.

    250,353 frames x 320 x 320 x 3 = 76.9 GB

That does not fit in RAM as per-worker copies, which is why it is a FILE and
not an in-process array: every DataLoader worker mmaps the same pages, and the
OS keeps them resident in whatever RAM is spare. On a 167 GB Colab A100 box the
whole thing stays cached after the first pass. Put it on local-scratch (368 GB,
otherwise unused), never on Drive.

The cache is keyed to the manifest's clean segments at stride 1, so one build
serves any --stride you later train with.

Usage:
    python build_crop_cache.py --corpus /content/corpus \\
        --manifest /content/manifest.json --cache_dir /content/boxcache \\
        --out /content/scratch/crops
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

SIDE = 320


def plan(manifest, corpus, cache_dir, min_box_ratio=0.35):
    """Which (video, frame) pairs go in the cache, and in what order."""
    from syncnet_corpus import CorpusDataset
    videos, layout, total = sorted(manifest["videos"]), {}, 0
    for vid in videos:
        d = os.path.join(corpus, vid)
        boxes = CorpusDataset._boxes(d, vid, cache_dir)
        usable = []
        for a, b in manifest["videos"][vid]["segments"]:
            usable.extend(range(a, b))
        w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        med = max(float(np.median(np.maximum(w, 1))), 1.0)
        ok = (w > 0) & (h > 0) & (w >= med * min_box_ratio) & \
             (h >= med * min_box_ratio)
        frames = [f for f in usable if f < len(ok) and ok[f]]
        layout[vid] = {"start": total, "frames": frames, "boxes": boxes}
        total += len(frames)
    return videos, layout, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/content/corpus")
    ap.add_argument("--manifest", default="/content/manifest.json")
    ap.add_argument("--cache_dir", default="/content/boxcache")
    ap.add_argument("--out", default="/content/scratch/crops")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    man = json.loads(open(a.manifest).read())
    videos, layout, total = plan(man, a.corpus, a.cache_dir)

    os.makedirs(a.out, exist_ok=True)
    dat = os.path.join(a.out, "crops.dat")
    meta_path = os.path.join(a.out, "crops.json")
    nbytes = total * SIDE * SIDE * 3
    print(f"{total:,} frames -> {nbytes/1e9:.1f} GB at {SIDE}x{SIDE}")

    # shutil, not os.statvfs: statvfs does not exist on Windows, and this
    # script has to be testable off Colab
    import shutil
    free = shutil.disk_usage(a.out).free
    print(f"free on {a.out}: {free/1e9:.1f} GB")
    if free < nbytes * 1.05:
        raise SystemExit("not enough space - point --out at local-scratch")

    if os.path.exists(meta_path) and not a.force:
        meta = json.loads(open(meta_path).read())
        if meta.get("total") == total and os.path.getsize(dat) == nbytes:
            print("cache already built and matches the manifest")
            return
        print("existing cache does not match the manifest, rebuilding")

    arr = np.memmap(dat, dtype=np.uint8, mode="w+",
                    shape=(total, SIDE, SIDE, 3))
    t0 = time.time()
    done = 0
    for vid in videos:
        info = layout[vid]
        img_dir = os.path.join(a.corpus, vid, "full_body_img")
        boxes, start, frames = info["boxes"], info["start"], info["frames"]

        def one(j):
            f = frames[j]
            img = cv2.imread(os.path.join(img_dir, f"{f}.jpg"))
            if img is None:
                return j, None
            x0, y0, x1, y1 = boxes[f]
            region = img[y0:y1, x0:x1]
            if region.size == 0:
                return j, None
            c = cv2.resize(region, (328, 328), cv2.INTER_AREA)
            return j, c[4:324, 4:324]

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for j, crop in ex.map(one, range(len(frames))):
                if crop is None:            # planned out, but be explicit
                    raise RuntimeError(f"{vid} frame {frames[j]} unreadable")
                arr[start + j] = crop
        done += len(frames)
        rate = done / max(time.time() - t0, 1e-6)
        print(f"  {vid:<24}{len(frames):>7} frames   "
              f"{rate:>6.0f}/s   {(total-done)/rate/60:>5.1f} min left")

    arr.flush()
    del arr
    json.dump({"total": total, "side": SIDE,
               "videos": {v: {"start": layout[v]["start"],
                              "frames": layout[v]["frames"]} for v in videos}},
              open(meta_path, "w"))
    print(f"\nbuilt in {(time.time()-t0)/60:.1f} min -> {dat}")
    print("Pass --crop_cache to syncnet_corpus.py to train from it.")


if __name__ == "__main__":
    main()
