"""Universal Bangla SyncNet: train across the multi-speaker corpus.

Why this exists rather than pointing syncnet_328.py at more data:

  * Its Dataset reads ONE directory. The corpus is 28 directories.
  * Its train/val split is a contiguous slice of one speaker's timeline. That
    is the wrong split here. The measured fault (research/CONTEXT.md section
    10) is that the audio->mouth mapping is entangled with SPEAKER IDENTITY:
    within a speaker a linear probe scores r = 0.73, across speakers 0.37,
    while the same probe moved to a different RECORDING of the same person
    holds at 0.79. A frame-wise split would let the model memorise 24 voices
    and still show a falling val loss - it would measure nothing we care
    about. Validation here holds out WHOLE SPEAKERS.
  * Frames are only sampled inside the clean segments in manifest.json, so
    black leader, flat cuts and carried-forward landmarks never enter a batch.

Everything else - the model, the loss, AMP, resume, best-val checkpointing -
is imported from syncnet_328 rather than reimplemented.

Negatives are drawn from the SAME video as the positive, never from another
speaker. A cross-speaker negative would be trivially separable by appearance,
and the model would learn to spot the identity mismatch instead of the sync
error - scoring well while learning nothing about lips.

Usage:
    python syncnet_corpus.py --corpus ../research/corpus/processed \\
        --manifest ../research/corpus/manifest.json \\
        --save_dir syncnet_ckpt/universal_bn --amp --batch_size 128
"""
from __future__ import annotations

import argparse
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from syncnet_328 import SyncNet_color, cosine_loss, evaluate  # noqa: F401


def _crop_box(lms: np.ndarray, h: int, w: int) -> tuple[int, int, int, int]:
    """The face box, clipped to the image exactly as inference_328.py does.

    Matching inference matters more than matching the old training code: a
    model trained on boxes that run off the edge would see a different framing
    at test time than it ever saw while learning.
    """
    x0, y0 = max(0, int(lms[1][0])), max(0, int(lms[52][1]))
    x1 = min(w - 1, int(lms[31][0]))
    y1 = min(h - 1, y0 + (x1 - x0))
    return x0, y0, x1, y1


class CorpusDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_root, manifest, speakers, mode="ave", stride=1,
                 neg_prob=0.5, min_neg_gap=5, cache_dir=None, audio_offset=0):
        self.root, self.mode = corpus_root, mode
        self.neg_prob, self.min_neg_gap = neg_prob, min_neg_gap
        # shifts the audio against the video by a fixed number of frames. Used
        # only by the offset-curve test: a SyncNet that has genuinely learned
        # sync must score highest at 0 and fall off either side. One that has
        # learned some other cue gives a flat curve while still showing a
        # respectable val loss - the failure that wasted the first training run.
        self.audio_offset = audio_offset

        vids = {v: e for v, e in manifest["videos"].items()
                if e["speaker"] in speakers}
        if not vids:
            raise ValueError(f"no videos for speakers {sorted(speakers)}")

        self.videos = sorted(vids)
        self.audio = []          # one memmap per video
        self.boxes = []          # (N, 4) int32 crop boxes per video
        self.frames = []         # sampled frame indices per video
        self.valid = []          # every usable frame, for negative sampling
        index = []

        for vi, vid in enumerate(self.videos):
            d = os.path.join(corpus_root, vid)
            # mmap: 28 feature arrays total ~0.7 GB, and every DataLoader worker
            # would otherwise hold its own copy of all of them
            self.audio.append(np.load(os.path.join(d, "aud_ave.npy"),
                                      mmap_mode="r"))
            usable = []
            for a, b in vids[vid]["segments"]:
                usable.extend(range(a, b))
            self.valid.append(np.array(usable, dtype=np.int64))
            taken = usable[::stride]
            self.frames.append(taken)
            self.boxes.append(self._boxes(d, vid, cache_dir))
            index.extend((vi, f) for f in taken)

        self.index = index

    def _boxes(self, d, vid, cache_dir):
        """Crop boxes for every frame, computed once and cached.

        Reading and parsing a .lms file inside __getitem__ costs a small file
        read per sample. Precomputing four ints per frame removes that from
        the hot loop entirely for ~4 MB across the whole corpus.
        """
        cache = os.path.join(cache_dir or d, f"{vid}_boxes.npy")
        if os.path.exists(cache):
            return np.load(cache)
        lms_dir = os.path.join(d, "landmarks")
        n = len([f for f in os.listdir(lms_dir) if f.endswith(".lms")])
        img0 = cv2.imread(os.path.join(d, "full_body_img", "0.jpg"))
        h, w = img0.shape[:2]

        def one(i):
            # only rows 1, 31 and 52 are needed; parsing all 110 and building a
            # float array per frame made this ~4 minutes per video, and there
            # are 250k frames
            with open(os.path.join(lms_dir, f"{i}.lms")) as fh:
                rows = fh.read().splitlines()
            x0 = max(0, int(float(rows[1].split()[0])))
            y0 = max(0, int(float(rows[52].split()[1])))
            x1 = min(w - 1, int(float(rows[31].split()[0])))
            return i, (x0, y0, x1, min(h - 1, y0 + (x1 - x0)))

        out = np.zeros((n, 4), np.int32)
        # threads, not processes: this is small-file I/O, which releases the GIL
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=16) as ex:
            for i, box in ex.map(one, range(n)):
                out[i] = box
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        tmp = cache.replace(".npy", ".part.npy")
        np.save(tmp, out)
        os.replace(tmp, cache)      # a killed run must not leave a half cache
        return out

    def __len__(self):
        return len(self.index)

    def _audio_window(self, feats, i):
        left, right = i - 8, i + 8
        pl, pr = max(0, -left), max(0, right - feats.shape[0])
        w = np.asarray(feats[max(0, left):min(right, feats.shape[0])],
                       dtype=np.float32)
        if pl:
            w = np.concatenate([np.zeros((pl, w.shape[1]), np.float32), w])
        if pr:
            w = np.concatenate([w, np.zeros((pr, w.shape[1]), np.float32)])
        return torch.from_numpy(w)

    def __getitem__(self, k):
        vi, idx = self.index[k]
        d = os.path.join(self.root, self.videos[vi])
        img = cv2.imread(os.path.join(d, "full_body_img", f"{idx}.jpg"))
        x0, y0, x1, y1 = self.boxes[vi][idx]
        crop = cv2.resize(img[y0:y1, x0:x1], (328, 328), cv2.INTER_AREA)
        face = torch.from_numpy(
            crop[4:324, 4:324].transpose(2, 0, 1).astype(np.float32) / 255.0)

        # negative from the same video only - see module docstring
        pool = self.valid[vi]
        if random.random() < self.neg_prob and len(pool) > 4 * self.min_neg_gap:
            for _ in range(20):
                wrong = int(random.choice(pool))
                if abs(wrong - idx) >= self.min_neg_gap:
                    break
            else:
                wrong = idx
            src, y = wrong, torch.zeros(1)
        else:
            src, y = idx, torch.ones(1)

        src = int(np.clip(src + self.audio_offset, 0,
                          self.audio[vi].shape[0] - 1))
        aud = self._audio_window(self.audio[vi], src)
        aud = aud.reshape(32, 16, 16) if self.mode == "ave" else aud.reshape(16, 32, 32)
        return face, aud, y.float()


def split_speakers(manifest, n_val, seed=0):
    """Hold out whole speakers, spread across the size distribution.

    Picking at random would happily hand back four one-minute speakers and a
    validation set too small and too easy to mean anything. Speakers are
    ordered by duration and sampled at even intervals instead.
    """
    mins = {}
    for e in manifest["videos"].values():
        mins[e["speaker"]] = mins.get(e["speaker"], 0) + e["frames"]
    order = [s for s, _ in sorted(mins.items(), key=lambda kv: -kv[1])]
    picks = np.linspace(0, len(order) - 1, n_val + 2)[1:-1].astype(int)
    val = {order[i] for i in dict.fromkeys(picks)}
    train = set(order) - val
    return train, val, mins


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="../research/corpus/processed")
    p.add_argument("--manifest", default="../research/corpus/manifest.json")
    p.add_argument("--save_dir", default="syncnet_ckpt/universal_bn")
    p.add_argument("--asr", default="ave")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--stride", type=int, default=2,
                   help="sample every Nth frame. Adjacent frames at 25 fps are "
                        "near-duplicates, so stride 2 halves the compute per "
                        "epoch for almost no loss of diversity")
    p.add_argument("--val_stride", type=int, default=8,
                   help="validation runs every epoch and only has to be a "
                        "stable estimate, not an exhaustive one")
    p.add_argument("--val_speakers", type=int, default=4)
    p.add_argument("--patience", type=int, default=5,
                   help="stop after this many epochs with no val improvement; "
                        "0 disables and runs the full --epochs")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume", default="")
    p.add_argument("--cache_dir", default="",
                   help="where to keep the crop-box caches (default: alongside each video)")
    p.add_argument("--dry_run", action="store_true",
                   help="build the datasets, report the split, and stop")
    a = p.parse_args()

    man = json.loads(open(a.manifest).read())
    train_spk, val_spk, mins = split_speakers(man, a.val_speakers)
    fmt = lambda s: ", ".join(f"{x} ({mins[x]/1500:.1f}m)" for x in sorted(s))
    print(f"train speakers ({len(train_spk)}): {fmt(train_spk)}")
    print(f"val speakers   ({len(val_spk)}): {fmt(val_spk)}")
    print("validation is SPEAKER-DISJOINT: these voices never appear in training\n")

    cache = a.cache_dir or None
    tr = CorpusDataset(a.corpus, man, train_spk, a.asr, a.stride, cache_dir=cache)
    va = CorpusDataset(a.corpus, man, val_spk, a.asr, a.val_stride, cache_dir=cache)
    print(f"train {len(tr):,} samples over {len(tr.videos)} videos")
    print(f"val   {len(va):,} samples over {len(va.videos)} videos")
    steps = len(tr) // a.batch_size
    print(f"{steps:,} steps/epoch at batch {a.batch_size} -> "
          f"{steps*a.epochs:,} steps over {a.epochs} epochs")
    if a.dry_run:
        return

    import syncnet_328 as S
    S.Dataset = None                      # nothing below may use the old one
    # prefetch_factor keeps each worker a few batches ahead so the GPU is not
    # waiting on JPEG decode between steps
    extra = {"prefetch_factor": 4} if a.num_workers > 0 else {}
    train_loader = DataLoader(tr, batch_size=a.batch_size, shuffle=True,
                              num_workers=a.num_workers,
                              persistent_workers=a.num_workers > 0,
                              pin_memory=True, drop_last=True, **extra)
    val_loader = DataLoader(va, batch_size=a.batch_size, shuffle=False,
                            num_workers=max(a.num_workers // 2, 0),
                            persistent_workers=a.num_workers > 1,
                            pin_memory=True)
    _run(a, train_loader, val_loader)


def _run(a, train_loader, val_loader):
    os.makedirs(a.save_dir, exist_ok=True)
    # Measured on an RTX 3050 at batch 32 (AMP, samples/s):
    #   baseline 146 · cudnn.benchmark 144 · +TF32 152 · +channels_last 28
    # TF32 is a real ~4% and free on Ampere. channels_last is FIVE TIMES SLOWER
    # for this network - do not re-add it without measuring again.
    # cudnn.benchmark measured neutral here; kept because input shapes are fixed
    # and it typically pays off on larger cards.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = SyncNet_color(a.asr).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    use_amp = a.amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best, start, stale, best_ep = float("inf"), 0, 0, 0
    log = os.path.join(a.save_dir, "train_log.csv")
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location="cuda")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        if use_amp and "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"], ck.get("best_val_loss", float("inf"))
        # carried across resumes: otherwise restarting a run resets the
        # patience counter and it can never stop
        stale, best_ep = ck.get("stale", 0), ck.get("best_epoch", 0)
        print(f"resumed at epoch {start} (best {best:.4f} at epoch {best_ep})")
    if not os.path.exists(log):
        open(log, "w").write("epoch,train_loss,val_loss,val_pos_sim,val_neg_sim,gap\n")

    from tqdm import tqdm
    for ep in range(start, a.epochs):
        model.train(); losses = []
        for face, aud, y in tqdm(train_loader, desc=f"epoch {ep+1}/{a.epochs}"):
            face, aud, y = face.cuda(non_blocking=True), aud.cuda(non_blocking=True), y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                ae, fe = model(face, aud)
            # BCELoss is unsafe under autocast - score in fp32, outside it
            loss = cosine_loss(ae.float(), fe.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(loss.item())
        tl = sum(losses) / len(losses)
        vl, pos, neg = evaluate(model, val_loader)
        print(f"epoch {ep+1}  train {tl:.4f}  val {vl:.4f}  "
              f"pos {pos:.4f}  neg {neg:.4f}  gap {pos-neg:.4f}")
        open(log, "a").write(f"{ep+1},{tl:.6f},{vl:.6f},{pos:.6f},{neg:.6f},{pos-neg:.6f}\n")
        if vl < best:
            best, stale, best_ep = vl, 0, ep + 1
            torch.save(model.state_dict(), os.path.join(a.save_dir, "best_val.pth"))
            print(f"   new best ({best:.4f}) saved")
        else:
            stale += 1
            print(f"   no improvement for {stale} epoch(s); "
                  f"best {best:.4f} at epoch {best_ep}")
        torch.save({"epoch": ep + 1, "model": model.state_dict(),
                    "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "best_val_loss": best, "stale": stale, "best_epoch": best_ep},
                   os.path.join(a.save_dir, "last.pth"))
        if a.patience and stale >= a.patience:
            print(f"\nearly stop: {a.patience} epochs without improvement.")
            break

    print(f"\nbest val {best:.4f} at epoch {best_ep} -> best_val.pth")
    if best_ep == a.epochs:
        print("NOTE: the best epoch is the LAST one - validation was still")
        print("      improving when the run ended. Raise --epochs and re-run;")
        print("      it resumes from last.pth rather than starting over.")


if __name__ == "__main__":
    main()
