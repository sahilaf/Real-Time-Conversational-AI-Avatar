"""Mouth-aperture metrics, faithful to the repo's own definition.

INNER_LIP_UPPER/LOWER and the face-width landmarks are taken from
SyncTalk_2D/avatar_server_ws.py and evaluation/scripts/common_eval.py so that
every dataset is scored with the same detector and the same indices. Do not
re-derive these numbers from a different landmarker and put them in one table.

Two statistics per video:
  open_fraction    fraction of DETECTED frames with aperture > 0.03
  median_aperture  the continuous median - threshold-free, report both

The threshold 0.03 sits at the MEDIAN of the source distribution, which is the
worst place for it: the frames nearest the threshold are the most numerous, so
landmark noise flips them. Measured reproducibility against the 2026-09-13
figures is about +/-0.02 on open_fraction, while median_aperture is stable.
"""
import os
import sys

import cv2
import numpy as np
import torch

REPO = os.environ.get('SYNCTALK_REPO', '/content/repo/SyncTalk_2D')
INNER_LIP_UPPER = [103, 104, 105]
INNER_LIP_LOWER = [107, 108, 109]
THRESH = 0.03

_LM = None


def _lm():
    """Load the repo's landmarker once. It resolves ./data_utils/* from cwd."""
    global _LM
    if _LM is None:
        cwd = os.getcwd()
        os.chdir(REPO)
        sys.path.insert(0, os.path.join(REPO, 'data_utils'))
        # torch >= 2.6 defaults weights_only=True; the PFLD checkpoint is a
        # full pickle and will not load under it.
        _orig = torch.load
        torch.load = lambda *a, **k: _orig(*a, **{**k, 'weights_only': False})
        from get_landmark import Landmark
        _LM = Landmark()
        torch.load = _orig
        os.chdir(cwd)
    return _LM


def frame_aperture(img):
    """Normalised inner-lip gap for one frame, or None if no face was found."""
    # Landmark.detect takes a path. Stage through tmpfs, per-process: a shared
    # path silently corrupts results as soon as this runs in a pool.
    p = f'/dev/shm/_ap_{os.getpid()}.bmp'
    cv2.imwrite(p, img)
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        r = _lm().detect(p)
    finally:
        os.chdir(cwd)
    if r is None:
        return None
    lms = r[0]
    width = float(lms[31][0] - lms[1][0])
    if width <= 0:
        return None
    gap = float(lms[INNER_LIP_LOWER, 1].mean() - lms[INNER_LIP_UPPER, 1].mean())
    return gap / width


def video_apertures(path, limit=None):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok or (limit and len(out) >= limit):
            break
        out.append(frame_aperture(f))
    cap.release()
    return out


def open_fraction(aps, thresh=THRESH):
    """Fraction of DETECTED frames with the mouth open.

    Undetected frames leave the denominator rather than counting as shut -
    counting them shut would reward a system whose output the detector fails on.
    """
    v = [a for a in aps if a is not None]
    return (sum(a > thresh for a in v) / len(v)) if v else float('nan'), len(v), len(aps)
