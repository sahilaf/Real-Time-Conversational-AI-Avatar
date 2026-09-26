"""Mouth opening and width for every frame of one or more videos.

  opening = inner-lip gap / face width        (how far the mouth is open)
  width   = outer-lip span / face width       (spread vs rounded lips)

Uses the repository's own landmark detector, so it must run in the SyncTalk_2D
environment. Grouping by Bangla sound happens later, in phoneme_eval.py.

  python mouth_by_phoneme.py out.npz N_FRAMES video1.mp4 [video2.mp4 ...]
"""
import os
import sys
import tempfile

import cv2
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'SyncTalk_2D')
INNER_UPPER, INNER_LOWER = [103, 104, 105], [107, 108, 109]
OUTER_RING = list(range(90, 102))


def load_detector():
    cwd = os.getcwd()
    os.chdir(REPO)
    sys.path.insert(0, os.path.join(REPO, 'data_utils'))
    from get_landmark import Landmark
    lm = Landmark()
    os.chdir(cwd)
    return lm


def measure(video, lm, n):
    tmp = os.path.join(tempfile.gettempdir(), f'_mouth_{os.getpid()}.bmp')
    cap = cv2.VideoCapture(video)
    opening, width = np.full(n, np.nan), np.full(n, np.nan)
    for j in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(tmp, frame)
        cwd = os.getcwd()
        os.chdir(REPO)
        try:
            r = lm.detect(tmp)
        finally:
            os.chdir(cwd)
        if r is None:
            continue
        p = r[0].astype(float)
        face = p[31, 0] - p[1, 0]
        if face <= 0:
            continue
        opening[j] = (p[INNER_LOWER, 1].mean() - p[INNER_UPPER, 1].mean()) / face
        width[j] = (p[OUTER_RING, 0].max() - p[OUTER_RING, 0].min()) / face
    cap.release()
    return opening, width


if __name__ == '__main__':
    out, n, videos = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    lm = load_detector()
    res = {}
    for v in videos:
        name = os.path.splitext(os.path.basename(v))[0]
        o, w = measure(v, lm, n)
        res[f'{name}_opening'], res[f'{name}_width'] = o, w
        print(f'{name}: {np.isfinite(o).sum()}/{n} frames measured', flush=True)
    np.savez(out, **res)
