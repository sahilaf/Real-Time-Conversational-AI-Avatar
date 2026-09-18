"""Score every HDTF benchmark output: LSE-D/LSE-C, leakage, articulation.

One pass over 78 jobs - 6 identities x (1 ground truth + 6 systems x 2
conditions). Six workers; the per-frame landmarker is CPU-bound on SCRFD, so
the pool is the difference between ~30 minutes and ~3 hours.

  real     -> LSE-D/LSE-C + articulation (open_fraction)
  silence  -> LipLeak only. There is nothing to be in sync WITH, so LSE is not
              computed; any mouth motion here is leakage from the input video.

Ground truth is scored as a row, not assumed. Real video is about LSE-D 7.4 /
LSE-C 5.1 - not 0 and not infinity - and without that anchor Wav2Lip's LSE-C
reads as better-than-human.
"""
import csv
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

D = os.environ.get('HDTF_OUT', '/content/drive/MyDrive/FYDP/benchmarks/20260917_stage2/hdtf')
CLIP = os.environ.get('HDTF_CLIPS', '/content/hdtf/clips')
# Default to Drive, not /content. A Colab runtime can be reclaimed at any
# moment and takes /content with it.
OUT = os.environ.get(
    'HDTF_SCORES',
    '/content/drive/MyDrive/FYDP/benchmarks/20260917_stage2/hdtf_scores.csv')
IDS = json.load(open(os.environ.get('HDTF_SUBSET', '/content/hdtf_subset6.json')))
SYSTEMS = ['wav2lip_gan', 'wav2lip', 'musetalk_v1', 'musetalk_v15',
           'latentsync', 'ip_lap']


def jobs():
    for i in IDS:
        yield dict(identity=i, system='ground_truth', cond='real',
                   video=f'{CLIP}/{i}.mp4', audio=f'{CLIP}/{i}.wav')
        for s in SYSTEMS:
            for c in ['real', 'silence']:
                yield dict(identity=i, system=s, cond=c,
                           video=f'{D}/{i}__{s}__{c}.avi',
                           audio=f'{CLIP}/{i}.wav' if c == 'real' else None)


def run(j):
    import numpy as np

    import aperture
    import lse as LSE
    r = dict(j)
    r.pop('audio', None)
    try:
        if not os.path.exists(j['video']):
            r['error'] = 'missing'
            return r
        aps = aperture.video_apertures(j['video'])
        frac, nd, nt = aperture.open_fraction(aps)
        v = np.array([a for a in aps if a is not None])
        r['open_fraction'] = round(float(frac), 4)
        r['median_aperture'] = round(float(np.median(v)), 4) if len(v) else None
        r['frames'], r['detected'] = nt, nd
        if j['cond'] == 'real':
            tag = f"{j['identity']}_{j['system']}"
            muxed = f'/dev/shm/{tag}.avi'
            if LSE.mux_pcm(j['video'], j['audio'], muxed):
                d, c, off, err = LSE.lse(muxed, tag)
                r['LSE_D'], r['LSE_C'], r['av_offset'] = d, c, off
                if err:
                    r['error'] = err[:200]
                os.remove(muxed)
            else:
                r['error'] = 'mux failed'
    except Exception:
        r['error'] = traceback.format_exc()[-300:]
    return r


KEYS = ['identity', 'system', 'cond', 'open_fraction', 'median_aperture',
        'LSE_D', 'LSE_C', 'av_offset', 'frames', 'detected', 'error']


def already_scored(path):
    """(identity, system, cond) triples already in the CSV, so a rerun resumes."""
    if not os.path.exists(path):
        return set()
    with open(path, newline='') as f:
        return {(r['identity'], r['system'], r['cond']) for r in csv.DictReader(f)}


if __name__ == '__main__':
    have = already_scored(OUT)
    J = [j for j in jobs()
         if (j['identity'], j['system'], j['cond']) not in have]
    print(f'{len(J)} jobs to run, {len(have)} already scored', flush=True)
    t0 = time.time()
    fresh = not os.path.exists(OUT)
    # Append and fsync per job. Holding results in memory until the end loses
    # everything when the runtime is reclaimed - which is exactly what happened
    # on 2026-09-18 at 29/78, about 20 minutes of compute.
    with open(OUT, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=KEYS, extrasaction='ignore')
        if fresh:
            w.writeheader()
            f.flush()
        with Pool(6) as p:
            for n, r in enumerate(p.imap_unordered(run, J), 1):
                w.writerow(r)
                f.flush()
                os.fsync(f.fileno())
                print(f"[{n}/{len(J)}] {r['identity']} {r['system']} {r['cond']} "
                      f"open={r.get('open_fraction')} LSE_C={r.get('LSE_C')} "
                      f"{'ERR ' + str(r.get('error'))[:80] if r.get('error') else ''}",
                      flush=True)
    print(f'DONE {time.time() - t0:.0f}s -> {OUT}', flush=True)
    print('SCORING_DONE', flush=True)
