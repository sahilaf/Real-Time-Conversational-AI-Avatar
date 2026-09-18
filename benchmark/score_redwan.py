"""Score the redwan (Bangla) outputs with the same code as the HDTF table.

Two datasets scored by two implementations cannot share a figure - any
difference in threshold, segment length or LSE invocation shows up as a dataset
effect. This exists so both tables come from one scorer.

Unlike the HDTF outputs, most of these files already carry PCM audio, and some
of them must NOT be re-muxed. See the README: several declare avg_frame_rate
50/1 over 772 real frames, and re-containering those moves LSE-C by ~1.0.
"""
import csv
import os
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

R1 = os.environ.get('REDWAN_DIR',
                    '/content/drive/MyDrive/FYDP/benchmarks/20260913_redwan_test')
R2 = os.environ.get('STAGE2_DIR',
                    '/content/drive/MyDrive/FYDP/benchmarks/20260917_stage2')
AUD = f'{R1}/inputs/redwan_test.wav'
OUT = f'{R2}/redwan_scores.csv'
KEYS = ['identity', 'system', 'cond', 'open_fraction', 'median_aperture',
        'LSE_D', 'LSE_C', 'av_offset', 'frames', 'detected', 'error']

# System names match the HDTF table so the two can sit in one figure.
# 'wav2lip' is the NON-GAN checkpoint, as on HDTF.
# ip_lap uses the right-half crop; the archived original is an uncropped
# 2160x1080 [sketch|result] composite and scores nonsense as found.
FILES = {
    'ground_truth':           (f'{R1}/videos_pcm/ground_truth.avi', None),
    'wav2lip_gan':            (f'{R1}/videos_pcm/wav2lip.avi',
                               f'{R1}/videos_silence/wav2lip_SILENT.mp4'),
    'latentsync':             (f'{R1}/videos_pcm/latentsync15.avi',
                               f'{R1}/videos_silence/latentsync15_SILENT.mp4'),
    'musetalk_v15':           (f'{R1}/videos_pcm/musetalk15.avi',
                               f'{R1}/videos_silence/musetalk15_SILENT.mp4'),
    'synctalk2d_final_v2':    (f'{R1}/videos_pcm/synctalk2d_final_v2.avi',
                               f'{R1}/videos_silence/synctalk2d_final_v2_SILENT.mp4'),
    'synctalk2d_legacy_fix':  (f'{R1}/videos_silence/rung0_legacy_fixedref_real.mp4',
                               f'{R1}/videos_silence/rung0_legacy_fixedref_SILENT.mp4'),
    'synctalk2d_legacy_ship': (None,
                               f'{R1}/videos_silence/rung0_legacy_asshipped_SILENT.mp4'),
    'wav2lip':                (f'{R2}/videos_redwan/wav2lip_nogan.avi', None),
    'musetalk_v1':            (f'{R2}/videos_redwan/musetalk_v1.0.avi', None),
    'ip_lap':                 (f'{R2}/videos_redwan/ip_lap_rightcrop.avi', None),
}


def has_audio(path):
    out = subprocess.run(
        f'ffprobe -v error -select_streams a:0 -show_entries stream=codec_name '
        f'-of csv=p=0 "{path}"', shell=True, capture_output=True, text=True).stdout
    return bool(out.strip())


def jobs():
    for s, (rv, sv) in FILES.items():
        if rv:
            yield dict(identity='redwan', system=s, cond='real', video=rv, audio=AUD)
        if sv:
            yield dict(identity='redwan', system=s, cond='silence', video=sv, audio=None)


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
            tag = f"redwan_{j['system']}"
            # Score AS-IS when audio is already present. Re-containering a file
            # that declares 50/1 over 772 real frames changes what run_pipeline
            # extracts at -r 25 and moves LSE-C by ~1.0.
            if has_audio(j['video']):
                src, tmp = j['video'], None
            else:
                tmp = f'/dev/shm/{tag}.avi'
                src = tmp if LSE.mux_pcm(j['video'], j['audio'], tmp) else None
                if not src:
                    r['error'] = 'mux failed'
            if src:
                d, c, off, err = LSE.lse(src, tag)
                r['LSE_D'], r['LSE_C'], r['av_offset'] = d, c, off
                if err:
                    r['error'] = err[:200]
                if tmp and os.path.exists(tmp):
                    os.remove(tmp)
    except Exception:
        r['error'] = traceback.format_exc()[-300:]
    return r


def already_scored(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline='') as f:
        return {(r['system'], r['cond']) for r in csv.DictReader(f)}


if __name__ == '__main__':
    have = already_scored(OUT)
    J = [j for j in jobs() if (j['system'], j['cond']) not in have]
    print(f'{len(J)} jobs to run, {len(have)} already scored', flush=True)
    t0 = time.time()
    fresh = not os.path.exists(OUT)
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
                print(f"[{n}/{len(J)}] {r['system']:24s} {r['cond']:8s} "
                      f"open={r.get('open_fraction')} LSE_C={r.get('LSE_C')}", flush=True)
    print(f'DONE {time.time() - t0:.0f}s', flush=True)
    print('REDWAN_DONE', flush=True)
