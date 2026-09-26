"""Bangla phoneme-level lip evaluation — reproduces Tables 4.3 and 4.4 of the report.

Question: does the mouth close on প ফ ব ভ ম and open on আ অ, as Bangla requires?
Measured as the ratio of mean mouth opening on "open" sounds to "closed" sounds.

Pipeline (two environments, because of a transformers/torch version rule):

  1. Sound timings — needs torch >= 2.6 + transformers:
       python bangla_phonemes.py <clip.wav> results/phoneme/phon_<name>.npz
  2. Mouth measurements — needs the SyncTalk_2D environment (landmark detector):
       python mouth_by_phoneme.py results/phoneme/mouth_testclip.npz 771
           redwan_test.mp4 phon_ours.mp4 phon_before.mp4
       python mouth_by_phoneme.py results/phoneme/mouth_cross.npz 771 x_ours_s02.mp4 ...
     (x_<ours|before>_s<speaker>.mp4 = each model driven by a corpus speaker's audio,
      rendered with inference_328.py --start_frame 6941 --ref_frame 2861)
  3. This script — any Python with numpy:
       python phoneme_eval.py

Timing calibration. The speech model marks each sound slightly after it starts,
and lips move just before a sound. The shift is chosen ONCE, on the real video
only, as the one that best separates open from closed sounds in the real
speaker's own mouth, and is then applied unchanged to every model. It is never
tuned on model output.
"""
import os

import numpy as np

from bangla_phonemes import assign

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results', 'phoneme')
SPEAKERS = ('02', '04', '08', '11', '23', '24')   # corpus speakers, unseen by the model
# The saved results name the final model 'ours'; it is called Alapon in the report.
NAMES = {'ours': 'Alapon', 'ours before fix': 'before fix', 'before': 'before fix'}


def ratio(opening, groups):
    return np.nanmean(opening[groups == 'open']) / np.nanmean(opening[groups == 'closed'])


def calibrate(ex, real_opening):
    best = None
    for mode in ('nearest', 'hold'):
        for off in range(-8, 9):
            g = assign(ex, len(real_opening), mode, off)
            gap = np.nanmean(real_opening[g == 'open']) - np.nanmean(real_opening[g == 'closed'])
            if best is None or gap > best[0]:
                best = (gap, mode, off)
    return best[1], best[2]


def by_system(npz_path, shift_range=range(-6, 7)):
    """Test-clip table for every benchmarked system (rendered on Drive, measured on
    Colab). Each video is scored at its own best timing shift, the same rule that
    calibrates the real video, so no system is penalised for a constant delay."""
    d = np.load(npz_path, allow_pickle=True)
    ex = {'spike_t': d['spike_t'], 'spike_ch': d['spike_ch'], 'rms': d['rms']}
    rows = []
    for i, name in enumerate(d['keys']):
        op = d[f'op{i}'].astype(float)
        def gap(s):
            g = assign(ex, len(op), 'nearest', s)
            return np.nanmean(op[g == 'open']) - np.nanmean(op[g == 'closed'])
        best = max(shift_range, key=gap)
        g = assign(ex, len(op), 'nearest', best)
        c, o = np.nanmean(op[g == 'closed']), np.nanmean(op[g == 'open'])
        rows.append((NAMES.get(str(name), str(name)), best, c, o, o / c))
    return rows


if __name__ == '__main__':
    print('TEST CLIP - all benchmarked systems (Table 4.3)')
    print(f'  {"system":16s} {"shift":>5s} {"closed":>7s} {"open":>7s} {"open/closed":>12s}')
    for name, best, c, o, r in by_system(os.path.join(RES, 'testclip_all_systems.npz')):
        print(f'  {name:16s} {best:+5d} {c:7.3f} {o:7.3f} {r:11.1f}x')

    # six unseen corpus speakers, ours vs before fix (rendered and measured locally)
    test = np.load(os.path.join(RES, 'mouth_testclip.npz'), allow_pickle=True)
    ex = dict(np.load(os.path.join(RES, 'phon_redwan_test.npz'), allow_pickle=True))
    mode, off = calibrate(ex, test['redwan_test_opening'])
    print()
    print(f'SIX NEW BANGLA SPEAKERS (Table 4.4) - timing from the real video: '
          f'{mode}, {off:+d} frames ({off * 40:+d} ms)')
    cross = np.load(os.path.join(RES, 'mouth_cross.npz'), allow_pickle=True)
    pooled, groups = {'ours': [], 'before': []}, []
    for s in SPEAKERS:
        exs = dict(np.load(os.path.join(RES, f'phon_speaker{s}_video01.npz'), allow_pickle=True))
        gs = assign(exs, len(cross[f'x_ours_s{s}_opening']), mode, off)
        groups.append(gs)
        row = []
        for m in ('ours', 'before'):
            o = cross[f'x_{m}_s{s}_opening']
            pooled[m].append(o)
            row.append(ratio(o, gs))
        print(f'  speaker {s}: Alapon {row[0]:.2f}x   before fix {row[1]:.2f}x')
    groups = np.concatenate(groups)
    for m in ('ours', 'before'):
        print(f'  ALL SIX   {NAMES.get(m, m):10s} {ratio(np.concatenate(pooled[m]), groups):.1f}x')
