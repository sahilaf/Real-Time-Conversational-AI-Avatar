"""Slide asset: the redwan training video and how it becomes Alapon's input.

    python benchmark/make_redwan_asset.py

Reads SyncTalk_2D/dataset/redwan (local, not in git) and writes
docs/fydp/figures/poster/fig_redwan_data.png. Every step drawn here is the real
one: the crop box comes from landmarks 1, 31 and 52 exactly as datasetsss_328.py
computes it, the mask is utils.apply_mouth_mask (jaw hidden), and the mel
settings match utils.melspectrogram (16 kHz, n_fft 800, 80 bins, 55-7600 Hz).
Frame 2861 is the fixed reference frame Alapon uses.
"""
import os
import wave

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'SyncTalk_2D', 'dataset', 'redwan')
OUT = os.path.join(ROOT, 'docs', 'fydp', 'figures', 'poster')
FRAME = 2861
N_FRAMES, TRAIN_END, VAL_END = 7717, 6170, 6941      # evaluation/manifests/redwan_splits.json

ACCENT, ACCENT_SOFT, COUNTER = '#2a78d6', '#e3eefb', '#eb6834'
INK, INK2, MUTED, GRID, REST = '#121a2a', '#52514e', '#898781', '#e1e0d9', '#c9c7bf'
plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'figure.facecolor': '#ffffff',
                     'savefig.facecolor': '#ffffff'})


def load():
    img = np.asarray(Image.open(os.path.join(DATA, 'full_body_img', f'{FRAME}.jpg')).convert('RGB'))
    lms = np.loadtxt(os.path.join(DATA, 'landmarks', f'{FRAME}.lms')).astype(int)
    return img, lms


def crop_box(lms):
    """Exactly datasetsss_328.process_img: x from landmark 1 to 31, y from 52, square."""
    xmin, xmax, ymin = lms[1][0], lms[31][0], lms[52][1]
    return xmin, ymin, xmax, ymin + (xmax - xmin)


def face_crop(img, box):
    x0, y0, x1, y1 = box
    c = Image.fromarray(img[y0:y1, x0:x1]).resize((328, 328), Image.LANCZOS)
    return np.asarray(c)[4:324, 4:324].copy()


def mel_spectrogram(seconds=6.0, start=120.0):
    with wave.open(os.path.join(DATA, 'aud.wav')) as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        w.setpos(int(start * sr))
        raw = w.readframes(int(seconds * sr))
    dtype = {2: np.int16, 4: np.int32}[width]
    x = np.frombuffer(raw, dtype=dtype).reshape(-1, ch)[:, 0].astype(np.float32)
    x /= np.abs(x).max() + 1e-9
    n_fft, hop = 800, 200
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 1
    # 80 triangular mel filters, 55-7600 Hz
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = imel(np.linspace(mel(55), mel(7600), 82))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((80, n_fft // 2 + 1))
    for m in range(1, 81):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    S = 20 * np.log10(np.maximum(fb @ spec.T, 1e-5))
    return x, sr, np.clip(S, S.max() - 70, None)


def label(ax, n, title, sub):
    ax.set_title(f'{title}', fontsize=13, color=INK, weight='bold', loc='left', pad=22)
    ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=10.5, color=INK2, va='bottom')
    ax.text(-0.02, 1.2, str(n), transform=ax.transAxes, fontsize=11, color='white', weight='bold',
            ha='center', va='center', bbox=dict(boxstyle='circle,pad=0.3', fc=INK, ec='none'))


def arrow(fig, a, b):
    """Horizontal arrow between two axes, in figure coordinates."""
    ra, rb = a.get_position(), b.get_position()
    y = (ra.y0 + ra.y1) / 2
    fig.add_artist(FancyArrowPatch((ra.x1 + 0.006, y), (rb.x0 - 0.006, y), transform=fig.transFigure,
                                   arrowstyle='-|>', mutation_scale=16, color=INK, lw=1.6))


def main():
    img, lms = load()
    box = crop_box(lms)
    crop = face_crop(img, box)
    masked = crop.copy()
    masked[5:, 5:315] = 0                        # utils.apply_mouth_mask, v2_no_jaw
    x, sr, S = mel_spectrogram()

    fig = plt.figure(figsize=(15, 8.4))
    # no title: the slide title carries it
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 0.36, 0.14], hspace=0.62, wspace=0.28,
                          left=0.03, right=0.985, top=0.92, bottom=0.05)

    # 1 source frame with the crop box
    a1 = fig.add_subplot(gs[0, 0])
    a1.imshow(img)
    x0, y0, x1, y1 = box
    a1.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=ACCENT, lw=2.4))
    label(a1, 1, 'Source frame', '25 fps, 1080 × 1080')
    # 2 landmarks
    a2 = fig.add_subplot(gs[0, 1])
    pad = 60
    zx0, zy0 = max(lms[:, 0].min() - pad, 0), max(lms[:, 1].min() - pad, 0)
    zx1, zy1 = lms[:, 0].max() + pad, lms[:, 1].max() + pad
    a2.imshow(img[zy0:zy1, zx0:zx1])
    a2.scatter(lms[:, 0] - zx0, lms[:, 1] - zy0, s=9, c=ACCENT, edgecolors='white', linewidths=0.5)
    label(a2, 2, 'Face landmarks', '110 points per frame')
    # 3 face crop
    a3 = fig.add_subplot(gs[0, 2])
    a3.imshow(crop)
    label(a3, 3, 'Face crop', 'from the landmarks, 320 × 320')
    # 4 masked input
    a4 = fig.add_subplot(gs[0, 3])
    a4.imshow(masked)
    a4.add_patch(Rectangle((5, 5 + 290), 310, 25, fill=False, ec=ACCENT, lw=2, ls='--'))
    a4.text(160, 287, 'jaw rows: now hidden too', color=ACCENT, ha='center', va='bottom',
            fontsize=9.5, weight='bold')
    a4.text(160, 250, 'mouth + jaw\nhidden', color='white', ha='center', va='center', fontsize=11,
            weight='bold')
    label(a4, 4, 'Model input', 'the model must repaint this')
    for a in (a1, a2, a3, a4):
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(GRID)
    fig.canvas.draw()
    for a, b in ((a1, a2), (a2, a3), (a3, a4)):
        arrow(fig, a, b)

    # 5 audio: waveform + mel
    a5 = fig.add_subplot(gs[1, 0:2])
    t = np.arange(len(x)) / sr
    a5.plot(t, x, color=INK2, lw=0.4)
    a5.set_xlim(0, t[-1])
    a5.set_yticks([])
    a5.set_xlabel('seconds', fontsize=9.5, color=MUTED)
    a5.tick_params(labelsize=9, colors=MUTED)
    for s in ('top', 'right', 'left'):
        a5.spines[s].set_visible(False)
    label(a5, 5, 'Audio track', '16 kHz, from the same video')
    a6 = fig.add_subplot(gs[1, 2:4])
    a6.imshow(S, origin='lower', aspect='auto', cmap='Blues',
              extent=[0, len(x) / sr, 0, 80])
    a6.set_yticks([])
    a6.set_xlabel('seconds', fontsize=9.5, color=MUTED)
    a6.tick_params(labelsize=9, colors=MUTED)
    for s in a6.spines.values():
        s.set_edgecolor(GRID)
    label(a6, 6, 'Mel spectrogram → audio features', '80 bins, one feature vector per video frame')
    fig.canvas.draw()
    arrow(fig, a5, a6)

    # 7 split
    a7 = fig.add_subplot(gs[2, :])
    parts = [(0, TRAIN_END + 1, ACCENT, f'Training  ·  frames 0–{TRAIN_END}  ·  4.1 min', 'white'),
             (TRAIN_END + 1, VAL_END + 1, REST, 'Validation  ·  31 s', INK),
             (VAL_END + 1, N_FRAMES, INK2, 'Test  ·  31 s', 'white')]
    for s0, s1, c, txt, tc in parts:
        a7.add_patch(Rectangle((s0, 0), s1 - s0 - 12, 1, color=c, lw=0))
        a7.text((s0 + s1) / 2, 0.5, txt, ha='center', va='center', fontsize=10.5, color=tc,
                weight='bold')
    a7.set_xlim(0, N_FRAMES)
    a7.set_ylim(0, 1)
    a7.axis('off')
    a7.set_title('Split into continuous blocks, so near-identical neighbouring frames never '
                 'cross from training into test', fontsize=11, color=INK2, loc='left', pad=6)
    a7.text(-0.004, 1.9, '7', transform=a7.transAxes, fontsize=11, color='white', weight='bold',
            ha='center', va='center', bbox=dict(boxstyle='circle,pad=0.3', fc=INK, ec='none'))

    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, 'fig_redwan_data.png')
    fig.savefig(out, dpi=220, bbox_inches='tight', pad_inches=0.12)
    print('written', out)


if __name__ == '__main__':
    main()
