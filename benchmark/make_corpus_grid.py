"""Face grid of the Bangla corpus: one face per speaker, for the report and poster.

    python benchmark/make_corpus_grid.py

Reads the local corpus (research/corpus/processed, not in git) and writes
docs/fydp/figures/{report,poster}/fig_corpus_speakers.png (+ .pdf for the report).

For each speaker it takes their first processed video, looks at frames from the
middle 60% of it, and picks the frame whose face is largest with the mouth most
nearly closed - a neutral, frontal face. The six speakers used as unseen test
voices in the report (Table 4.4) are outlined in the Alapon blue.
"""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, 'research', 'corpus')
OUT = os.path.join(ROOT, 'docs', 'fydp', 'figures')
TEST_SPEAKERS = {'speaker02', 'speaker04', 'speaker08', 'speaker11', 'speaker23', 'speaker24'}  # file IDs


def display_number(file_id, all_ids):
    """Speakers are shown numbered 1..N in order. The raw folders skip speaker17,
    which failed the quality filter, so file IDs 18-25 display as 17-24
    (the test speakers speaker23 / speaker24 appear as Speaker 22 / 23)."""
    return f'{sorted(all_ids).index(file_id) + 1:02d}'
INNER_UPPER, INNER_LOWER = [103, 104, 105], [107, 108, 109]
ACCENT, INK, INK2, SURFACE = '#2a78d6', '#0b0b0b', '#52514e', '#ffffff'

plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'figure.facecolor': SURFACE,
                     'savefig.facecolor': SURFACE, 'pdf.fonttype': 42})


def load_lms(path):
    try:
        pts = np.loadtxt(path)
        return pts if pts.shape == (110, 2) else None
    except (OSError, ValueError):
        return None


def face_looks_real(video_dir, idx, lms):
    """Reject fades, cutaways and landmark misfits: the crop must be lit, textured,
    face-shaped and inside the frame."""
    x0, y0 = lms.min(0)
    x1, y1 = lms.max(0)
    w, h = x1 - x0, y1 - y0
    if not (0.6 < w / max(h, 1) < 1.6) or x0 < 0 or y0 < 0 or x1 > 480 or y1 > 480:
        return False
    img = np.asarray(Image.open(os.path.join(video_dir, 'full_body_img', f'{idx}.jpg')).convert('L'),
                     dtype=np.float32)
    patch = img[int(y0):int(y1), int(x0):int(x1)]
    return patch.size > 0 and patch.mean() > 55 and patch.std() > 18


def best_face(video_dir, n_frames, samples=80):
    """Largest real face with the most nearly closed mouth, from the middle of the video."""
    best, best_score = None, -1e9
    for i in np.linspace(n_frames * 0.15, n_frames * 0.85, samples).astype(int):
        lms = load_lms(os.path.join(video_dir, 'landmarks', f'{i}.lms'))
        if lms is None:
            continue
        width = lms[:, 0].max() - lms[:, 0].min()
        if width < 40 or not face_looks_real(video_dir, i, lms):
            continue
        aperture = (lms[INNER_LOWER, 1].mean() - lms[INNER_UPPER, 1].mean()) / width
        score = width / 100 - 8 * max(aperture, 0)   # prefer big faces, closed mouths
        if score > best_score:
            best, best_score = (i, lms), score
    return best


def crop_face(video_dir, idx, lms, size=320):
    img = Image.open(os.path.join(video_dir, 'full_body_img', f'{idx}.jpg')).convert('RGB')
    x0, y0 = lms.min(0)
    x1, y1 = lms.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - 0.12 * (y1 - y0)   # landmarks sit low: lift to include the forehead
    half = 0.85 * max(x1 - x0, y1 - y0)
    box = (cx - half, cy - half, cx + half, cy + half)
    return img.crop(tuple(int(round(v)) for v in box)).resize((size, size), Image.LANCZOS)


def collect():
    manifest = json.load(open(os.path.join(CORPUS, 'manifest.json'), encoding='utf-8'))
    videos = {}
    for vid, info in manifest['videos'].items():
        videos.setdefault(info['speaker'], []).append((vid, info['frames']))
    faces = []
    for spk in sorted(videos):
        for vid, n in videos[spk]:   # fall back to the speaker's next video if needed
            vdir = os.path.join(CORPUS, 'processed', vid)
            found = best_face(vdir, n)
            if found is not None:
                faces.append((spk, crop_face(vdir, *found)))
                break
        else:
            print('no usable face for', spk)
    return faces, manifest


def draw(faces, manifest, mode):
    cols = 6
    rows = int(np.ceil(len(faces) / cols))
    poster = mode == 'poster'
    fs = 1.6 if poster else 1.0
    w = 10.0 if poster else 6.2
    footer = 0.95 if poster else 0.62   # room for the last row's labels plus the note
    fig = plt.figure(figsize=(w, w * rows / cols * 1.16 + (1.1 if poster else 0) + footer))
    top = 1 - (1.2 / fig.get_figheight() if poster else 0.02)
    bottom = footer / fig.get_figheight()
    gs = fig.add_gridspec(rows, cols, left=0.01, right=0.99, top=top, bottom=bottom,
                          wspace=0.06, hspace=0.22)
    for k, (spk, face) in enumerate(faces):
        ax = fig.add_subplot(gs[k // cols, k % cols])
        ax.imshow(face)
        ax.set_xticks([])
        ax.set_yticks([])
        test = spk in TEST_SPEAKERS
        for s in ax.spines.values():
            s.set_edgecolor(ACCENT if test else '#e1e0d9')
            s.set_linewidth(3.2 * fs if test else 0.8)
        ax.set_xlabel(f'Speaker {display_number(spk, [s for s, _ in faces])}', fontsize=8.5 * fs, color=ACCENT if test else INK2,
                      weight='bold' if test else 'normal', labelpad=3)
    note = (f'{len(faces)} speakers · {len(manifest["videos"])} videos · {manifest["hours"]:.2f} hours of Bangla speech.   '
            'Blue: the six speakers used as unseen test voices.')
    fig.text(0.5, 0.012, note, ha='center', va='bottom', fontsize=8.5 * fs, color=INK2)
    if poster:
        fig.text(0.01, 0.995, 'The Bangla audiovisual corpus', fontsize=15 * fs, weight='bold',
                 color=INK, va='top')
        fig.text(0.01, 0.995 - 0.62 / fig.get_figheight(), 'One face per speaker. The model never trained '
                 'on these voices; six of them test whether its lips follow a new Bangla speaker.',
                 fontsize=9.5 * fs, color=INK2, va='top')
    d = os.path.join(OUT, mode)
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, 'fig_corpus_speakers.png'), dpi=300, bbox_inches='tight', pad_inches=0.06)
    if mode == 'report':
        fig.savefig(os.path.join(d, 'fig_corpus_speakers.pdf'), bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)


if __name__ == '__main__':
    faces, manifest = collect()
    print(len(faces), 'faces')
    for mode in ('report', 'poster'):
        draw(faces, manifest, mode)
    print('written to', OUT)
