"""Benchmark figures for the FYDP report and poster.

    python benchmark/make_figures.py

Writes two versions of every figure:
  docs/fydp/figures/report/  vector PDF + 300 dpi PNG, no title (the LaTeX
                             caption carries it), sized for a 15 cm text width
  docs/fydp/figures/poster/  300 dpi PNG with title and takeaway, large type

Every number is copied from docs/fydp/benchmark_result.md, which records where
each one comes from. Colour is emphasis, not identity: Alapon is the accent,
the model before the fix (or "above real video") is the counterpoint, and
everything else is recessive grey. The two colours are the first two slots of
a CVD-validated categorical palette.
"""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'fydp', 'figures')

# ---- palette (light surface) --------------------------------------------------
SURFACE = '#ffffff'
INK = '#0b0b0b'          # primary text
INK2 = '#52514e'         # secondary text
MUTED = '#898781'        # axis labels
GRID = '#e1e0d9'         # hairline grid
AXIS = '#c3c2b7'         # baseline
ACCENT = '#2a78d6'       # Alapon
COUNTER = '#eb6834'      # before fix / above real video
REST = '#c9c7bf'         # everything else
DATASET_BN = '#4a3aa7'   # dataset identity (figure 7 only)
DATASET_HD = '#1baf7a'

plt.rcParams.update({
    'font.family': ['Segoe UI', 'Nirmala UI', 'DejaVu Sans'],  # Nirmala: Bangla fallback
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.linewidth': 0.8,
    'xtick.color': MUTED, 'ytick.color': INK, 'xtick.major.size': 0, 'ytick.major.size': 0,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'pdf.fonttype': 42,
})

# ---- data (docs/fydp/benchmark_result.md) -------------------------------------
BN_SYSTEMS = ['Alapon (ours)', 'SyncTalk_2D, before fix', 'Wav2Lip (GAN)', 'Wav2Lip (non-GAN)',
              'LatentSync 1.5', 'MuseTalk 1.5', 'MuseTalk 1.0', 'IP-LAP']
BN_SILENCE = [0.1, 41, 46, 0.8, 36, 0.4, 0.1, 20]            # % frames, report Table 4.5
BN_LSEC = [5.11, 5.16, 6.08, 6.44, 5.09, 4.90, 4.67, 4.36]
BN_REAL_LSEC = 5.14
HD_SYSTEMS = ['Wav2Lip (GAN)', 'Wav2Lip (non-GAN)', 'LatentSync 1.5', 'MuseTalk 1.5',
              'MuseTalk 1.0', 'IP-LAP']
HD_SILENCE = [24, 8, 24, 46, 41, 10]                            # report Table 4.6
HD_LSEC = [8.75, 8.94, 8.82, 8.38, 7.52, 7.88]
HD_REAL_LSEC = 8.01
PH_SYSTEMS = BN_SYSTEMS
PH_RATIO = [3.8, 3.0, 3.0, 3.0, 3.1, 2.8, 3.5, 1.9]             # report Table 4.3; real 4.0
PH_REAL = 4.0
# Display numbers (1..24 in order); file IDs speaker23/24 show as 22/23, see make_corpus_grid.py
SPEAKERS = ['Speaker 02', 'Speaker 04', 'Speaker 08', 'Speaker 11', 'Speaker 22', 'Speaker 23']
SPK_ALAPON = [2.55, 1.96, 2.01, 1.88, 3.18, 2.76]                # report Table 4.4
SPK_BEFORE = [1.99, 1.49, 1.82, 1.63, 1.76, 1.59]
JAW_ROWS = ['10\n(original)', '8', '6', '4', '2', '0\n(covered)']
JAW_SILENCE = [39, 36, 26, 33, 36, 8]                            # report Table 4.2
JAW_LSEC = [5.07, 5.09, 4.88, 4.98, 5.14, 4.93]
SIZE_SYSTEMS = ['Alapon (ours)', 'Wav2Lip', 'IP-LAP', 'MuseTalk 1.5', 'LatentSync 1.5']
SIZE_MB = [49, 436, 475, 3400, 5072]                             # checkpoint files

# ---- output modes --------------------------------------------------------------
MODES = {
    # name: (width in inches, font scale, show title)
    'report': (6.2, 1.0, False),
    'poster': (10.0, 1.6, True),
}


def colour_for(name):
    if name.startswith('Alapon'):
        return ACCENT
    if 'before fix' in name:
        return COUNTER
    return REST


def px(ax):
    """Data units per pixel along x and y, for pixel-exact mark geometry."""
    ax.figure.canvas.draw()
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    return 1 / abs(x1 - x0), 1 / abs(y1 - y0)


def hbar(ax, y, v, h, colour, r_px=4):
    """Horizontal bar from 0 to v: square at the baseline, 4 px rounded data end."""
    ux, uy = px(ax)
    rx, ry = r_px * ux, r_px * uy
    sign = 1 if v >= 0 else -1
    mag = abs(v)
    if mag <= 2 * rx:  # too short to round: a plain sliver
        ax.add_patch(Rectangle((min(0, v), y - h / 2), mag, h, color=colour, lw=0))
        return
    ax.add_patch(Rectangle((min(0, v - sign * rx), y - h / 2), mag - rx, h, color=colour, lw=0))
    left = v - 2 * rx if sign > 0 else v
    ax.add_patch(FancyBboxPatch((left, y - h / 2), 2 * rx, h, color=colour, lw=0,
                                boxstyle=f'round,pad=0,rounding_size={rx}',
                                mutation_aspect=ry / rx))


def vbar(ax, x, v, w, colour, r_px=4):
    """Vertical column from 0 to v: square at the baseline, 4 px rounded cap."""
    ux, uy = px(ax)
    rx, ry = r_px * ux, r_px * uy
    if v <= 2 * ry:
        ax.add_patch(Rectangle((x - w / 2, 0), w, v, color=colour, lw=0))
        return
    ax.add_patch(Rectangle((x - w / 2, 0), w, v - ry, color=colour, lw=0))
    ax.add_patch(FancyBboxPatch((x - w / 2, v - 2 * ry), w, 2 * ry, color=colour, lw=0,
                                boxstyle=f'round,pad=0,rounding_size={rx}',
                                mutation_aspect=ry / rx))


def bar_thickness(ax, n_px=22):
    """Band thickness in data units, capped at n_px (never fill the slot)."""
    _, uy = px(ax)
    return min(0.62, n_px * uy)


def frame(ax, fs, grid_axis='x'):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9 * fs)
    if grid_axis == 'x':
        ax.spines['left'].set_visible(False)


def bold_label(ax, prefix, axis='y'):
    labels = ax.get_yticklabels() if axis == 'y' else ax.get_xticklabels()
    for lbl in labels:
        if lbl.get_text().startswith(prefix):
            lbl.set_fontweight('semibold')


def titles(fig, fs, show, title, sub):
    if not show:
        return
    fig.suptitle(title, x=0.012, y=0.995, ha='left', va='top', fontsize=15 * fs, color=INK,
                 weight='semibold')
    fig.text(0.012, 0.995 - 0.075, sub, fontsize=10 * fs, color=INK2, va='top', ha='left')


def legend(ax, fs, items, loc='lower right'):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in items]
    ax.legend(handles, [t for _, t in items], loc=loc, frameon=False, fontsize=8.5 * fs,
              handlelength=0.9, handleheight=0.9, labelcolor=INK2)


def legend_row(fig, fs, items, y=0.0):
    """One-row legend below the plot, for charts with no free corner."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in items]
    fig.legend(handles, [t for _, t in items], loc='lower center', ncol=len(items),
               bbox_to_anchor=(0.5, y), frameon=False, fontsize=8.5 * fs, handlelength=0.9,
               handleheight=0.9, labelcolor=INK2, columnspacing=1.8)


def save(fig, mode, name):
    d = os.path.join(OUT, mode)
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, name + '.png'), dpi=300, bbox_inches='tight', pad_inches=0.08)
    if mode == 'report':
        fig.savefig(os.path.join(d, name + '.pdf'), bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


def layout(mode, height_ratio, extra=1.1):
    """Figure size and the subplot top edge, leaving room for the poster title."""
    w, fs, show = MODES[mode]
    h = height_ratio * w + (extra if show else 0)
    top = 1 - (extra + 0.1) / h if show else 0.97
    return w, h, fs, show, top


# ---- 1. silence leak on the Bangla clip ----------------------------------------
def fig_silence(mode):
    w, h, fs, show, top = layout(mode, 0.42)
    order = sorted(range(len(BN_SYSTEMS)), key=lambda i: BN_SILENCE[i])
    names = [BN_SYSTEMS[i] for i in order]
    vals = [BN_SILENCE[i] for i in order]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.30, right=0.93, top=top, bottom=0.14)
    ax.set_xlim(0, 52)
    ax.set_ylim(-0.6, len(names) - 0.4)
    th = bar_thickness(ax)
    for i, (n, v) in enumerate(zip(names, vals)):
        hbar(ax, i, v, th, colour_for(n))
        ax.text(v + 0.8, i, f'{v:g}%', va='center', fontsize=9 * fs,
                color=INK if n.startswith('Alapon') or 'before' in n else INK2,
                weight='semibold' if n.startswith('Alapon') else 'normal')
    ax.set_yticks(range(len(names)), names)
    bold_label(ax, 'Alapon')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.set_xlabel('Frames with the mouth open while the audio is silent', fontsize=9 * fs)
    frame(ax, fs)
    legend(ax, fs, [(ACCENT, 'Alapon (ours)'), (COUNTER, 'Same model before the fix'),
                    (REST, 'Published systems')])
    titles(fig, fs, show, 'The mouth should stay still when nobody is speaking',
           'Bangla test clip. Before the fix our model copied the jaw: 41% → 0.1%.')
    save(fig, mode, 'fig_silence_bangla')


# ---- 2. lip-sync score against real video --------------------------------------
def fig_above_real(mode):
    w, h, fs, show, top = layout(mode, 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(w, h), sharex=True)
    fig.subplots_adjust(left=0.21, right=0.98, wspace=0.75, top=top - 0.06, bottom=0.27)
    panels = [(axes[0], 'Bangla test clip', BN_SYSTEMS, BN_LSEC, BN_REAL_LSEC),
              (axes[1], 'HDTF, 6 English speakers', HD_SYSTEMS, HD_LSEC, HD_REAL_LSEC)]
    for ax, name, systems, vals, real in panels:
        diffs = [v - real for v in vals]
        order = sorted(range(len(systems)), key=lambda i: diffs[i])
        ax.set_xlim(-1.45, 1.7)
        ax.set_ylim(-0.6, len(systems) - 0.4)
        th = bar_thickness(ax, 16)
        for row, i in enumerate(order):
            d, s = diffs[i], systems[i]
            c = ACCENT if s.startswith('Alapon') else (COUNTER if d > 0.04 else REST)
            hbar(ax, row, d, th, c)
            ax.text(d + (0.05 if d >= 0 else -0.05), row, f'{d:+.2f}', va='center',
                    ha='left' if d >= 0 else 'right', fontsize=8 * fs, color=INK2)
        ax.set_yticks(range(len(systems)), [systems[i] for i in order])
        bold_label(ax, 'Alapon')
        ax.tick_params(axis='y', labelsize=8.5 * fs)
        ax.tick_params(axis='x', labelsize=8 * fs)
        ax.axvline(0, color=INK, lw=1.2)
        ax.grid(axis='x', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
        ax.set_title(f'{name}  (real video = {real:.2f})', fontsize=9 * fs, color=INK, loc='left',
                     pad=6)
    fig.supxlabel("LSE-C minus the real video's own score.  Right of the line = rated above a real person.",
                  fontsize=8.5 * fs, color=INK2, y=0.1)
    legend_row(fig, fs, [(COUNTER, 'Above real video'), (ACCENT, 'Alapon (ours)'),
                         (REST, 'At or below real')])
    titles(fig, fs, show, 'The standard lip-sync score rates some fakes above real people',
           'A real speaker is perfectly in sync with their own voice, so no system should cross the line.')
    save(fig, mode, 'fig_lsec_vs_real')


# ---- 3. jaw rows: leak falls, lip-sync score does not move -----------------------
def fig_jaw(mode):
    w, h, fs, show, top = layout(mode, 0.62)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(w, h), sharex=True,
                                 gridspec_kw={'height_ratios': [1.6, 1]})
    fig.subplots_adjust(left=0.13, right=0.97, hspace=0.3, top=top, bottom=0.14)
    xs = list(range(len(JAW_ROWS)))
    a1.set_ylim(0, 46)
    a1.set_xlim(-0.6, len(JAW_ROWS) - 0.4)
    ux, _ = px(a1)
    wbar = min(0.55, 34 * ux)
    for x, v in zip(xs, JAW_SILENCE):
        c = ACCENT if x == len(JAW_ROWS) - 1 else REST
        vbar(a1, x, v, wbar, c)
        a1.text(x, v + 1.2, f'{v}%', ha='center', fontsize=9 * fs,
                color=INK if c == ACCENT else INK2, weight='semibold' if c == ACCENT else 'normal')
    a1.set_ylabel('Mouth moving\nduring silence', fontsize=8.5 * fs)
    a1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    frame(a1, fs, 'y')
    a2.plot(xs, JAW_LSEC, color=INK2, lw=2, solid_capstyle='round', zorder=2)
    a2.scatter(xs, JAW_LSEC, s=40 * fs, color=INK2, edgecolor=SURFACE, lw=2, zorder=3)
    a2.set_ylim(4.0, 6.0)
    a2.set_ylabel('LSE-C', fontsize=8.5 * fs)
    a2.text(len(JAW_ROWS) - 0.45, 5.55, 'barely moves: 4.88 – 5.14', ha='right',
            fontsize=8.5 * fs, color=INK2)
    frame(a2, fs, 'y')
    a2.set_xticks(xs, JAW_ROWS)
    a2.tick_params(axis='x', labelsize=9 * fs, colors=INK)
    a2.set_xlabel('Rows of the jaw left visible to the model', fontsize=9 * fs)
    titles(fig, fs, show, 'Only hiding the whole jaw stops the copying',
           'Six identical models; only the visible jaw differs. The lip-sync score cannot tell them apart.')
    save(fig, mode, 'fig_jaw_rows')


# ---- 4. Bangla sounds: open / closed contrast -------------------------------------
def fig_phoneme(mode):
    w, h, fs, show, top = layout(mode, 0.44)
    rows = sorted(zip(PH_SYSTEMS, PH_RATIO), key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.30, right=0.90, top=top, bottom=0.25)
    ax.set_xlim(0, 4.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    th = bar_thickness(ax)
    for i, (n, v) in enumerate(rows):
        hbar(ax, i, v, th, colour_for(n))
        if v + 0.4 > PH_REAL:  # label would cross the reference line: set it inside the bar
            ax.text(v - 0.07, i, f'{v:.1f}×', va='center', ha='right', fontsize=9 * fs,
                    color=SURFACE, weight='semibold')
        else:
            ax.text(v + 0.05, i, f'{v:.1f}×', va='center', fontsize=9 * fs, color=INK2)
    ax.axvline(PH_REAL, color=INK, lw=1.2)
    ax.text(PH_REAL + 0.04, -0.45, 'real speaker\n4.0×', fontsize=8.5 * fs, color=INK, va='bottom')
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    bold_label(ax, 'Alapon')
    ax.set_xlabel('Mouth opening on open sounds (আ অ) ÷ on closed sounds (প ফ ব ভ ম)', fontsize=9 * fs)
    frame(ax, fs)
    legend_row(fig, fs, [(ACCENT, 'Alapon (ours)'), (COUNTER, 'Same model before the fix'),
                         (REST, 'Published systems')])
    titles(fig, fs, show, 'Lips that close on প ব ম and open on আ, like a real speaker',
           'Bangla test clip. Closer to the real speaker is better.')
    save(fig, mode, 'fig_bangla_sounds')


# ---- 5. six new Bangla voices: before -> after dumbbell ---------------------------
def fig_speakers(mode):
    w, h, fs, show, top = layout(mode, 0.46)
    labels = SPEAKERS + ['All six']
    before = SPK_BEFORE + [1.7]
    after = SPK_ALAPON + [2.3]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.17, right=0.95, top=top, bottom=0.14)
    ys = list(range(len(labels)))[::-1]
    ax.set_xlim(1.2, 3.5)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    for y, b, a, lab in zip(ys, before, after, labels):
        total = lab == 'All six'
        ax.plot([b, a], [y, y], color=AXIS if total else GRID, lw=4 if total else 3,
                solid_capstyle='round', zorder=1)
        size = (110 if total else 70) * fs
        ax.scatter([b], [y], s=size, color=COUNTER, edgecolor=SURFACE, lw=2, zorder=3)
        ax.scatter([a], [y], s=size, color=ACCENT, edgecolor=SURFACE, lw=2, zorder=3)
        fmt = '{:.1f}×' if total else '{:.2f}×'
        ax.text(a + 0.07, y, fmt.format(a), va='center', fontsize=9 * fs, color=INK,
                weight='semibold' if total else 'normal')
        ax.text(b - 0.07, y, fmt.format(b), va='center', ha='right', fontsize=9 * fs, color=INK2,
                weight='semibold' if total else 'normal')
    ax.axhline(0.5, color=AXIS, lw=0.8)
    ax.set_yticks(ys, labels)
    bold_label(ax, 'All six')
    ax.set_xlabel('Open ÷ closed Bangla sounds (higher = the mouth follows the voice more clearly)',
                  fontsize=9 * fs)
    frame(ax, fs)
    handles = [plt.Line2D([], [], marker='o', ls='', ms=8 * fs ** 0.5, color=c) for c in (COUNTER, ACCENT)]
    ax.legend(handles, ['Before the fix', 'Alapon (ours)'], loc='upper right', frameon=False,
              fontsize=8.5 * fs, labelcolor=INK2)
    titles(fig, fs, show, 'Better on every Bangla voice it had never heard',
           'Our speaker\'s face, six other speakers\' audio: the mouth can only follow the voice.')
    save(fig, mode, 'fig_six_speakers')


# ---- 6. model size ------------------------------------------------------------------
def fig_size(mode):
    w, h, fs, show, top = layout(mode, 0.34)
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.22, right=0.93, top=top, bottom=0.22)
    names = SIZE_SYSTEMS[::-1]
    vals = SIZE_MB[::-1]
    ax.set_xlim(0, 6000)
    ax.set_ylim(-0.6, len(names) - 0.4)
    th = bar_thickness(ax)
    for i, (n, v) in enumerate(zip(names, vals)):
        hbar(ax, i, v, th, ACCENT if n.startswith('Alapon') else REST)
        star = '*' if n == 'Wav2Lip' else ''
        ax.text(v + 60, i, f'{v:,} MB{star}', va='center', fontsize=9 * fs,
                color=INK if n.startswith('Alapon') else INK2,
                weight='semibold' if n.startswith('Alapon') else 'normal')
    ax.set_yticks(range(len(names)), names)
    bold_label(ax, 'Alapon')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set_xlabel('Checkpoint file (MB).   * includes training optimiser state', fontsize=9 * fs)
    frame(ax, fs)
    titles(fig, fs, show, 'Small enough for a 4 GB laptop GPU',
           'Alapon renders at 30 fps on an RTX 3050 using 0.3 GB of GPU memory.')
    save(fig, mode, 'fig_model_size')


# ---- 7. leakage is not a fixed property of a system -----------------------------------
def fig_two_datasets(mode):
    w, h, fs, show, top = layout(mode, 0.42)
    bn = dict(zip(BN_SYSTEMS, BN_SILENCE))
    hd = dict(zip(HD_SYSTEMS, HD_SILENCE))
    names = sorted(HD_SYSTEMS, key=lambda s: abs(hd[s] - bn[s]))
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.22, right=0.95, top=top, bottom=0.14)
    ax.set_xlim(-2, 52)
    ax.set_ylim(-0.6, len(names) - 0.4)
    for y, n in enumerate(names):
        b, d = bn[n], hd[n]
        ax.plot([b, d], [y, y], color=GRID, lw=3, solid_capstyle='round', zorder=1)
        ax.scatter([b], [y], s=70 * fs, color=DATASET_BN, edgecolor=SURFACE, lw=2, zorder=3, marker='o')
        ax.scatter([d], [y], s=70 * fs, color=DATASET_HD, edgecolor=SURFACE, lw=2, zorder=3, marker='s')
    ax.set_yticks(range(len(names)), names)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.set_xlabel('Frames with the mouth open while the audio is silent', fontsize=9 * fs)
    frame(ax, fs)
    handles = [plt.Line2D([], [], marker=m, ls='', ms=8 * fs ** 0.5, color=c)
               for m, c in (('o', DATASET_BN), ('s', DATASET_HD))]
    ax.legend(handles, ['Bangla clip (1 speaker)', 'HDTF (6 English speakers)'], loc='lower right',
              frameon=False, fontsize=8.5 * fs, labelcolor=INK2)
    titles(fig, fs, show, 'The same system leaks differently on different speakers',
           'MuseTalk 1.5: 0.4% on our Bangla clip, 46% on HDTF. One clip is not enough to judge a system.')
    save(fig, mode, 'fig_leak_two_datasets')


FIGS = [fig_silence, fig_above_real, fig_jaw, fig_phoneme, fig_speakers, fig_size, fig_two_datasets]

if __name__ == '__main__':
    for mode in MODES:
        for f in FIGS:
            f(mode)
    for mode in MODES:
        print(mode, sorted(os.listdir(os.path.join(OUT, mode))))
