"""Pick a fixed HDTF test subset and freeze it as a manifest.

HDTF ships URLs and annotations, not media, so the subset has to be defined by
identity + timestamps + crop window and rebuilt by anyone who wants to check it.
Selection is deterministic: sorted, seeded, no randomness that a rerun could
change.

One identity per person. Names carry a trailing clip index (CarolynMaloney1,
CarolynMaloney2), so the base name is the identity - taking both would put the
same face in the test set twice and inflate any per-identity average.
"""
import json, os, re, sys
from collections import defaultdict

META = os.path.join(os.path.dirname(__file__), 'meta', 'HDTF_dataset')
SPLITS = ['RD', 'WDA', 'WRA']
N_WANTED = 15


def load(split, kind):
    p = os.path.join(META, f'{split}_{kind}.txt')
    out = {}
    for line in open(p, encoding='utf-8-sig'):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        out[parts[0].replace('.mp4', '')] = parts[1:]
    return out


def identity(name):
    """Strip the trailing clip index: CarolynMaloney2 -> CarolynMaloney."""
    return re.sub(r'\d+$', '', name)


def to_sec(ts):
    m, s = ts.split(':')
    return int(m) * 60 + int(s)


rows = []
for sp in SPLITS:
    urls = load(sp, 'video_url')
    times = load(sp, 'annotion_time')
    crops = load(sp, 'crop_wh')
    ratios = load(sp, 'crop_ratio')
    res = load(sp, 'resolution')
    for name, u in urls.items():
        if name not in times or f'{name}_0' not in crops:
            continue                      # annotations incomplete for this one
        rng = times[name][0]
        if '-' not in rng:
            continue
        a, b = rng.split('-')
        start, end = to_sec(a), to_sec(b)
        if end - start < 45:              # need 40s of talking head plus margin
            continue
        # HDTF order is x, out_w, y, out_h - NOT w h x y. Verified against
        # universome/HDTF download.py, which builds crop=out_w:out_h:x:y.
        # Misreading this silently produces off-frame crops that still encode.
        x, w, y, h = (int(v) for v in crops[f'{name}_0'][:4])
        rows.append(dict(
            split=sp, name=name, identity=identity(name), url=u[0],
            start=start, end=end,
            crop=dict(w=w, h=h, x=x, y=y),
            ratio=float(ratios.get(f'{name}_0', ['1.0'])[0]),
            resolution=int(res.get(name, ['720'])[0]),
        ))

# one clip per identity: prefer 1080p, then the longest usable range, then name
best = {}
for r in rows:
    k = r['identity']
    cur = best.get(k)
    key = (r['resolution'], r['end'] - r['start'], r['name'])
    if cur is None or key > (cur['resolution'], cur['end'] - cur['start'], cur['name']):
        best[k] = r

# spread across splits rather than taking whichever split sorts first
by_split = defaultdict(list)
for r in best.values():
    by_split[r['split']].append(r)
for sp in by_split:
    by_split[sp].sort(key=lambda r: (-r['resolution'], r['identity']))

picked, i = [], 0
while len(picked) < N_WANTED and any(by_split.values()):
    sp = SPLITS[i % len(SPLITS)]
    if by_split[sp]:
        picked.append(by_split[sp].pop(0))
    i += 1

print(f'{len(rows)} annotated clips -> {len(best)} unique identities -> picking {len(picked)}\n')
print(f'{"identity":28s} {"split":5s} {"res":>5s} {"range":>12s} {"crop wxh+x+y":>18s}')
for r in picked:
    c = r['crop']
    print(f'{r["identity"]:28s} {r["split"]:5s} {r["resolution"]:5d} '
          f'{r["start"]:5d}-{r["end"]:<6d} {c["w"]}x{c["h"]}+{c["x"]}+{c["y"]:>6}')

man = dict(
    dataset='HDTF', subset='fydp_test_v1', n=len(picked),
    clip_seconds=30.88, fps=25, out_resolution=512,
    note=('One clip per identity. Take 30.88s starting 5s into the annotated '
          'range so the window never touches the boundary. Crop window and '
          'ratio are HDTF originals; media is never redistributed.'),
    clips=picked)
out = os.path.join(os.path.dirname(__file__), 'hdtf_test_manifest.json')
json.dump(man, open(out, 'w'), indent=1)
print('\nwrote', out)
