# HDTF test subset — `fydp_test_v1`

A frozen 14-identity English test set, built to mirror the `redwan` Bangla test
clip exactly so the two datasets are directly comparable.

| | |
|---|---|
| Identities | **14** (15 selected, 1 unavailable) |
| Per clip | 30.88 s · 772 frames · 25 fps · 512×512 · 16 kHz mono PCM |
| Retrieved | **2026-09-17** |
| Source | HDTF (Zhang et al.), annotations CC BY 4.0 |

Frame count and duration match `redwan_test` deliberately — 772 frames at 25 fps
— so per-segment scoring and frame-wise metrics need no resampling between
datasets.

## Attrition

**1 of 15 unavailable: `AllenWest` — source video is now Private on YouTube.**

This is inherent to HDTF, which distributes URLs rather than media. The rate will
grow over time. **Report the retrieval date and the attrition count in any paper
using this subset** — anyone rebuilding it later will get fewer clips, and the
comparison is only meaningful if both numbers are stated.

The manifest still lists all 15. It records the *intended* subset;
`build_report.json` records what was actually obtained on the date above. No
substitute was selected — backfilling to a round number would hide the attrition
rather than report it.

### `Radio` is defective — do not use it

It rebuilt without error and must still be excluded. Its 4K source returned only
**10.84 s of audio against a 772-frame video**, so every system silently truncated
to about 268 frames. Nothing in the build reports this: the video is full length,
the file sizes look ordinary, and only a frame-count check against the audio
duration catches it.

The cause is the section fetch — `--download-sections` on that 4K source returned
a short audio stream. Any rebuild should verify **audio duration against frame
count**, not merely that both exist.

## The evaluation subset — 6 identities

The benchmark uses six of the fourteen, balanced 3 WDA / 3 WRA:

    AdamSchiff  AnnWagner  AdamSmith  AustinScott  AmyKlobuchar  CoryGardner

`CoryGardner` replaced `Radio`. Six was chosen to keep 6 systems × 6 identities ×
2 conditions = 72 generations inside the compute budget; the other eight clips are
built and available if a reviewer wants the subset widened.

**Articulation varies enormously across these speakers** — ground-truth
open-mouth fraction ranges from 0.089 (`AustinScott`) to 0.817 (`AmyKlobuchar`).
That is not noise to be averaged away: leakage scales with it (Spearman +0.642,
p=0.00002, n=36), so any leakage figure must be normalised per identity. See
`benchmark/README.md`.

**These clips are too short to train a person-specific model.** They are 30.88 s,
against the 247 s the redwan model trained on. The annotated ranges are longer —
50 s to 343 s depending on identity — so longer training clips could be cut, but
`AustinScott` has only 50 s in total and cannot support it at all.

## Rebuilding

```bash
python select_subset.py     # deterministic; regenerates the manifest
python download_prep.py     # fetches, crops, emits clips/  (skips existing)
```

`download_prep.py` pulls only the needed ~35 s section via `yt-dlp
--download-sections`, not whole videos, so the whole set is a few minutes of
bandwidth rather than hundreds of gigabytes.

## Selection rule

Deterministic, no randomness. One clip per **identity** — HDTF names carry a
trailing clip index (`CarolynMaloney1`, `CarolynMaloney2`), so the base name is
the identity; taking both would put the same face in the test set twice and
inflate any per-identity average. Preference order: 1080p or better, then the
longest usable annotated range, then name. Picks are drawn round-robin across
the three HDTF splits (RD / WDA / WRA) rather than taking whichever sorts first.

Only ranges ≥45 s are eligible, so a 30.88 s clip taken 5 s into the range never
touches the boundary.

## The crop format trap

`xx_crop_wh.txt` is **`x, out_w, y, out_h`** — *not* `w h x y`.

```
AdamSchiff_0.mp4 608 726 0 726    ->  ffmpeg crop=726:726:608:0
```

Read as `w h x y` it still produces a valid-looking command and still encodes
successfully — it just crops the wrong pixels, usually off-frame. Verified
against `universome/HDTF/download.py`, which builds `crop={out_w}:{out_h}:{x}:{y}`.
Every crop in this subset was checked square and inside the source frame before
downloading.

`download_prep.py` also caps yt-dlp's format to the resolution HDTF measured
against (`-f "bv*[height<=N]"`). Without that cap yt-dlp may return 4K or 480p,
and the published crop window would address the wrong pixels silently.

## Licence and redistribution

HDTF's **annotations** are CC BY 4.0. The **videos** remain under their original
YouTube terms and are *not* redistributed here — same model as LRS3 and
VoxCeleb. Ship the manifest and these scripts; never the media.

Cite HDTF: Zhang et al., *Flow-guided One-shot Talking Face Generation with a
High-resolution Audio-visual Dataset*, CVPR 2021.

## Contents

```
select_subset.py         deterministic selection -> manifest
download_prep.py         rebuild from the manifest
hdtf_test_manifest.json  the 15 intended clips: url, range, crop, ratio, res
build_report.json        what was obtained on 2026-09-17, plus failures
clips/                   14 x (mp4 + wav)   ~107 MB   NOT redistributable
meta/                    upstream HDTF annotation repo
```
