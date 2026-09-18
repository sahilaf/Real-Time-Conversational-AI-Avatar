# Research Proposal — Evaluation Integrity in Audio-Driven Talking-Head Generation

**Author:** Sahil al Farib
**Date:** 13 September 2026 · **revised 18 September 2026**
**Target venue:** IEEE Transactions on Multimedia (Q1) · fallback: IEEE TCSVT, Pattern Recognition

> **Revision note.** The original thesis was that lip-sync score and leakage rank
> systems *inversely* — that the metric **rewards** leakage. We designed the
> experiment that could falsify that, ran it, and **it did** (§4.4). The claim is
> withdrawn. What replaced it is narrower, better supported, and arrives with two
> findings the original proposal did not anticipate (§4.5, §4.6).

---

## 1. Thesis

> A widely-deployed talking-head baseline is **not audio-driven at all** — and the
> field's standard metrics rank it as working. We trace the defect to a specific
> masking error and repair it. We then show that the standard protocol could not
> have caught it, for three independent reasons: it scores generated video **above
> real human footage**, its leakage counter-measure **measures the test clip rather
> than the system**, and both are unstable at the clip lengths everyone uses.

The contribution is not the bug fix. It is that a defect this severe survived in a
widely-used model because the evaluation protocol cannot see it — and that the
protocol's failure modes are specific, measurable and fixable.

---

## 2. How we found it

We set out to deploy a lightweight talking-head model for Bangla. The avatar moved
its mouth during silence.

The model receives the current video frame with the mouth blacked out, plus audio,
and must paint the mouth back in. The masking code used OpenCV's `(x, y, w, h)`
rectangle form where the point form was intended, so it blacked out rows 5–310 of a
**320**-row crop. Rows 310–319 — the chin and jaw — stayed visible in training and
inference.

Jaw drop predicts mouth opening almost perfectly. Holding audio and appearance
reference constant, that ten-pixel strip accounted for **91% of all mouth motion**.
The model had learned to read the chin rather than listen.

Its lip-sync expert could not object: it was trained with `y = torch.ones(1)`
unconditionally — positive pairs only, no negatives anywhere in the file — so it
approved of everything and supplied no gradient.

**All four defects are confirmed present upstream at `ZiqiaoPeng/SyncTalk_2D @ 9e82dcc`.**

| Defect | Location |
|---|---|
| SyncNet trained on positives only | `syncnet_328.py:106` |
| Mask leaves the jaw exposed | `datasetsss_328.py:98`, `inference_328.py:115` |
| Sync loss weighted 10× | `train_328.py:109` |
| Train/inference reference mismatch | `datasetsss_328.py:135` vs `inference_328.py:112-121` |

---

## 3. The repair, measured — ⚠ PENDING RE-MEASUREMENT

| | as-released | repaired |
|---|---:|---:|
| PSNR | 29.830 | 33.114 |
| SSIM | 0.8824 | 0.9114 |
| MAE | 0.0206 | 0.0144 |
| Leakage (mouth open on silence) | 0.422 | **0.000** |

**These reconstruction numbers are not yet publishable.** On 2026-09-18 we found
that `MyDataset` enumerated every frame in the dataset directory and `train_328.py`
passed it through unfiltered, so both models trained on frames 0–7713 — including
the test split 6942–7713 on which the table above is measured. The appearance
reference was drawn from the whole video as well.

Fixed in `datasetsss_328.py` / `train_328.py`; training now takes `--manifest` and
records the frame range in `train_config.json`. Matched 100-epoch reruns on the
train split cost 14.6 h and are scheduled before submission.

**The leakage row is unaffected.** It does not depend on which frames were trained
on: the repaired mask removes the jaw pixels from the model's input, and §4.4 shows
causally that those pixels are the mechanism. A model cannot copy what it cannot see.

A control run isolates that mechanism. Re-running the as-released model with a fixed
appearance reference instead of the shipped per-frame reference moves leakage from
0.427 to 0.422 — 1%. **The mask is the entire mechanism; the reference contributes
nothing.**

---

## 4. The field-level findings

Two datasets, one scorer, ground-truth anchored on both. The scorer is
`benchmark/` in the repository; it reproduces the September figures to within 0.04
and is stable to 0.004 across three different GPUs.

### 4.1 HDTF — English, public, 6 identities

Means over 6 identities. Ground truth is a scored row, not an assumption.

| System | LSE-D ↓ | LSE-C ↑ | leak raw | **leak norm** ↓ | artic ratio |
|---|---:|---:|---:|---:|---:|
| **Ground truth (real video)** | 7.416 | **8.009** | — | — | 1.000 |
| wav2lip | **6.694** | **8.940** | 0.080 | **0.001** | 0.977 |
| latentsync 1.5 | 6.697 | 8.816 | 0.240 | 0.082 | 0.643 |
| wav2lip_gan | 6.929 | 8.752 | 0.241 | 0.299 | 1.034 |
| musetalk v1.5 | 7.120 | 8.384 | 0.456 | **0.934** | 0.994 |
| ip_lap | 7.067 | 7.883 | 0.097 | 0.003 | 0.339 |
| musetalk v1.0 | 7.722 | 7.524 | 0.413 | 0.608 | 0.750 |

### 4.2 redwan — Bangla, 1 clip

Rescored with the same code. SyncTalk rows carry the §3 contamination caveat.

| System | LSE-D ↓ | LSE-C ↑ | leak raw | leak norm ↓ | artic ratio |
|---|---:|---:|---:|---:|---:|
| **Ground truth** | 7.392 | 5.135 | — | — | 1.000 |
| wav2lip | 6.832 | **6.444** | — | — | 1.050 |
| wav2lip_gan | 7.262 | 6.077 | 0.458 | 0.906 | 1.207 |
| SyncTalk_2D legacy ⚠ | 7.317 | 5.164 | 0.409 | 0.809 | 1.045 |
| SyncTalk_2D final_v2 ⚠ | 7.300 | 5.111 | **0.001** | **0.003** | 0.942 |
| latentsync 1.5 | 7.288 | 5.088 | 0.359 | 0.710 | 0.790 |
| musetalk v1.5 | 7.863 | 4.899 | 0.004 | 0.008 | 0.533 |
| musetalk v1.0 | 7.720 | 4.665 | — | — | 0.221 |
| ip_lap | 8.712 | 4.361 | — | — | 0.763 |

**LSE values are not comparable across datasets** — real video scores 8.009 on
HDTF and 5.135 on redwan. Only within-dataset ranks are meaningful, which is
exactly why every table needs its own anchor row.

### 4.3 Generated video outscores real video, significantly

Paired over the 6 HDTF identities:

| System | ΔLSE-C vs real | paired t | Wilcoxon |
|---|---:|---:|---:|
| wav2lip | **+0.931** | **0.004** | **0.031** |
| latentsync | **+0.807** | **0.020** | **0.031** |
| wav2lip_gan | **+0.742** | **0.007** | **0.031** |
| musetalk v1.5 | +0.375 | 0.232 | 0.312 |
| ip_lap | −0.126 | 0.666 | 0.562 |
| musetalk v1.0 | −0.485 | 0.187 | 0.312 |

Three systems score above genuine human footage on the field's standard lip-sync
metric, on both parametric and rank tests. Five of six beat it on LSE-D. Real video
is the ceiling by construction; a metric that ranks synthesis above it is not
measuring what its name claims.

Wav2Lip and LatentSync both train with SyncNet supervision. They are optimising the
scorer, and the scorer rewards them for it.

**Most published tables omit the real-video row.** Without it, 8.94 against 8.75
reads as a fair contest rather than two systems both beating reality.

### 4.4 The falsified claim — mask dose-response

Six arms, one architecture, one dataset, matched epochs, varying **only** how many
jaw rows the mask leaves visible. This was designed to test causally whether the
metric rewards leakage.

| arm | jaw rows | train L1 ↓ | leak | LSE-C |
|---|---:|---:|---:|---:|
| jaw10 (= the shipped bug) | 10 | 0.0178 | 0.389 | 5.073 |
| jaw8 | 8 | 0.0179 | 0.355 | 5.093 |
| jaw6 | 6 | 0.0180 | 0.259 | 4.883 |
| jaw4 | 4 | 0.0185 | 0.334 | 4.982 |
| jaw2 | 2 | 0.0188 | 0.358 | 5.137 |
| **jaw0 (repaired)** | 0 | 0.0202 | **0.080** | 4.925 |

Three results:

- **Training L1 rises monotonically** as jaw pixels are removed (0.0178 → 0.0202).
  The model genuinely exploited those pixels; reconstruction degrades without them.
- **Leak is a step, not a gradient.** Only full closure collapses it. Partial
  repairs do nothing.
- **LSE-C is flat** — 4.88 to 5.14, no trend, Spearman with jaw rows 0.143.

The predicted monotonic rise did not occur. **The "sync metrics reward leakage"
claim is withdrawn.** On HDTF at n=6 the rank correlation between LSE-C and leak is
−0.543, p=0.266 — underpowered, indistinguishable from zero, and of the *opposite
sign* to the original hypothesis. It does not return with more systems either.

What survives is weaker and true: **the metric's sensitivity to leakage is too small
to matter.** The cleanest and dirtiest systems differ by 0.556 (p=0.030), while the
dirtiest — reproducing 93% of source mouth motion under digital silence — still
outscores real video. The separation between listening and not-listening is smaller
than the margin by which not-listening beats reality.

### 4.5 New finding: leakage measures the test clip, not the system

LipLeak feeds digital silence and counts mouth movement. It is a good idea with a
confound we did not anticipate: **how much leakage you detect depends on how much
the source speaker moves their own mouth.**

Spearman between ground-truth articulation and measured leak, per system:

    wav2lip_gan +0.94   latentsync +0.94   wav2lip +0.82
    ip_lap      +0.79   musetalk_v15 +0.60  musetalk_v1 +0.60
    POOLED (n=36): +0.642,  p = 0.00002

Mechanically obvious in hindsight — leakage is copying the input mouth, so a speaker
who barely opens theirs leaves nothing to copy. `AustinScott`'s ground-truth
articulation is 0.089 and nearly every system scores ~0 leakage on him.

**Consequence: raw LipLeak is not comparable across datasets, and a system evaluated
on low-articulation speakers looks clean regardless of behaviour.** Dividing by
source articulation fixes it. We propose reporting normalised leak, with raw
alongside, and we report both here.

### 4.6 New finding: single-clip leakage is uninformative

Normalised leak, same checkpoint, two datasets:

| System | redwan (1 clip) | HDTF (6 identities) |
|---|---:|---:|
| musetalk v1.5 | **0.008** | **0.934** |
| latentsync | 0.710 | 0.082 |
| wav2lip_gan | 0.906 | 0.299 |

The ranking inverts completely. MuseTalk goes from cleanest to dirtiest.

The honest reading is not that the datasets disagree. It is that **redwan is one
clip**, and on HDTF the per-identity leak spread routinely exceeds its own mean. A
leakage number from a single clip carries almost no information about a system —
which also means the leakage column in our own September table, and in KeySync's,
needs error bars before it can be read.

---

## 5. Supporting evidence that the protocol is unreliable

- **LSE-C is unstable at clip length.** Four consecutive 7.7-second windows of the
  *same* video from the *same* model scored 6.484, 5.659, 4.828, **2.167**.
- **Leakage is unstable across identities** (§4.6), often with a standard deviation
  exceeding its own mean.
- **The reference implementation is inconsistent.** The script the field uses to
  compute LSE contains `cv2.resize(img_input, (224,224))` with the literal comment
  `#HARD CODED, CHANGE BEFORE RELEASE`; the original `syncnet_python` applies no
  such resize. The two disagree and published numbers do not state which was used.
- **Audio muxing silently corrupts offsets.** AAC priming introduces a 2-frame
  (80 ms) apparent AV offset. And re-containering a file that already has audio can
  move LSE-C by ~1.0 — we measured ground truth shifting 5.135 → 6.109 with
  byte-identical audio, because several AVIs declare `avg_frame_rate 50/1` over 772
  real frames and SyncNet re-extracts at `-r 25`.
- **LSE cannot validate a harness.** SyncNet reads the score at its best offset, so
  it is insensitive to alignment error by design — the correct choice for tolerating
  a fixed pipeline delay, and the reason it cannot detect one. Our own 2-frame
  LatentSync misalignment moved LSE-C by 0.05, roughly 1%, while a geometric check
  found it immediately.
- **Most papers report no ground-truth anchor** (§4.3).
- **The standard expert under-scores non-English.** Real Bangla video scores LSE-C
  5.135 where real English measures 8.009 on the same scorer.

---

## 6. Proposed contributions

1. **A defect audit** of a widely-used baseline, with file:line evidence and a
   quantified causal mechanism (91% of mouth motion from a ten-pixel strip).
2. **Evidence that the standard metric ranks synthesis above reality** — three
   systems, significant, on a public benchmark, with the anchor row that makes it
   visible.
3. **A confound in the leakage metric, and its correction** (§4.5). LipLeak as
   published measures the test clip as much as the system; normalising by source
   articulation is a one-line fix with a strong effect.
4. **A paired protocol** (normalised Leak + Articulation) constructed so degenerate
   solutions cannot pass: a frozen mouth fails articulation, a copying model fails
   leak, an over-animating model fails articulation.
5. **A reproducibility artifact** — frozen splits, pinned environments, a validated
   open scorer, and the documented corrections that make published numbers
   comparable, including a negative result (§4.4) reported in full.
6. **A Bangla corpus** — 2.78 h, 24 speakers, 28 videos, quality-filtered and
   manifested. *Conditional on provenance; see §8.*

*Dropped from the original list: the universal Bangla SyncNet. A better-trained
SyncNet would share the structural insensitivity described in §4.4, so it would not
answer the question the paper asks.*

---

## 7. Work remaining

| | status |
|---|---|
| §7.1 mask dose-response | ✅ **done — falsified the hypothesis (§4.4)** |
| §7.3 scale the comparison | ✅ done — 7 systems redwan, 6 HDTF, one scorer |
| §7.4 second dataset (HDTF) | ✅ done — 6 identities, frozen manifest |
| §7.5 universal Bangla SyncNet | ⬛ dropped, see §6 |
| **§7.0 retrain on the train split** | ⛔ **blocking — 14.6 h, see §3** |
| §7.2 human study | ⏳ not started — **critical path** |
| §7.6 reconstruction metrics | ⏳ PSNR/SSIM/LPIPS/FID, lossless reruns |
| §7.7 viseme head | ⏳ gated on Bangla phoneme alignment quality |
| §7.8 speed on RTX 3050 | ⏳ the consumer-hardware claim is ours to prove |

**§7.2 gates the paper.** ≥15 native Bangla speakers, blind, randomised, rating
lip-sync and naturalness; MOS correlated against LSE-C and against the proposed
protocol. **If the proposed metric tracks human judgment better, that justifies it;
if not, we do not propose it.** No compute cost, longest lead time, not started.

---

## 8. Risks and responses

| Risk | Response |
|---|---|
| **"You propose a metric your own system wins."** | (a) Justify by human correlation, never by ranking. (b) Our own as-released model is in the main table and scores badly on it. (c) Our headline reconstruction numbers were contaminated and we found and reported it ourselves (§3). |
| **"Your central claim changed."** | Yes — we designed the experiment that could kill it, ran it, and it did (§4.4). The negative result is in the paper, with the table. This is a strength, and it is why §4.5 and §4.6 are trustworthy. |
| **"Wav2Lip trained on that SyncNet, so obviously."** | Stated first, and quantified: 2 of 6 systems use SyncNet supervision — yet **four** beat real video on LSE-C, so training-against-the-metric is not the whole explanation. |
| **Leakage instability cuts both ways.** Our own 0.001 for final_v2 is a single clip (§4.6). | Acknowledged in the table. Mitigated by the §4.4 causal argument, which does not depend on clip count: the repaired model cannot leak jaw pixels it cannot see. Cross-speaker confirmation on 2–3 Bangla corpus speakers is the honest next step. |
| **KeySync (arXiv 2505.00497) reached the masking conclusion independently.** | Cite as corroboration. They design a new model; we audit a deployed one, and §4.5 shows their leakage metric needs a normalisation they do not apply. |
| **Corpus provenance.** `sources.csv` has `source`/`licence`/`consent` empty for all 38 videos. | **Unresolved and blocking for release.** The LRS3/VoxCeleb model requires publishing video IDs so others can rebuild — impossible without source URLs. If unrecoverable, contribution 6 is withdrawn. |
| **Person-specific vs person-generic asymmetry.** | Declared in the results table, not a footnote. SyncTalk_2D is excluded from the HDTF table rather than trained on 50 s of footage to manufacture a row. |

---

## 9. Budget and schedule

| Item | Compute | Calendar |
|---|---|---|
| Retrain on the train split (§7.0) | ~15 units | 2 days |
| Human study (§7.2) | 0 | 4–6 weeks *(start now)* |
| Reconstruction metrics (§7.6) | ~5 units | 3 days |
| Viseme head (§7.7) | ~25 units | 1–2 weeks |
| Speed on RTX 3050 (§7.8) | 0 | 1 day |
| **Total** | **~45 units** | **~6 weeks to submission** |

Spent so far: the dose-response, HDTF collection, six systems × two datasets, and
the rescoring. The human study remains the critical path — everything else can be
bought with compute; fifteen people's calendars cannot.

---

## 10. Requests

1. **Confirm the reframing holds after the negative result.** The original pitch was
   "sync metrics reward leakage." That is falsified. The paper is now "the standard
   protocol cannot see this class of failure, for three measurable reasons" —
   supported by a significant result on public data (§4.3) and two new findings
   (§4.5, §4.6). I believe it is still a TMM-tier paper and a more honest one, but
   I would like your view before committing six weeks.
2. **A written ethics determination** for the corpus. Needed under either framing
   and cannot be retrofitted after review.
3. **Help recruiting** ≥15 native Bangla speakers.
4. **Confirm APC coverage.** IEEE Access is $2,160 for 2026; TMM differs.

---

## Appendix — reproducibility and definitions

All results reproduce from `benchmark/` in this repository plus `MyDrive/FYDP/`:
pinned lockfiles, a one-command setup script, baseline weights, frozen test clips,
and the validation table that must pass before any new number is trusted.

**Test splits.** redwan frames 6942–7713 (772 frames, 25 fps, 1080×1080) per
`evaluation/manifests/redwan_splits.json`; HDTF per `hdtf/hdtf_test_manifest.json`,
6 identities of 14 rebuilt from 15 selected, retrieved 2026-09-17.

**Metric definition.** Aperture = (mean *y* of inner-lower-lip − mean *y* of
inner-upper-lip) ÷ face width, using this repository's `pfld_mobileone` detector.
"Mouth open" = aperture > 0.03, chosen from the source distribution (p10 0.004,
median 0.029, max 0.126). KeySync uses MediaPipe MAR at threshold 0.25 — the same
construction at a different scale. **These numbers are not interchangeable with
theirs.**

**Metric precision.** The 0.03 threshold sits at the *median* of the source
distribution, where the most frames lie, so landmark noise flips them.
Reproducibility is **±0.02** on the thresholded fraction; differences smaller than
that mean nothing. We report `median_aperture` — continuous and stable — alongside.

**Normalised leak** = leak ÷ that identity's ground-truth articulation, taken as the
median over identities (§4.5). Aggregating ratios by the mean is unsafe here:
`AustinScott`'s denominator of 0.089 alone moved MuseTalk v1.0's articulation ratio
from 0.750 to 1.584.
