# Research Proposal — Evaluation Integrity in Audio-Driven Talking-Head Generation

**Author:** Sahil al Farib
**Date:** 13 September 2026
**Target venue:** IEEE Transactions on Multimedia (Q1) · fallback: IEEE TCSVT, Pattern Recognition

---

## 1. Thesis

> A widely-deployed talking-head baseline is **not audio-driven at all** — and the field's
> standard metrics rank it as working. We trace the defect to a specific masking error,
> repair it, and then show why no existing protocol could have caught it: lip-sync score and
> expression leakage rank systems **inversely**, so the leakiest system scores best.

The contribution is not the bug fix. It is the demonstration that a defect this severe
survived for years in a widely-used model *because the community's evaluation protocol is
blind to it, and in fact rewards it.*

---

## 2. How we found it

We set out to deploy a lightweight talking-head model for Bangla. The avatar moved its
mouth during silence.

The model receives the current video frame with the mouth blacked out, plus audio, and must
paint the mouth back in. The masking code used OpenCV's `(x, y, w, h)` rectangle form where
the point form was intended, so it blacked out rows 5–310 of a **320**-row crop. Rows
310–319 — the chin and jaw — stayed visible in both training and inference.

Jaw drop predicts mouth opening almost perfectly. Holding audio and appearance reference
constant, that ten-pixel strip accounted for **91% of all mouth motion**. The model had
learned to read the chin rather than listen.

Its lip-sync expert could not object: it was trained with `y = torch.ones(1)`
unconditionally — positive pairs only, no negatives anywhere in the file — so it approved of
everything and supplied no gradient.

**All four defects are confirmed present upstream at `ZiqiaoPeng/SyncTalk_2D @ 9e82dcc`,**
with file:line references. They are not artefacts of our fork.

| Defect | Location |
|---|---|
| SyncNet trained on positives only | `syncnet_328.py:106` |
| Mask leaves the jaw exposed | `datasetsss_328.py:98`, `inference_328.py:115` |
| Sync loss weighted 10x | `train_328.py:109` |
| Train/inference reference mismatch | `datasetsss_328.py:135` vs `inference_328.py:112-121` |

---

## 3. The repair, measured

| | as-released | repaired | note |
|---|---:|---:|---|
| PSNR | 29.830 | **33.114** | repaired scored under a *stricter* mask |
| SSIM | 0.8824 | **0.9114** | |
| MAE | 0.0206 | **0.0144** | |
| Leakage (mouth open on silence) | 0.422 | **0.000** | |

A control run isolates the mechanism. Re-running the as-released model with a fixed
appearance reference instead of the shipped per-frame reference moves leakage from 0.427 to
0.422 — a 1% difference. **The mask is the entire mechanism; the reference contributes
nothing.** This independently reproduces an earlier probe that attributed 0.4% of mouth
motion to the reference.

---

## 4. The field-level finding

Five systems, one 31-second Bangla test clip, one harness, ground-truth anchored.

### 4.1 Lip-sync (standard metrics)

| System | ckpt | T4 time | regime | offset | LSE-D ↓ | LSE-C ↑ |
|---|---:|---:|---|---:|---:|---:|
| Ground truth (real video) | — | — | real | 0 | 7.436 | 5.103 |
| Wav2Lip (gan) | 436 MB | 9m06s | person-generic | −2 | 7.259 | **6.080** |
| LatentSync 1.5 | 5072 MB | ~30m | person-generic | 0 | 7.259 | 5.128 |
| MuseTalk v1.5 | 3400 MB | ~15m | person-generic | 0 | 7.860 | 4.910 |
| SyncTalk_2D final_v2 (ours) | **49 MB** | **2m33s** | person-specific | 0 | 7.276 | 5.140 |

### 4.2 Leakage and articulation (paired silent/real protocol)

| System | Leak ↓ | Articulation → 1 | distance from ideal |
|---|---:|---:|---:|
| Source video | — | 1.00 | — |
| **SyncTalk_2D final_v2 (ours)** | **0.000** | **0.97** | **0.03** |
| LatentSync 1.5 | 0.368 | 0.86 | 0.39 |
| SyncTalk_2D as-released | 0.422 | 1.08 | 0.43 |
| MuseTalk v1.5 | 0.005 | 0.55 | 0.45 |
| Wav2Lip | 0.444 | 1.28 | 0.53 |

### 4.3 The inversion

**Wav2Lip ranks 1st on LSE-C and last on leakage.** Fed pure digital silence, its mouth is
open in 44% of frames against real speech's 49% — it is barely listening — and the standard
metric calls it the winner.

The mechanism is explicable: a leaking model copies the ground-truth mouth, and the
ground-truth mouth matches the audio by construction. **Leakage is a shortcut to a high sync
score.** The metric cannot distinguish cheating from success.

This is not confined to one model. LatentSync, a 5 GB diffusion model from ByteDance, leaks
at 0.368. Both it and Wav2Lip use SyncNet supervision during training — two of the four
systems tested optimise directly against the metric that evaluates them.

---

## 5. Supporting evidence that the protocol is unreliable

- **LSE-C is unstable at clip length.** Four consecutive 7.7-second windows of the *same*
  video from the *same* model scored 6.484, 5.659, 4.828, **2.167** — a 3x spread with
  nothing varying but which eight seconds you look at.
- **The reference implementation is inconsistent.** The script the field uses to compute LSE
  contains `cv2.resize(img_input, (224,224))` with the literal comment
  `#HARD CODED, CHANGE BEFORE RELEASE`. The original `syncnet_python` applies no such
  resize. The two disagree, and published numbers do not state which was used.
- **Audio muxing silently corrupts offsets.** AAC encoder priming introduces a 2-frame
  (80 ms) apparent AV offset. Measured: identical video scored offset −2 via AAC, 0 via PCM.
- **Most papers report no ground-truth anchor.** Real video here scores LSE-C 5.103. Without
  that row, Wav2Lip's 6.080 reads as "better lip-sync than a real human."
- **The standard expert under-scores non-English.** Real Bangla video scores LSE-C 5.103
  where real English is typically reported at 6.8–8.2.

---

## 6. Proposed contributions

1. **A defect audit** of a widely-used baseline, with file:line evidence and a quantified
   causal mechanism (91% of mouth motion from a ten-pixel strip).
2. **Evidence that sync metrics reward leakage** — correlational across systems and, pending
   §7.1, causal within one architecture.
3. **A paired evaluation protocol** (Leak + Articulation) constructed so that degenerate
   solutions cannot pass: a frozen mouth fails on articulation, a copying model fails on
   leak, an over-animating model fails on articulation. Only genuine audio-dependence
   satisfies both.
4. **A universal Bangla SyncNet** — a multi-speaker lip-sync expert that no system in the
   comparison trained against, released as an independent instrument.
5. **A Bangla corpus** — 2.78 h, 24 speakers, 28 videos, quality-filtered and manifested.
6. **A reproducibility artifact** — frozen splits, pinned environments, one-command
   evaluation, and ten documented corrections that make published numbers comparable.

---

## 7. Work remaining

### 7.1 The decisive experiment — mask dose-response *(highest priority)*

Current evidence for the inversion is correlational across four systems with different
architectures, training data and objectives. We own the confound-free version: **the mask is
a continuous dial.**

Train one architecture, one dataset, matched epochs, varying only the number of jaw rows
exposed (10 → 8 → 6 → 4 → 2 → 0). If LSE-C rises monotonically with leakage, we demonstrate
**causally, within a single architecture**, that the standard metric rewards leakage.

Six arms run concurrently on one A100 (the model needs 4 GB of 80). Approx. 10–15 units.

**Run this first.** If the curve is flat, the causal claim dies and we revert to the
correlational one — and we want that answer in week one, not week six.

### 7.2 Human study *(gates the whole paper)*

≥15 native Bangla speakers, blind, randomised. Rate lip-sync and naturalness. Correlate MOS
against LSE-C and against the proposed protocol. **If the proposed metric tracks human
judgment better, that justifies it; if not, we do not propose it.** No compute cost; longest
calendar lead time; start immediately.

### 7.3 Scale the comparison

Add VideoReTalking, IP-LAP, TalkLip (all lightweight, all T4-feasible). Target 7–8 systems,
then report a rank correlation with a p-value rather than an observation.

### 7.4 Second dataset

HDTF test split — English, public, standard. Demonstrates the effects are not Bangla-specific
and answers the "why not a public benchmark" objection.

### 7.5 Universal Bangla SyncNet

Corpus built, preprocessing complete. ~40 units on A100. Acceptance test: the offset curve
must peak at 0 **on held-out speakers**, not merely on training voices.

### 7.6 Reconstruction metrics

PSNR/SSIM/LPIPS/FID. Requires lossless reruns — Wav2Lip and MuseTalk both re-encode
internally before their frames are reachable.

---

## 8. Risks and responses

| Risk | Response |
|---|---|
| **"You propose a metric your own system wins."** Serious — can sink the paper on its own. | (a) Justify by human correlation, never by ranking. (b) Our own as-released model scores 0.43 on it, worse than LatentSync — the metric indicts our earlier work, and that row goes in the main table. (c) Pre-register the threshold and report a sweep. |
| **"Two systems is not an anti-correlation."** | §7.1 (causal, confound-free) plus §7.3 (n=8 with statistics). |
| **"Wav2Lip trained on that SyncNet, so obviously."** | That is the point, stated first — and quantified: 2 of 4 systems tested use SyncNet supervision. |
| **KeySync (arXiv 2505.00497) reached the same masking conclusion independently.** | Cite as corroboration, not competition. They design a new model; we audit a deployed one and show *why the metrics let it survive*. Adopt their LipLeak framing with credit. |
| **Corpus provenance.** `sources.csv` has `source` / `licence` / `consent` empty for all 38 videos; footage arrived as third-party archives. | **Unresolved and blocking.** The LRS3/VoxCeleb release model requires publishing video IDs so others can rebuild — impossible without source URLs. Action item this week. |
| **Epoch confound in the before/after.** Rung 0 is epoch 19; repaired is epoch 59. | Matched 100-epoch runs planned. The mechanism control (§3) already argues the leak is structural rather than a symptom of undertraining. |
| Person-specific vs person-generic asymmetry. | Declared in the results table, not a footnote. |

---

## 9. Budget and schedule

| Item | Compute | Calendar |
|---|---|---|
| §7.1 dose-response | 10–15 units | 1 week |
| §7.2 human study | 0 | 4–6 weeks *(start now)* |
| §7.3 three more systems | ~5 units | 1 week |
| §7.4 HDTF | ~10 units | 1 week |
| §7.5 universal SyncNet | ~40 units | 1 week |
| §7.6 reconstruction metrics | ~5 units | 3 days |
| **Total** | **~75 of ~90 units remaining** | **~8 weeks to submission** |

The human study is the critical path. Everything else can be bought with compute; fifteen
people's calendars cannot.

---

## 10. Requests

1. **Confirm the reframing.** This moves the paper from "Bangla dataset + benchmark"
   (IEEE Access tier) to "evaluation integrity" (TMM tier), and makes our own system
   supporting evidence rather than the headline. I believe it is the stronger paper, but I
   would like your view before committing eight weeks.
2. **A written ethics determination** for the corpus (broadcast footage, no per-speaker
   consent, no media redistribution, published takedown route). Needed under either framing
   and cannot be retrofitted after review.
3. **Help recruiting** ≥15 native Bangla speakers for the human study.
4. **Confirm APC coverage.** IEEE Access is $2,160 for 2026; TMM differs.

---

## Appendix — reproducibility and definitions

All results reproduce from `MyDrive/FYDP/`: pinned environment lockfiles, a one-command
setup script, all baseline weights, the frozen test clip, and a runbook notebook. Ten
methodological corrections are documented there, including the PCM-vs-AAC requirement and
the LSE implementation discrepancy.

**Test split.** Frames 6942–7713 of `redwan` (772 frames, 25 fps, 1080×1080), per
`evaluation/manifests/redwan_splits.json`.

**Metric definition.** Aperture = (mean *y* of inner-lower-lip − mean *y* of inner-upper-lip)
÷ face width, using this repository's `pfld_mobileone` landmark detector. "Mouth open" =
aperture > 0.03, chosen from the source distribution (p10 0.004, median 0.029, max 0.126).
KeySync uses MediaPipe MAR at threshold 0.25 — the same construction at a different scale.
**These numbers are not interchangeable with theirs**, and any comparison must state both
definitions.
