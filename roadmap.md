# Execution Roadmap

**Order fixed 2026-09-17.** Human study runs last, on the final model, so its
ratings apply to the version that ships rather than an intermediate one.

    1. HDTF data          →  2. Three baselines  →  3. Viseme head
                          →  4. Full benchmark   →  5. Human study

**One thing must start immediately, out of order:** the ethics determination and
participant recruitment for Stage 5. Those are calendar, not work. If they only
begin when Stage 4 ends, they add 4–6 weeks to the end of the project instead of
overlapping with it. Everything else follows the order above.

**Budget: ~90 Colab units remaining. This plan spends ~65.**

---

## Stage 1 — HDTF: collect and prepare

**Why:** IEEE Access and TMM both reject evaluation on a private dataset with no
public benchmark. HDTF is the field standard (15.8 h, English, 512p) and it is
distributed the same way our corpus would be — a URL list plus crop metadata,
not media.

**Work**

- [ ] Clone the HDTF repo, read `*_video_url.txt` and the crop annotations
- [ ] `yt-dlp` the source videos; **expect attrition** — HDTF is 2021 and YouTube
      videos disappear. Record how many resolve and report it.
- [ ] Apply the published crop windows, resample to **25 fps**, extract audio at
      16 kHz mono PCM
- [ ] Select a **fixed test subset** — 10–15 identities, ~30 s each — and freeze
      it as a manifest with checksums. Never re-pick it later.
- [ ] Store prepared clips in `FYDP/hdtf/` with a README recording the retrieval
      date and the attrition count

**Gate:** a frozen HDTF test manifest that a third party could rebuild, plus a
stated attrition figure.

**Cost:** ~5 units (mostly download and CPU). **Calendar: 2–3 days.**

**Risks:** attrition may be severe enough that the subset is small — report it
rather than quietly substituting videos. Check HDTF's own terms before
redistributing anything beyond the manifest.

---

## Stage 2 — Three more baselines

**Why:** five systems is an observation; eight supports a rank correlation with
a p-value. These three are all lightweight and T4-feasible.

| system | notes |
|---|---|
| **VideoReTalking** | SIGGRAPH Asia 2022, appears in nearly every comparison table |
| **IP-LAP** | CVPR 2023, standard baseline in recent work |
| **TalkLip** | has a published leakage number (0.66) to cross-check ours against |

**Work**

- [ ] One isolated venv each — assume version conflicts, they are the norm here
- [ ] Run each on **both** the redwan test split and the HDTF subset
- [ ] Three conditions per system: reconstruction, cross-audio, **silent**
- [ ] Archive weights and lockfiles to `FYDP/baselines/` as with the first four

**Gate:** eight systems scored on two datasets, all through the same harness.

**Cost:** ~15 units. **Calendar: 3–5 days** — budget a day per system for
environment problems, based on how the first four went.

**Carry forward:** PCM-never-AAC before scoring; ground-truth anchor row on both
datasets; record which LSE implementation was used.

---

## Stage 3 — Viseme head

**Why:** every measurement so far is **aperture** — how far open the mouth is.
The project proposal claims correct Bangla *lip shape*, and nothing currently
supports that. This is the first component that would make /a/ differ from /e/
by construction.

**Work**

- [ ] Bangla ASR with phoneme-level alignment over the redwan training audio.
      **Validate the alignment before building on it** — a misaligned phoneme
      track would silently teach the wrong shapes.
- [ ] Map phonemes to a Bangla viseme inventory. Group the pairs that matter:
      aspirated/unaspirated (ক/খ, ত/থ, দ/ধ), retroflex/dental (ট/ত, ড/দ),
      nasal vowels.
- [ ] Auxiliary classification head on the generated mouth region, predicting
      viseme class; cross-entropy alongside the existing losses.
- [ ] Train with the head; **ablate it** — same recipe with the head off is the
      only way to show it earns its place.

**Gate:** the head improves viseme discriminability on held-out Bangla speech,
with the ablation arm to prove the improvement is the head and not the extra
training.

**Cost:** ~25 units (several runs plus the ablation). **Calendar: 1–2 weeks.**

**Biggest risk in the plan.** It depends on a Bangla ASR with usable phoneme
alignment, and that may not exist at sufficient quality. **Check this first** —
before any design work — because if the alignment is poor the whole stage is
unbuildable and the order needs revisiting.

---

## Stage 4 — Full benchmark

**Why:** the results table for both the FYDP and the journal.

**Work**

- [ ] 8 systems × 2 datasets × 3 conditions (reconstruction / cross-audio / silent)
- [ ] Metrics: LSE-D, LSE-C, **Leak**, **Articulation**, PSNR, SSIM, LPIPS, FID
- [ ] Speed re-measured on the **RTX 3050** — not the T4. The consumer-hardware
      claim is the paper's, and A100/T4 numbers must never stand in for it.
- [ ] Per-8-second-segment scores as well as per-clip, since the human study
      correlates per segment
- [ ] Spearman between LSE-C rank and Leak rank across all systems, with p-value
- [ ] Final model = repaired + viseme head; keep as-released as the lower anchor

**Gate:** every cell filled by a system actually run, with a ground-truth anchor
row on both datasets.

**Cost:** ~20 units. **Calendar: 1 week.**

---

## Stage 5 — Human study

Runs on the **final** model. Full design in
[human_study_plan.md](human_study_plan.md) — two stages, 6-person pilot to
validate the instrument, then ≥15 for the result.

**Started in parallel from day one:**

- [ ] Written ethics determination from the department
- [ ] Recruit ≥15 native Bangla speakers — provisional commitments are enough
      until stimuli exist

**Done after Stage 4:**

- [ ] Cut 8-second segments from the final benchmark outputs
- [ ] Pilot (6) → four instrument checks → fix → main study (≥15)
- [ ] Correlate MOS against LSE-C and against Leak, per segment

**Cost:** 0 units. **Calendar: 4–6 weeks**, almost entirely recruitment.

---

## Timeline

| stage | units | calendar |
|---|---:|---|
| 1 HDTF | ~5 | 2–3 days |
| 2 Baselines | ~15 | 3–5 days |
| 3 Viseme head | ~25 | 1–2 weeks |
| 4 Full benchmark | ~20 | 1 week |
| 5 Human study | 0 | 4–6 weeks *(recruitment overlaps 1–4)* |
| **Total** | **~65 of 90** | **~8–10 weeks** |

Sequential, so the calendar is the **sum**, not the maximum — except recruitment
and ethics, which overlap everything.

---

## Decision points

**After Stage 1.** If HDTF attrition leaves fewer than ~8 usable identities, say
so and consider a second public set rather than reporting a thin subset.

**Before Stage 3.** If Bangla phoneme alignment is not good enough, the viseme
head is unbuildable. Fall back to the human study as the sole evidence for lip
shape, and move Stage 5 forward.

**After Stage 4.** If Spearman between LSE-C and Leak is near zero across eight
systems, that confirms the blindness finding at scale and it becomes the paper's
headline. If it is strongly negative, the stronger "rewards leakage" claim
returns — but only with eight systems, never with the five we have.

---

## Carried forward from what is already done

Do not rediscover these:

- **PCM-in-AVI before scoring, never AAC** — AAC priming adds a 2-frame offset
- **Ground-truth anchor row**, every dataset, every table
- **`--ref_frame` pinned to the train split** — the auto-pick can take an
  appearance reference from inside the test split
- **`--start_frame 6941`**, not 6942 — `img_idx` increments before the first read
- **`train_config.json` beside every checkpoint** or the mask silently falls
  back to legacy
- **Normalise input loudness** — measured: RMS 1316 gives LSE-C 4.601 against
  6.0–6.9 for louder speech. Fix in `StreamingFeatureExtractor.add()`.
- Full list: `FYDP/benchmarks/20260913_redwan_test/README.md`

---

## Deliberately not in this plan

- **Universal Bangla SyncNet.** A better-trained SyncNet would be blind to
  leakage for the same structural reason the current one is. Parked.
- **InsTaG / TalkingGaussian / SyncTalk NeRF.** Person-specific comparison class;
  belongs to a system paper, not this one.
- **Sync-loss dose-response.** The no-sync version came out flat; the loss uses
  our SyncNet while the score uses Oxford's, so it was never the Wav2Lip case.
