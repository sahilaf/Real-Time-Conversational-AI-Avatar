# Execution Roadmap

**Order fixed 2026-09-17. Status updated 2026-09-18.**

    1. HDTF data ✅  →  2. Baselines ✅  →  3. Viseme head  →  4. Full benchmark
                                        →  5. Human study

Stages 1 and 2 are **done**. Both datasets are scored by one validated scorer.
Stage 3 is next, gated on a check that has not been run yet.

**One thing is still overdue, out of order:** the ethics determination and
participant recruitment for Stage 5. Those are calendar, not work, and nothing
about them has started. Every day they wait adds a day to the end of the project,
because compute cannot shorten them.

**A new blocker appeared on 2026-09-18** and sits ahead of Stage 3 — see
*Stage 2b*.

---

## Stage 1 — HDTF: collect and prepare ✅ DONE

Frozen as `hdtf/hdtf_test_manifest.json`, rebuildable by a third party from
`select_subset.py` + `download_prep.py`.

| | |
|---|---|
| Intended | 15 identities, one clip each, 30.88 s @ 25 fps @ 512×512 |
| Rebuilt | **14** — `AllenWest` is now Private on YouTube |
| Used for evaluation | **6** |
| Retrieved | 2026-09-17 |

`Radio` was dropped after the fact: its 4K source returned only 10.84 s of audio
against a 772-frame video, and every system silently truncated to ~268 frames.
`CoryGardner` replaced it. The evaluation subset is 3 WDA / 3 WRA:

    AdamSchiff  AnnWagner  AdamSmith  AustinScott  AmyKlobuchar  CoryGardner

**Report the retrieval date and the attrition count in the paper.** Anyone
rebuilding later gets fewer clips.

---

## Stage 2 — Baselines ✅ DONE

Scaled past the original plan: **six** systems on HDTF rather than three added to
five, with every cell run by us through one harness.

| | |
|---|---|
| HDTF | 6 systems × 6 identities × {real, silence} = **72 outputs**, all scored |
| redwan | 9 real + 6 silence = **15**, rescored with the same code |
| Systems | wav2lip, wav2lip_gan, musetalk_v1, musetalk_v15, latentsync, ip_lap |
| Metrics | LSE-D, LSE-C, AV offset, leak, articulation, median aperture |

VideoReTalking was dropped as too slow. TalkLip was not run — its published
leakage number (0.66) remains an unused cross-check.

**The scorer is `benchmark/` in this repo**, validated against the 2026-09-13
figures and reproducing across three Colab machines to within 0.004 on LSE-D.
The first version of it was lost with a dead Colab session; it lives in the
repository now for that reason.

### What Stage 2 actually found

1. **Generated video beats real video, significantly.** Paired over 6 identities,
   wav2lip (+0.931, p=0.004), latentsync (+0.807, p=0.020) and wav2lip_gan
   (+0.742, p=0.007) all score above ground truth on LSE-C. Five of six beat it
   on LSE-D. This is the redwan observation replicated on public data with
   statistics.

2. **Leakage is confounded with how much the source speaker moves.** Pooled
   Spearman between ground-truth articulation and measured leak is **+0.642,
   p=0.00002** (n=36). A system tested on a low-articulation speaker looks
   leak-free regardless of its behaviour. Raw LipLeak is not comparable across
   datasets; divide by source articulation.

3. **Leakage rank reverses between datasets.** Normalised leak for musetalk_v15
   is 0.008 on redwan and 0.934 on HDTF — best to worst, same checkpoint.
   redwan is one clip; HDTF's per-identity spread routinely exceeds its own mean.
   **A single-clip leakage number carries almost no information.**

4. **LSE-C is not blind to leakage, but its sensitivity is too small to matter.**
   The cleanest and dirtiest systems differ by 0.556 (p=0.030) — while the dirtiest,
   copying 93% of source mouth motion, still outscores real video. Across six
   systems the rank correlation is −0.543, p=0.266, indistinguishable from zero.
   State it that way; "LSE-C is blind" overclaims.

---

## Stage 2b — Retrain on the train split ⛔ BLOCKING

**Found 2026-09-18.** `MyDataset` enumerated every frame and `train_328.py`
passed it through, so every SyncTalk_2D checkpoint trained on frames 0..7713 —
including the test split 6942..7713 that every redwan number is measured on. The
appearance reference was drawn from the whole video too.

Fixed in `datasetsss_328.py` / `train_328.py` (`--manifest`), pushed to
`evaluation-tooling`. `setup_colab.sh` clones that branch until it is merged.

**Consequence:** every `final_v2` and legacy number in the current tables is
measured on frames the model trained on, and is not comparable with the
person-generic baselines, which have never seen the video. The **leakage** result
is unaffected — it rests on the mask removing the jaw pixels, shown causally
across the six ablation arms.

**Work**

- [ ] Retrain `final_v2` and the as-released arm, 100 epochs, `--manifest`
- [ ] Verify the log prints `Frames 0..6170 (6171 of 7714 usable)` before
      committing the full run
- [ ] Re-measure PSNR / SSIM / MAE / LSE / articulation from the clean models
- [ ] Replace the affected rows in `proposal.md` §3 and §4

**Cost:** 4.4 min/epoch measured on an L4 → **7.3 h per model, 14.6 h total.**
This is the run that was already planned; it now uses the fixed loader, so the
fix costs nothing extra. Doing it later costs the 14.6 hours twice.

---

## Stage 3 — Viseme head

**Why:** everything measured so far is **aperture** — how far the mouth opens.
The proposal claims correct Bangla *lip shape*, and nothing supports that yet.
This is the first component that would make /a/ differ from /e/ by construction.

**Check this before any design work.** The stage depends on Bangla ASR with
usable phoneme alignment. If the alignment is poor the stage is unbuildable and
the order needs revisiting.

**Work**

- [ ] **Gate:** run Bangla phoneme alignment over a few minutes of redwan audio
      and inspect it by hand against the waveform, specifically on the pairs that
      matter — aspirated/unaspirated (ক/খ, ত/থ, দ/ধ), retroflex/dental (ট/ত, ড/দ),
      nasal vowels. If those collapse, no head can learn the distinction.
- [ ] Map phonemes to a Bangla viseme inventory
- [ ] Auxiliary classification head on the generated mouth region
- [ ] Train with the head; **ablate it** — same recipe with the head off is the
      only way to show it earns its place

**Gate:** the head improves viseme discriminability on held-out Bangla speech,
with the ablation arm to prove the gain is the head and not the extra training.

**Cost:** ~25 units. **Calendar: 1–2 weeks.** Biggest risk in the plan.

---

## Stage 4 — Full benchmark

Much of this is done. What remains:

- [ ] **PSNR / SSIM / LPIPS / FID** — needs lossless reruns; Wav2Lip and MuseTalk
      re-encode internally before their frames are reachable
- [ ] **Speed on the RTX 3050.** The consumer-hardware claim is the paper's, and
      T4/A100/L4 numbers must never stand in for it
- [ ] Per-8-second-segment scores as well as per-clip, since the human study
      correlates per segment
- [ ] Add SyncTalk_2D's clean arms once Stage 2b lands
- [ ] Report leak **normalised by source articulation**, with raw leak alongside

**Cost:** ~10 units remaining. **Calendar: 3–4 days.**

---

## Stage 5 — Human study

Runs on the **final** model. Full design in
[human_study_plan.md](human_study_plan.md) — 6-person pilot, then ≥15.

**Overdue, should have started on day one:**

- [ ] Written ethics determination from the department
- [ ] Recruit ≥15 native Bangla speakers — provisional commitments suffice until
      stimuli exist

**After Stage 4:**

- [ ] Cut 8-second segments from the final benchmark outputs
- [ ] Pilot (6) → four instrument checks → fix → main study (≥15)
- [ ] Correlate MOS against LSE-C and against normalised leak, per segment

**Cost:** 0 units. **Calendar: 4–6 weeks**, almost entirely recruitment.

---

## Decision points

**After Stage 1 — resolved.** 14 of 15 rebuilt, 6 used. Above the ~8 threshold
that would have forced a second public set.

**After Stage 2 — resolved, and not as predicted.** The roadmap said a near-zero
Spearman between LSE-C and leak would confirm blindness at scale, and a strongly
negative one would revive the "rewards leakage" claim. The measured value is
**−0.543, p=0.266** at n=6: underpowered, indistinguishable from zero, and
*negative* rather than positive — the opposite sign from "rewards leakage". That
claim does not return. The defensible statement is the one in Stage 2, finding 4.

**Before Stage 3 — open.** If Bangla phoneme alignment is not good enough, the
viseme head is unbuildable. Fall back to the human study as the sole evidence for
lip shape and move Stage 5 forward.

**New — how much does SyncTalk_2D need to generalise?** Training it on 6 HDTF
identities is not possible: `AustinScott` has 50 s of usable footage against the
247 s redwan trained on, and three others have roughly half. At 7.3 h per
identity, even the three viable ones cost 22 h. Prefer 2–3 speakers from the
**Bangla corpus** — same cost, and it supports the claim the proposal actually
makes.

---

## Carried forward from what is already done

Do not rediscover these:

- **PCM-in-AVI before scoring, never AAC** — AAC priming adds a 2-frame offset
- **But do not re-mux a file that already has audio.** Several redwan AVIs declare
  `avg_frame_rate 50/1` over 772 real frames; re-containering those moved
  ground-truth LSE-C from 5.135 to 6.109. Files declaring 25/1 were unaffected,
  so the error hit six systems and spared two.
- **Ground-truth anchor row**, every dataset, every table
- **Verify alignment per system, geometrically** — slide the output against the
  source on the top 100 rows, sweep symmetrically. LSE reads at the best offset
  and so cannot detect a harness misalignment.
- **LatentSync pads at the START when chunked, at the END when not.** Re-measure
  whenever the chunking changes.
- **IP_LAP writes `[sketch|result]`** — crop the right half. The redwan archive
  was stored uncropped and scored articulation 0.9935 instead of 0.3854.
- **`--manifest` on every training run**, or the model trains on its own test set
- **`--ref_frame` pinned to the train split**; **`--start_frame 6941`**, not 6942
- **`train_config.json` beside every checkpoint** or the mask falls back to legacy
- **Normalise input loudness** — RMS 1316 gives LSE-C 4.601 against 6.0–6.9 for
  louder speech. Fix in `StreamingFeatureExtractor.add()`. Still open.
- **Articulation reproduces to ±0.02.** The 0.03 threshold sits at the median of
  the source distribution, where the most frames are. Report `median_aperture`
  alongside; differences below 0.02 mean nothing.
- Full lists: `benchmark/README.md`, `FYDP/benchmarks/20260913_redwan_test/README.md`

---

## Deliberately not in this plan

- **Universal Bangla SyncNet.** A better-trained SyncNet would be blind to
  leakage for the same structural reason the current one is. Parked.
- **InsTaG / TalkingGaussian / SyncTalk NeRF.** Person-specific comparison class;
  belongs to a system paper, not this one.
- **Sync-loss dose-response.** Came out flat; the loss uses our SyncNet while the
  score uses Oxford's, so it was never the Wav2Lip case.
- **SyncTalk_2D on the HDTF identities.** Not affordable and not what the
  proposal claims. See the decision point above.
