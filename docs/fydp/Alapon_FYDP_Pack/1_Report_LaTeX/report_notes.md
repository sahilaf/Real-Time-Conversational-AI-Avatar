# Final report

The final report is written in the official UIU LaTeX format — the same
template, preamble, file split and styles as the FYDP-1 report.

| | |
|---|---|
| **[latex/](latex/)** | Source: `fydp.tex` (main), `0.x` front matter, one file per chapter, `fydp.bib`, images |
| **[UIU_FYDP_Final_Report.zip](UIU_FYDP_Final_Report.zip)** | The same folder zipped — upload to Overleaf as a new project |

**Compile with XeLaTeX** (Overleaf: *Menu → Compiler → XeLaTeX*), then BibTeX.
The template uses `fontspec` + `polyglossia` with *Noto Serif Bengali*; the IPA
symbols in Table 3.3 use DejaVu Sans, which ships with TeX Live. The report
was not compiled locally (no TeX installation here), so check the first
Overleaf build.

**The story it tells:** no open real-time Bangla avatar existed → we built the
system → chose SyncTalk_2D and confirmed the choice by benchmark → adapted it to
Bangla → found four gaps in the model that limited its performance, the worst a
jaw leak → closed them (**Alapon**) and measured the improvement → found the same
leak in five of six published systems, invisible to the standard metric →
journal paper in progress → the finished system runs Alapon.

## What changed from the FYDP-1 report

| Chapter | Change |
|---|---|
| Abstract, Ch.1 | Past tense; 2D not 3D; methodology as seven stages; delivered system and measured numbers; the gap is stated as *open, local, Bangla-evaluated* (commercial cloud services such as HeyGen already offer Bangla voices) |
| Ch.2 | Literature rewritten; "Similar Applications" no longer claims the system uses InsTaG / 3DGS; 2D lip-sync methods, SyncTalk_2D and leakage added; gap table limited to rows we can verify |
| Ch.3 | Gemini 3.1 Flash Live pipeline; **Figures 3.1, 3.2 and 3.5 redrawn** in TikZ to match the built system; new Figure 3.6 (Alapon and its mask); model selection (Table 3.1); the four model gaps (Table 3.2); the unsupported "200–300 ms" claim removed |
| Ch.4 | Mask change, jaw-row ablation, Bangla sounds, comparison with all six published checkpoints, leakage in published systems (Bangla + HDTF), speed, XLS-R |
| Ch.5 | "BanglaTalk3D" and 3D rendering removed; consent and corpus statements accurate; RAM 8 GB; cost table updated; A1–A5 table header order fixed |
| Ch.6 | Written (was a template): proposed-vs-delivered, limitations, journal paper in progress (6.3.1), extensions |
| References | Every entry checked against where it was published (Crossref / publisher); arXiv only where no peer-reviewed version exists (DFA-NeRF, LatentSync, MuseTalk) |

**Credibility fixes applied in this version** (from the review of 27 Sep):
the "no Bangla avatar exists" claim narrowed; reconstruction presented as
original model vs Alapon, not as the mask's effect; LSE-C "closest to real"
replaced by "matches real, within precision"; silence values under 1% not
ranked (±2-point precision); significance stated after multiple-comparison
correction (two systems, not three) and the correlation without a p-value;
phoneme timing described as it is actually computed; Table 3.1 times labelled
as offline and the Wav2Lip file size flagged as including optimiser state;
unverifiable Table 2.1 rows removed; the live response-time breakdown and
the unlogged "~28 fps" removed.

## The numbers — one value per claim, used everywhere

| Claim | Value |
|---|---|
| Model name | **Alapon** — built on SyncTalk_2D; checkpoint `SyncTalk_2D/checkpoint/alapon/59.pth` (formerly `final_v2`) |
| Checkpoint files | Alapon 49 MB; Wav2Lip 436 MB (includes optimiser state); IP-LAP 475; MuseTalk 3,400; LatentSync 5,072 |
| Speed on RTX 3050 laptop | 30 fps (33 ms per frame), 0.3 GB GPU memory |
| Response time | 1.5–2 s, including a deliberate 0.55 s end-of-turn wait |
| Mouth moving during silence | 41% → 0.1% (precision ±2 points) |
| Reconstruction (original model, 19 epochs → Alapon, 59) | PSNR 29.8 → 33.1 dB, SSIM 0.88 → 0.91, MAE 0.021 → 0.014 |
| Lip-sync, Bangla test clip (LSE-C) | real 5.14, Alapon 5.11, LatentSync 5.09 — equal within ±0.04 |
| Bangla sounds, open ÷ closed | Table 4.3: real 4.0×, **Alapon 3.8×**, before fix 3.0× · Table 4.4 (six unseen speakers): **Alapon 2.3×**, before fix 1.7× |
| Leakage in published systems | 5 of 6 leak 20–46% on at least one dataset; only Wav2Lip non-GAN low on both (0.8% / 8%) |
| Metric above real video (HDTF) | 4 of 6 above real on LSE-C; significant after correction for Wav2Lip (p = 0.004) and Wav2Lip GAN (0.007); LatentSync (0.02) not |
| Leakage depends on speaker | rank correlation +0.64 over 36 speaker–system pairs |
| XLS-R | lip-sync expert separated sync ~half as well as default (0.27 vs 0.50); not used |

## Before submitting

Search the `.tex` files for **TODO** (rendered in bold as `[TODO: …]`):

| Where | What |
|---|---|
| Table 3.3 | Have someone confident in Bangla phonetics check the viseme table |
| Section 5.1.4 | Confirm the `redwan` recording is of Sheikh Redwanul Islam, with consent |
| Table 5.1 | Colab spend and hours; Gemini API cost (free tier or amount) |

**When the clean retrain finishes:** replace Alapon's rows in Tables 4.1, 4.3
and 4.5 with the new numbers, and delete the caveat in 4.2 and the "Training
overlapped the test frames" limitation in 6.2.

## Where every number comes from

For the viva, when someone asks "where does that figure come from?"

| Number in the report | Source |
|---|---|
| Silence 41% → 0.1%; Table 4.5 (Bangla comparison) | `FYDP/benchmarks/20260917_stage2/redwan_scores.csv` (Alapon's rows under its old name `synctalk2d_final_v2`); silence videos for Wav2Lip non-GAN, MuseTalk 1.0, IP-LAP in `videos_silence_redwan/` |
| Table 4.2 (visible jaw rows); training error 0.0178 → 0.0202 | `FYDP/ablation_mask/results_dose_response.csv`; `docs/research/proposal.md` §4.4 |
| Table 4.1 reconstruction, Alapon | `SyncTalk_2D/evaluation/runs/20260818_005707_redwan_59_reconstruction_test/` |
| Table 4.1 reconstruction, original model | `SyncTalk_2D/evaluation/runs/20260710_192744_redwan_19_reconstruction_test/` (checkpoint `redwan/19.pth`) |
| Tables 4.3 and 4.4 (Bangla sounds), 80 ms offset | `python benchmark/phoneme_eval.py` — reproduces both from `benchmark/results/phoneme/` |
| Table 4.6 (HDTF) | `FYDP/benchmarks/20260917_stage2/hdtf_scores.csv` |
| Paired t-tests, +0.64 correlation | `docs/research/proposal.md` §4.3–4.5 |
| Table 3.1 sizes and T4 times | checkpoint files (Colab, 25 Sep); `FYDP/benchmarks/20260913_redwan_test/README.md` |
| Table 3.2 gaps; 91% of mouth motion; 1% reference control | `docs/research/proposal.md` §2–3 |
| 30 fps, 33 ms, 0.3 GB | render benchmark on the RTX 3050, 24 Sep — **move `bench_3050.py` into `benchmark/`** |
| Response 1.5–2 s; reply time 8 → 66 s on the old model | root `README.md` (Measured latency); `agent/agent_bangla.py` |
| XLS-R: separation 0.27 vs 0.50 | `syncnet_ckpt/redwan_ssl/train_log.csv` and the Colab run of the three arms |
| Corpus: 38 collected, 28 kept, 24 speakers, 2.78 h | local `research/corpus/` (`sources.csv`, `manifest.json`) |
| Laptop: Ryzen 7 5800H, 8 GB RAM, RTX 3050 4 GB | measured on the machine, 24 Sep |
