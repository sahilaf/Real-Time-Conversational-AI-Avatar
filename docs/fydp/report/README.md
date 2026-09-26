# Final report

**[FYDP_Final_Report.md](FYDP_Final_Report.md)** is the complete final report —
every chapter, rewritten against the system as built and the measured data. It
replaces the FYDP-1 report (`UIU_FYDP__Report.pdf`), which described a planned
system in the future tense, a cascaded pipeline that was later replaced, a 3D
InsTaG design that was never built, and results from a model since improved.

**The story it tells:** no real-time Bangla avatar existed → we built the system
→ tested published models and picked SyncTalk_2D → adapted it to Bangla → found
four gaps in the model that limited its performance, the worst a jaw leak →
closed them (**Alapon**) and measured the
improvement → found the same leak in five of six published systems, invisible to
the standard metric → journal paper in progress → the finished system runs
Alapon.

## What changed from the FYDP-1 report

| Chapter | Change |
|---|---|
| Abstract, Ch.1 | Past tense; 2D not 3D; methodology told as the seven stages above; delivered system and measured numbers |
| Ch.2 | Literature review rewritten; **"Similar Applications" no longer claims the system uses InsTaG / 3D Gaussian Splatting**; 2D lip-sync methods, SyncTalk_2D and leakage (KeySync, LatentSync) added; gap table made accurate |
| Ch.3 | Pipeline is Gemini 3.1 Flash Live (speech-to-speech); **model selection** (Table 3.1) and **the four performance gaps found in the SyncTalk_2D model** (Table 3.2) added; **the unsupported "200–300 ms" latency claim removed**; phoneme-to-viseme table |
| Ch.4 | Mask fix, jaw-row ablation, Bangla phonemes, comparison with all six published checkpoints, **leakage in published systems (Bangla + HDTF)**, speed, XLS-R result |
| Ch.5 | "BanglaTalk3D" and 3D rendering removed; consent and corpus-licensing statements accurate; RAM 8 GB; cost table updated |
| Ch.6 | Written (was template text): proposed-vs-delivered table, limitations, **6.3.1 journal paper in progress**, system extensions |
| References | Every entry checked against where it was actually published (conference / journal, pages, DOI); arXiv kept only where no peer-reviewed version exists (DFA-NeRF, LatentSync, MuseTalk) |

## The numbers — one value per claim, used everywhere

| Claim | Value |
|---|---|
| Model name | **Alapon** — built on SyncTalk_2D; checkpoint `SyncTalk_2D/checkpoint/alapon/59.pth` (formerly `final_v2`) |
| Model size | 49 MB (Wav2Lip 436, IP-LAP 475, MuseTalk 3,400, LatentSync 5,072) |
| Speed on RTX 3050 laptop | 30 fps (33 ms per frame), 0.3 GB GPU memory; ~28 fps in the live system |
| Response time | 1.5–2 s (end-of-turn wait 0.55 s + Gemini ~0.5 s + avatar ~1.0 s, overlapping) |
| Mouth moving during silence | 41% → 0.1% |
| Reconstruction | PSNR 29.8 → 33.1 dB, SSIM 0.88 → 0.91, MAE 0.021 → 0.014 |
| Lip-sync, Bangla test clip (LSE-C) | real 5.14, Alapon 5.11 |
| Bangla phonemes, open ÷ closed | Table 4.3 (test clip): real 4.0×, **Alapon 3.8×**, before fix 3.0× · Table 4.4 (six unseen speakers): **Alapon 2.3×**, before fix 1.7× |
| Leakage in published systems | 5 of 6 leak ≥20% on at least one dataset; only Wav2Lip non-GAN low on both (0.8% / 8%) |
| Metric above real video (HDTF) | 4 of 6 above real on LSE-C; 3 significantly (Wav2Lip, Wav2Lip GAN, LatentSync; paired t-test p < 0.05) |
| Leakage depends on speaker | rank correlation +0.64 over 36 speaker–system pairs, p < 0.001 |
| XLS-R | lip-sync expert separated sync ~half as well as default (0.27 vs 0.50); not used |

## Before submitting

Search the file for **TODO** — 4 items, plus the figures. None may remain:

| Where | What |
|---|---|
| Table 3.3 | Have someone confident in Bangla phonetics check the viseme table |
| Section 5.1.4 | Confirm the `redwan` recording is of Sheikh Redwanul Islam, with consent |
| Table 5.1 | Actual Colab spend; Gemini API cost (free tier or amount) |

**Figures** are marked `[Figure x.y: … — insert image]`. Reuse the FYDP-1 images,
except **3.1, 3.2 and 3.5**, which must be redrawn to match Section 3.2.

**When the clean retrain finishes:** replace the Alapon rows of Tables 4.1, 4.3
and 4.5 with the new numbers and delete the "Training included the test frames"
limitation (6.2) and the caveat in 4.2.

## Converting to Word or PDF

```bash
pandoc FYDP_Final_Report.md -o FYDP_Final_Report.docx
```

Bangla letters and the IPA symbols in Table 3.3 need a font that has them
(for PDF via XeLaTeX, for example `-V mainfont="Noto Serif Bengali"`).

## Where every number comes from

For the viva, when someone asks "where does that figure come from?"

| Number in the report | Source |
|---|---|
| Silence 41% → 0.1%; Table 4.5 (Bangla comparison) | `FYDP/benchmarks/20260917_stage2/redwan_scores.csv` (Alapon's rows under its old name `synctalk2d_final_v2`); silence videos for Wav2Lip non-GAN, MuseTalk 1.0, IP-LAP in `videos_silence_redwan/` |
| Table 4.2 (visible jaw rows) | `FYDP/ablation_mask/results_dose_response.csv` |
| Table 4.1 PSNR / SSIM / MAE after fix | `SyncTalk_2D/evaluation/runs/20260818_005707_redwan_59_reconstruction_test/` |
| Table 4.1 PSNR / SSIM / MAE before fix | `SyncTalk_2D/evaluation/runs/20260710_192744_redwan_19_reconstruction_test/` |
| Tables 4.3 and 4.4 (Bangla phonemes), 80 ms timing shift | `python benchmark/phoneme_eval.py` — reproduces both from `benchmark/results/phoneme/` |
| Table 4.6 (HDTF) | `FYDP/benchmarks/20260917_stage2/hdtf_scores.csv` |
| Paired t-tests, +0.64 correlation, dose-response trend | `docs/research/proposal.md` §4.3–4.5 |
| Table 3.1 sizes and T4 times | checkpoint files (Colab, 25 Sep); `FYDP/benchmarks/20260913_redwan_test/README.md` |
| Table 3.2 gaps; 91% of mouth motion | `docs/research/proposal.md` §2 |
| 30 fps, 33 ms, 0.3 GB | render benchmark on the RTX 3050, 24 Sep — **move `bench_3050.py` into `benchmark/`** |
| Response 1.5–2 s and its parts; reply time 8 → 66 s on the old model | root `README.md` (Measured latency); `agent/agent_bangla.py` |
| XLS-R: separation 0.27 vs 0.50; silence 2.03 vs 1.12 | `syncnet_ckpt/redwan_ssl/train_log.csv` and the Colab training run of the three arms |
| Corpus 2.78 h, 24 speakers, 28 videos | local `research/corpus/` |
| Laptop: Ryzen 7 5800H, 8 GB RAM, RTX 3050 4 GB | measured on the machine, 24 Sep |
