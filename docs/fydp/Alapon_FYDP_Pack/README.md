# Alapon — FYDP pack

Everything made for the final report, poster and defence, in one place.
Group E1 · CSE · United International University.

## 1_Report_LaTeX
The final report in the official UIU LaTeX format.

| File | Use |
|---|---|
| `UIU_FYDP_Final_Report.zip` | Upload to Overleaf as a new project. Set **Menu → Compiler → XeLaTeX** (a `latexmkrc` inside forces it). |
| `latex_source/` | The same files unzipped: `fydp.tex` (main), one `.tex` per chapter, `fydp.bib`, images, figure PDFs |
| `report_notes.md` | What changed from FYDP-1, the remaining TODOs, and where every number comes from |

Still to fill in (bold `[TODO: …]` in the PDF): Bangla phonetics check of Table 3.3,
consent for the `redwan` recording, Colab spend and hours, Gemini API cost.

## 2_Poster
`alapon-poster.html` — the poster design, open it in a browser (keep the PNGs
beside it). Use it as a layout reference and rebuild at A0/A1 in PowerPoint,
Illustrator or Canva with the figures from `4_Figures_for_Poster`.

## 3_Figures_for_Report
Benchmark figures without titles (the report caption carries it): vector PDF + 300 dpi PNG.

| Figure | Shows |
|---|---|
| `fig_silence_bangla` | Mouth moving during silence: 41% → 0.1% |
| `fig_bangla_sounds` | Open ÷ closed Bangla sounds: Alapon 3.8× vs real 4.0× |
| `fig_six_speakers` | Six unseen Bangla voices: 1.7× → 2.3×, better on every one |
| `fig_lsec_vs_real` | The lip-sync score rates some systems above real video |
| `fig_jaw_rows` | Only hiding the whole jaw stops the copying |
| `fig_leak_two_datasets` | The same system leaks differently on different speakers |
| `fig_model_size` | 49 MB against up to 5 GB |
| `fig_corpus_speakers` | The 24-speaker Bangla corpus |

## 4_Figures_for_Poster
The same figures at 300 dpi with a title and one-line takeaway, larger type.

## 5_Diagrams
| File | Shows |
|---|---|
| `fig3_1_system_diagram` | Report Figure 3.1 |
| `fig3_2_sequence_diagram` | Report Figure 3.2 |
| `fig3_5_system_architecture` | Report Figure 3.5 |
| `fig3_6_alapon` | Report Figure 3.6 (model + mask change) |
| `system_architecture` | Detailed poster version: one conversation turn, numbered 1–9 |
| `alapon_model_architecture` | Detailed poster version: the full U-Net with shapes and losses |
| `alapon-architecture.html` | Both detailed diagrams on one page |

Every diagram comes as a 3× PNG and an editable SVG.

## 6_Benchmark_Results
`benchmark_result.md` — every result table, with the source of each number.

## 7_Scripts
`make_figures.py`, `make_corpus_grid.py`, `phoneme_eval.py` — regenerate the
figures (for example after the clean retrain). Run them from the project repository.

## Colours used everywhere
Blue `#2A78D6` = Alapon · Orange `#EB6834` = the leak / original mask · Grey = everything else.

## Speaker numbers
Corpus speakers are numbered 01–24 in order. The raw folders skip `speaker17`
(failed the quality filter), so data folders `speaker23` / `speaker24` are
Speakers 22 / 23 in every figure and table.

## Before using the corpus grid publicly
The corpus faces come from public recordings whose speakers did not individually
consent. Keep `fig_corpus_speakers` to the report, or use a blurred / mouth-only
version on the poster.
