# Benchmark Results

All results for **Alapon**, the real-time Bangla avatar model (built on
SyncTalk_2D), in one place. Every number here
matches the final report; tables marked **(report)** appear there too.

**Test clip:** the last 772 frames (31 s) of the Bangla recording `redwan`.
**Laptop:** AMD Ryzen 7 5800H, 8 GB RAM, NVIDIA RTX 3050 Laptop GPU (4 GB).
**All other systems were run by us**, on the same clip, with the same scoring code.

| Metric | Meaning | Better |
|---|---|---|
| LSE-D | Lip-sync distance (original SyncNet) | Lower |
| LSE-C | Lip-sync confidence (original SyncNet) | Close to the real-video score |
| Mouth moving during silence | % of frames with the mouth open when the audio is silent | Lower |
| Articulation | How often the mouth opens, relative to the real speaker (1.00 = same) | Close to 1.00 |
| Open ÷ closed | Mouth opening on আ অ divided by mouth opening on প ফ ব ভ ম | Close to the real speaker |

> **Caveat for Alapon.** Alapon was trained before we found that the training
> data included the test frames. This affects its rows on the Bangla test clip; the
> other systems never saw this video. The six-new-speaker test (Table 6) is not
> affected.

---

## 1. Comparison on the Bangla test clip **(report, Table 4.5)**

| System | Size | LSE-D ↓ | LSE-C | Mouth moving during silence ↓ | Articulation |
|---|--:|:-:|:-:|:-:|:-:|
| Real video | — | 7.39 | 5.14 | — | 1.00 |
| **Alapon (ours)** | **49 MB** | 7.30 | 5.11 | **0.1%** | 0.94 |
| SyncTalk_2D, before fix | 49 MB | 7.32 | 5.16 | 41% | 1.05 |
| Wav2Lip (GAN) | 436 MB | 7.26 | 6.08 | 46% | 1.21 |
| Wav2Lip (non-GAN) | 436 MB | 6.83 | 6.44 | 0.8% | 1.05 |
| LatentSync 1.5 | 5,072 MB | 7.29 | 5.09 | 36% | 0.79 |
| MuseTalk 1.5 | 3,400 MB | 7.86 | 4.90 | 0.4% | 0.53 |
| MuseTalk 1.0 | 3,400 MB | 7.72 | 4.67 | 0.1% | 0.22 |
| IP-LAP | 475 MB | 8.71 | 4.36 | 20% | 0.76 |

Real video is not a model, so it has no size and cannot be re-run with
the sound removed. Size is the checkpoint file (IP-LAP: its two checkpoints together).

**Reading it:**
- Alapon is closer to the real-video LSE-C than any published system, and has the
  least mouth movement during silence (0.1%, level with MuseTalk 1.0).
- Both Wav2Lip checkpoints score **above** real video — they were trained against
  the same SyncNet that scores them.
- MuseTalk barely moves during silence, but also barely moves during speech
  (articulation 0.53 and 0.22). Alapon is equally still in silence while moving
  almost as much as the real speaker when talking (0.94).
- The two Wav2Lip checkpoints behave very differently in silence: the GAN one
  moves its mouth in 46% of frames, the non-GAN one in 0.8%. The same gap appears
  on HDTF (Table 7).

---

## 2. The mouth-mask fix **(report, Table 4.1)**

| | Before fix | After fix (Alapon) |
|---|:-:|:-:|
| Mouth moving during silence ↓ | 41% | **0.1%** |
| PSNR ↑ | 29.8 dB | **33.1 dB** |
| SSIM ↑ | 0.88 | **0.91** |
| MAE ↓ | 0.021 | **0.014** |
| LSE-C (real video: 5.14) | 5.16 | 5.11 |
| Articulation | 1.05 | 0.94 |

LSE-C barely changes, so the standard lip-sync metric alone could not have found
this problem. PSNR, SSIM and MAE share the caveat above.

---

## 3. Visible jaw rows **(report, Table 4.2)**

Six models trained identically except for how many rows of the jaw stay visible.

| Visible jaw rows | 10 (original) | 8 | 6 | 4 | 2 | **0 (fixed)** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Mouth moving during silence | 39% | 36% | 26% | 33% | 36% | **8%** |
| LSE-C | 5.07 | 5.09 | 4.88 | 4.98 | 5.14 | 4.93 |

Only covering the jaw completely stops the copying. These six were trained without
the lip-sync loss; Alapon, which adds it, reaches 0.1%.

---

## 4. Speed and size on the laptop **(report, Tables 4.7 and 3.1)**

| | Result | Requirement |
|---|:-:|:-:|
| Model size | 49 MB | — |
| Rendering speed | 30 fps (33 ms per frame) | 25 fps |
| GPU memory | 0.3 GB of 4 GB | — |
| Response time (user stops speaking → avatar answers) | 1.5–2 s | 1–2 s |

**Model size compared:**

| System | Checkpoint | Relative size |
|---|--:|--:|
| **Alapon** | **49 MB** | **1×** |
| Wav2Lip | 436 MB | 9× |
| IP-LAP | 475 MB | 10× |
| MuseTalk 1.5 / 1.0 | 3,400 MB | 69× |
| LatentSync 1.5 | 5,072 MB | 104× |

**Offline generation of the 31 s test clip** (not the live system — these read and
write video files):

| Where | Alapon | Wav2Lip | MuseTalk 1.5 | LatentSync 1.5 |
|---|:-:|:-:|:-:|:-:|
| Cloud T4 GPU | 2 min 33 s | 9 min 06 s | about 15 min | about 30 min |

On the laptop, the offline script runs at 17 fps because it reads every 1080p frame
from disk and writes the video file; the live system keeps frames in memory and
renders at 30 fps.

---

## 5. Bangla phonemes — test clip **(report, Table 4.3)**

Mouth opening for closed sounds (প ফ ব ভ ম) and open sounds (আ অ). Opening = gap
between the inner lips ÷ face width.

| System | Closed sounds | Open sounds | Open ÷ closed |
|---|:-:|:-:|:-:|
| Real video | 0.011 | 0.045 | 4.0× |
| **Alapon (ours)** | **0.011** | **0.043** | **3.8×** |
| SyncTalk_2D, before fix | 0.015 | 0.045 | 3.0× |
| Wav2Lip (GAN) | 0.018 | 0.053 | 3.0× |
| Wav2Lip (non-GAN) | 0.014 | 0.041 | 3.0× |
| LatentSync 1.5 | 0.010 | 0.031 | 3.1× |
| MuseTalk 1.5 | 0.009 | 0.025 | 2.8× |
| MuseTalk 1.0 | 0.005 | 0.018 | 3.5× |
| IP-LAP | 0.016 | 0.029 | 1.9× |

**Reading it:**
- Alapon is the only system close to the real speaker on **both** columns.
- Wav2Lip opens too wide and does not fully close.
- LatentSync closes correctly but opens too little.
- MuseTalk barely moves — its ratio looks reasonable only because both numbers are
  tiny.

---

## 6. Bangla phonemes — six new speakers **(report, Table 4.4)**

The face is our speaker's; the audio is from six Bangla speakers in our corpus that
the model had never heard. Open ÷ closed:

| Speaker | Alapon | Before fix |
|---|:-:|:-:|
| 02 | **2.55×** | 1.99× |
| 04 | **1.96×** | 1.49× |
| 08 | **2.01×** | 1.82× |
| 11 | **1.88×** | 1.63× |
| 23 | **3.18×** | 1.76× |
| 24 | **2.76×** | 1.59× |
| **All six** | **2.3×** | **1.7×** |

Alapon is higher for every speaker. The contrast is lower than the real speaker's on
their own voice (4.0×), so Alapon follows new voices less strongly than the
voice it was trained on. Lip width (rounded উ ও against spread ই এ) did not
separate the models, so no claim is made about lip shape.

---

## 7. Public English dataset (HDTF) **(report, Table 4.6)**

Six speakers, one 31 s clip each. Averages over the six. Alapon is not included:
it is person-specific and must be trained for each new face.

| System | LSE-D ↓ | LSE-C | Above real video? | Mouth moving during silence ↓ | Articulation |
|---|:-:|:-:|:-:|:-:|:-:|
| Real video | 7.42 | 8.01 | — | — | 1.00 |
| Wav2Lip (GAN) | 6.93 | 8.75 | yes | 24% | 1.03 |
| Wav2Lip (non-GAN) | 6.69 | 8.94 | yes | 8% | 0.98 |
| LatentSync 1.5 | 6.70 | 8.82 | yes | 24% | 0.64 |
| MuseTalk 1.5 | 7.12 | 8.38 | yes | 46% | 0.99 |
| MuseTalk 1.0 | 7.72 | 7.52 | no | 41% | 0.75 |
| IP-LAP | 7.07 | 7.88 | no | 10% | 0.34 |

Four of six published systems score above real video on LSE-C here too. Mouth
movement during silence varies a lot from speaker to speaker (it is higher for
speakers who move their mouths more), so compare systems within this table only.

---

## 8. Bangla audio encoder (XLS-R)

We trained the model with the multilingual XLS-R encoder in place of the default
audio features. It did not produce correct lip movement, so Alapon uses
the default features. No numbers are reported.

---

## Summary

| Result | Number |
|---|---|
| Model size | **49 MB** — LatentSync is 104× larger |
| Speed on a laptop RTX 3050 | **30 fps**, 0.3 GB GPU memory |
| Response time | **1.5–2 s** |
| Mouth moving during silence | **41% → 0.1%** after the fix |
| Reconstruction | PSNR **29.8 → 33.1 dB**, SSIM **0.88 → 0.91** |
| Lip-sync (LSE-C) | **5.11** against real video 5.14 |
| Bangla sounds, test clip (open ÷ closed) | **3.8×** against real 4.0× — closest of all systems |
| Bangla sounds, six new speakers | **2.3×** against 1.7× before the fix, all six speakers |

---

## Where the numbers come from

| Table | Source |
|---|---|
| 1, 2 (LSE, silence, articulation) | `FYDP/benchmarks/20260917_stage2/redwan_scores.csv` (Drive); Alapon's rows are under its old name, `synctalk2d_final_v2`. Silence videos for Wav2Lip (non-GAN), MuseTalk 1.0 and IP-LAP: `videos_silence_redwan/`, made by `scripts/run_silence_redwan.sh` (25 Sep) |
| 1, 4 (sizes) | Checkpoint files measured on Colab, 25 Sep |
| 2 (PSNR, SSIM, MAE) | `SyncTalk_2D/evaluation/runs/20260818_005707_…` and `20260710_192744_…` |
| 3 | `FYDP/ablation_mask/results_dose_response.csv` (Drive) |
| 4 | RTX 3050 measurements, 24 Sep; T4 times from `FYDP/benchmarks/20260913_redwan_test/README.md` |
| 5, 6 | `python benchmark/phoneme_eval.py` — reproduces both from `benchmark/results/phoneme/` |
| 7 | `FYDP/benchmarks/20260917_stage2/hdtf_scores.csv` (Drive) |
