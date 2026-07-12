# SyncTalk 2D Evaluation Plan

This folder is the dedicated workspace for SyncTalk_2D evaluation work. Keep
all evaluation scripts, manifests, generated reports, plots, and temporary
outputs here so the core training and inference code stays untouched.

## Current Repository Findings

- Primary model path is the 328-resolution variant:
  `training_328.sh`, `train_328.py`, `datasetsss_328.py`,
  `syncnet_328.py`, and `inference_328.py`.
- Default audio mode is `ave`.
- Training uses:
  - processed frames from `dataset/<name>/full_body_img`
  - landmarks from `dataset/<name>/landmarks`
  - audio features from `dataset/<name>/aud_ave.npy`
  - person-specific SyncNet from `syncnet_ckpt/<name>`
  - generator checkpoints from `checkpoint/<name>`
- Inference writes final videos to `result/`.
- Server paths exist for performance testing:
  `synctalk_server.py` for MJPEG streaming and `avatar_server_ws.py` for
  WebSocket segment streaming.

## Evaluation Goals

1. Measure visual reconstruction quality.
2. Measure inference and streaming performance.
3. Measure temporal stability and mouth motion consistency.
4. Keep lip-sync scoring diagnostic until a reliable independent evaluator is available.
5. Produce repeatable reports that compare checkpoints, datasets, and runtime
   settings.

## Proposed Folder Structure

Create these under `SyncTalk_2D/evaluation` as the next implementation step:

```text
evaluation/
  EVALUATION_PLAN.md
  configs/
  manifests/
  scripts/
  reports/
  runs/
  samples/
```

- `configs/`: YAML or JSON configs for dataset name, checkpoint path, audio
  path, frame ranges, metric toggles, and device settings.
- `manifests/`: train/validation/test frame splits and generated video lists.
- `scripts/`: evaluation runners and metric utilities.
- `reports/`: CSV, JSON, Markdown summaries, and plots.
- `runs/`: per-run raw metric outputs.
- `samples/`: short generated clips or contact sheets for visual inspection.

## Data Split Plan

Use contiguous time ranges, not random individual frames, because adjacent video
frames are highly correlated.

Recommended split for each person dataset:

- Train: first 80 percent of usable frames.
- Validation: next 10 percent.
- Test: final 10 percent.

If the original training already used all frames, still create a held-out
evaluation range and label it clearly as "post-training diagnostic", not a true
unseen test. For proper accuracy reporting, retraining should exclude validation
and test ranges.

Manifest fields:

```json
{
  "dataset_name": "May",
  "fps": 25,
  "mode": "ave",
  "splits": {
    "train": {"start": 0, "end": 999},
    "val": {"start": 1000, "end": 1124},
    "test": {"start": 1125, "end": 1249}
  }
}
```

## Accuracy Metrics

### 1. Reconstruction Quality

Run the model on held-out frames using the matching audio features and compare
the predicted mouth crop to the ground-truth mouth crop.

Metrics:

- MAE / L1 on mouth crop.
- MSE and PSNR.
- SSIM on mouth crop.
- Optional LPIPS if dependency is available.
- Full-frame PSNR/SSIM after pasting the generated mouth region back into the
  source frame.

Why: this directly checks whether the generator predicts the correct mouth
appearance when the target frame is known.

### 2. Lip-Sync Diagnostic

Evaluate generated videos against audio.

Metrics:

- SyncNet cosine similarity and BCE-style sync loss using local
  `syncnet_328.py`.
- Offset sweep from -15 to +15 frames: the best score should occur near zero
  offset. This catches videos that look plausible but are delayed.
- Optional external expert metric such as Wav2Lip SyncNet confidence/distance
  if installed later. Use this for final reporting because the local SyncNet is
  also part of training.

Why diagnostic only: the current local SyncNet can saturate near 1.0 across
many offsets, so it should not be treated as the official accuracy score until
negative-pair validation or an independent evaluator is added.

### 3. Mouth Motion and Landmark Consistency

Detect landmarks on generated frames and compare mouth-region motion against
ground truth where ground truth exists.

Metrics:

- Mouth landmark distance on points around lips.
- Mouth opening curve correlation between generated and ground truth.
- Temporal jitter: frame-to-frame landmark velocity/acceleration variance.
- Failure rate: frames where face or landmarks cannot be detected.

Why: catches unstable mouths, jitter, and silent/overactive lips.

### 4. Visual Artifacts

Measure artifacts around the pasted mouth box.

Metrics:

- Boundary difference around the paste region.
- Face crop blur via Laplacian variance.
- Color drift between generated crop and surrounding face region.
- Optional identity embedding distance if a face recognition model is added.

Why: SyncTalk can score well on sync while still producing visible seams,
blur, or identity drift.

## Performance Metrics

### Offline Inference

Use `inference_328.py` behavior as the baseline.

Metrics:

- Audio feature extraction time.
- Model frame generation time.
- Video encoding time.
- Total wall-clock time.
- Frames per second generated.
- Real-time factor: generated video duration / wall-clock time.
- Peak GPU memory if CUDA is available.
- CPU memory usage if measurable.

### Server/Streaming

Use `synctalk_server.py` and `avatar_server_ws.py` for deployment-like tests.

Metrics:

- First-frame latency.
- Per-frame generation latency: mean, p50, p95, p99.
- Segment processing latency.
- Output FPS stability.
- Dropped/late frames.
- Concurrent session behavior for 1, 2, 4 sessions if hardware allows.

## Evaluation Scripts To Build Next

1. `scripts/create_manifest.py`
   - Reads `dataset/<name>/full_body_img` and `aud_ave.npy`.
   - Validates frame, landmark, and audio feature counts.
   - Writes `manifests/<name>_splits.json`.

2. `scripts/eval_reconstruction_328.py`
   - Loads a checkpoint and test split.
   - Reuses the crop and mask logic from `datasetsss_328.py`.
   - Computes MAE, PSNR, SSIM, optional LPIPS.
   - Writes per-frame CSV and summary JSON.

3. `scripts/eval_sync_328.py`
   - Runs generated videos through SyncNet-style scoring.
   - Performs audio/video offset sweep.
   - Writes sync curves and summary scores.

4. `scripts/benchmark_inference_328.py`
   - Times audio encoding, frame generation, and ffmpeg muxing separately.
   - Reports FPS, real-time factor, and memory usage.

5. `scripts/make_report.py`
   - Aggregates all run outputs into one Markdown report.
   - Includes tables, metric thresholds, and selected sample frames.

## Reporting Format

Each evaluation run should produce:

- `runs/<timestamp>_<dataset>_<checkpoint>/metrics.json`
- `runs/<timestamp>_<dataset>_<checkpoint>/per_frame.csv`
- `runs/<timestamp>_<dataset>_<checkpoint>/sync_offset.csv`
- `reports/<timestamp>_<dataset>_<checkpoint>_report.md`

Minimum summary table:

| Area | Metric | Value | Target |
| --- | --- | ---: | --- |
| Reconstruction | mouth MAE | TBD | lower is better |
| Reconstruction | mouth SSIM | TBD | higher is better |
| Sync | zero-offset score | TBD | higher is better |
| Sync | best offset | TBD | near 0 frames |
| Stability | jitter | TBD | lower is better |
| Performance | generated FPS | TBD | above 25 for real time |
| Performance | first-frame latency | TBD | lower is better |

## Acceptance Criteria

A checkpoint is considered ready for demo only if:

- Generated FPS is at least 25 FPS on the target machine, or the real-time
  server path can buffer without visible stalls.
- Landmark detection failure rate is low enough for reliable review.
- Sample clips pass human inspection for mouth timing, jitter, blur, and paste
  artifacts.
- Reconstruction and performance reports are generated for the same checkpoint.

## Immediate Next Step

Implement `scripts/create_manifest.py` first. It is the foundation for every
other metric because it verifies that frames, landmarks, and audio features are
aligned before model scoring begins.
