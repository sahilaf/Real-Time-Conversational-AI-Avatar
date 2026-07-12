# How To Run SyncTalk_2D Evaluation

Run all commands from the `SyncTalk_2D` folder:

```powershell
cd C:\Users\sahil\.codex\worktrees\7464\Fydp_v2\SyncTalk_2D
```

Activate the environment you use for SyncTalk before running these commands.

```powershell
conda activate synctalk_2d
```

Replace `May` with your dataset/person name.

## Step 1: Create The Manifest

This checks frames, landmarks, and audio features, then creates train/val/test
frame ranges.

```powershell
python evaluation\scripts\create_manifest.py --dataset-name May --mode ave
```

Output:

```text
evaluation\manifests\May_splits.json
```

Open this file and check the `issues` section. If there are frame/landmark/audio
count warnings, fix those before trusting the metrics.

## Step 2: Evaluate Reconstruction Accuracy

This tests the generator on the manifest split and compares predicted mouth
crops against ground truth.

Use the latest checkpoint:

```powershell
python evaluation\scripts\eval_reconstruction_328.py --manifest evaluation\manifests\May_splits.json --checkpoint checkpoint\May\latest --split test --device auto
```

Or use a specific checkpoint:

```powershell
python evaluation\scripts\eval_reconstruction_328.py --manifest evaluation\manifests\May_splits.json --checkpoint checkpoint\May\99.pth --split test --device auto
```

Outputs:

```text
evaluation\runs\<timestamp>_May_<checkpoint>_reconstruction_test\metrics.json
evaluation\runs\<timestamp>_May_<checkpoint>_reconstruction_test\per_frame.csv
```

Important metrics:

- `mae.mean`: lower is better.
- `mse.mean`: lower is better.
- `psnr.mean`: higher is better.
- `ssim.mean`: higher is better.

## Step 3: Generate A Video For Sync And Performance Tests

If you already have a generated video in `result\`, skip to Step 4.

Otherwise run the repo inference script manually:

```powershell
python inference_328.py --name May --audio_path demo\talk_hb.wav
```

Expected output location:

```text
result\May_talk_hb_<checkpoint>.mp4
```

## Step 4: Optional Diagnostic Sync Check

For now, treat this as diagnostic only. The local SyncNet evaluator can saturate
near `1.0` across many offsets, so it should not be used as the main Bangla
accuracy score.

This command scores the generated video against the audio using the local
SyncNet model and tests offsets from -15 to +15 frames.

Use latest SyncNet checkpoint:

```powershell
python evaluation\scripts\eval_sync_328.py --dataset-name May --dataset-dir dataset\May --video-path result\May_talk_hb_99.mp4 --audio-path demo\talk_hb.wav --syncnet-checkpoint syncnet_ckpt\May\latest --mode ave --device auto
```

Or use a specific SyncNet checkpoint:

```powershell
python evaluation\scripts\eval_sync_328.py --dataset-name May --dataset-dir dataset\May --video-path result\May_talk_hb_99.mp4 --audio-path demo\talk_hb.wav --syncnet-checkpoint syncnet_ckpt\May\42.pth --mode ave --device auto
```

If the generated video was created with `inference_328.py --start_frame N`,
add the same value here:

```powershell
python evaluation\scripts\eval_sync_328.py --dataset-name May --dataset-dir dataset\May --video-path result\May_talk_hb_99.mp4 --audio-path demo\talk_hb.wav --syncnet-checkpoint syncnet_ckpt\May\latest --mode ave --device auto --start-frame N
```

Outputs:

```text
evaluation\runs\<timestamp>_May_<syncnet_checkpoint>_sync\metrics.json
evaluation\runs\<timestamp>_May_<syncnet_checkpoint>_sync\sync_offset.csv
evaluation\runs\<timestamp>_May_<syncnet_checkpoint>_sync\per_frame_sync_zero_offset.csv
```

Important metrics:

- `zero_offset.mean_sync_score`: higher is better.
- `best_offset.offset`: should be close to `0`.
- `acceptance.best_offset_within_2_frames`: should be `true`.

If the best offset is far from zero, the video may be visually plausible but
delayed or advanced relative to the audio.

## Step 5: Benchmark Offline Inference Performance

This measures audio feature extraction, model frame generation, optional ffmpeg
encoding, FPS, real-time factor, and peak CUDA memory if available.

```powershell
python evaluation\scripts\benchmark_inference_328.py --dataset-name May --checkpoint checkpoint\May\latest --audio-path demo\talk_hb.wav --mode ave --device auto
```

For a quick smoke test you can cap frames manually:

```powershell
python evaluation\scripts\benchmark_inference_328.py --dataset-name May --checkpoint checkpoint\May\latest --audio-path demo\talk_hb.wav --mode ave --device auto --max-frames 100
```

To skip ffmpeg muxing and only measure model generation:

```powershell
python evaluation\scripts\benchmark_inference_328.py --dataset-name May --checkpoint checkpoint\May\latest --audio-path demo\talk_hb.wav --mode ave --device auto --skip-encode
```

Outputs:

```text
evaluation\runs\<timestamp>_May_<checkpoint>_benchmark\metrics.json
evaluation\runs\<timestamp>_May_<checkpoint>_benchmark\per_frame_timing.csv
evaluation\runs\<timestamp>_May_<checkpoint>_benchmark\generated_temp_mjpg.mp4
evaluation\runs\<timestamp>_May_<checkpoint>_benchmark\generated_with_audio.mp4
```

Important metrics:

- `generated_fps_excluding_encode`: should be above `25` for real-time model generation.
- `real_time_factor_total`: above `1.0` means faster than real time overall.
- `frame_generation.mean`: average seconds per generated frame.
- `frame_generation.p95` and `frame_generation.p99`: tail latency for generated frames.
- `peak_cuda_memory_mb`: useful for GPU capacity planning.

## Step 6: Generate A Combined Report

After running reconstruction and benchmark steps, generate one Markdown report
from the reliable run outputs. Sync runs are excluded by default.

```powershell
python evaluation\scripts\make_report.py --all
```

Output:

```text
evaluation\reports\evaluation_report.md
```

You can also report selected runs:

```powershell
python evaluation\scripts\make_report.py --run-dir evaluation\runs\<reconstruction_run> --run-dir evaluation\runs\<sync_run> --run-dir evaluation\runs\<benchmark_run> --output evaluation\reports\May_eval_report.md
```

To include SyncNet diagnostic runs anyway:

```powershell
python evaluation\scripts\make_report.py --all --include-sync
```

## Suggested Evaluation Order

1. `create_manifest.py`
2. `eval_reconstruction_328.py`
3. `benchmark_inference_328.py`
4. `make_report.py`
5. Optional: `inference_328.py` and `eval_sync_328.py` for visual/manual sync review

## Demo Readiness Checklist

A model is ready for demo only if:

- Generated FPS excluding encode is at least 25 FPS on the target machine.
- Reconstruction SSIM is stable across the test split.
- Sample videos do not show obvious mouth jitter, blur, paste artifacts, or lag.
- The report includes reconstruction and performance results for the same checkpoint.
- Sync is checked manually from generated videos until a stronger independent
  sync evaluator is added.
