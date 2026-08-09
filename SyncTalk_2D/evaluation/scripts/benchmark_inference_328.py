import argparse
import shutil
import subprocess
import time
from pathlib import Path

from common_eval import (
    add_project_to_path,
    choose_device,
    crop_bounds_from_landmarks,
    get_audio_window,
    latest_checkpoint,
    load_landmarks,
    pingpong_indices,
    project_root,
    reshape_audio_feature,
    resolve_from_project,
    run_dir,
    summarize,
    write_csv,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description="Benchmark SyncTalk_2D 328 offline inference.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-dir", default=None, help="Defaults to dataset/<dataset-name>.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth, checkpoint dir, or .../latest.")
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--mode", default="ave", choices=["ave", "hubert", "wenet", "ssl"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--skip-encode", action="store_true", help="Generate temp MJPG only; skip ffmpeg mux.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    add_project_to_path()
    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from unet_328 import Model
    from utils import AudioEncoder, AudDataset

    root = project_root()
    dataset_dir = resolve_from_project(args.dataset_dir or f"dataset/{args.dataset_name}", root)
    audio_path = resolve_from_project(args.audio_path, root)
    checkpoint = latest_checkpoint(args.checkpoint, root)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir(args.dataset_name, checkpoint, "benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(torch, args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    img_dir = dataset_dir / "full_body_img"
    lms_dir = dataset_dir / "landmarks"
    jpgs = sorted(img_dir.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not jpgs:
        raise FileNotFoundError(f"No jpg frames found in {img_dir}")

    example = cv2.imread(str(img_dir / "0.jpg"))
    if example is None:
        raise FileNotFoundError(f"Could not read example frame: {img_dir / '0.jpg'}")
    frame_h, frame_w = example.shape[:2]

    timing = {}
    t0 = time.perf_counter()
    audio_encoder = AudioEncoder().to(device).eval()
    audio_ckpt = torch.load(root / "model" / "checkpoints" / "audio_visual_encoder.pth", map_location=device)
    audio_encoder.load_state_dict({f"audio_encoder.{k}": v for k, v in audio_ckpt.items()})

    audio_dataset = AudDataset(str(audio_path))
    audio_loader = DataLoader(audio_dataset, batch_size=64, shuffle=False)
    outputs = []
    with torch.no_grad():
        for mel in audio_loader:
            mel = mel.to(device)
            outputs.append(audio_encoder(mel).detach().cpu())
    if not outputs:
        raise RuntimeError("Audio feature extraction produced no frames.")
    raw_outputs = torch.cat(outputs, dim=0)
    first_frame, last_frame = raw_outputs[:1], raw_outputs[-1:]
    audio_features = torch.cat([first_frame, raw_outputs, last_frame], dim=0).numpy()
    timing["audio_feature_seconds"] = time.perf_counter() - t0

    total_frames = audio_features.shape[0]
    if args.max_frames and args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    t1 = time.perf_counter()
    model = Model(6, mode=args.mode).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    timing["model_load_seconds"] = time.perf_counter() - t1

    temp_video = output_dir / "generated_temp_mjpg.mp4"
    final_video = output_dir / "generated_with_audio.mp4"
    writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (frame_w, frame_h))
    ref_indices = pingpong_indices(total_frames, len(jpgs))

    per_frame_rows = []
    generation_start = time.perf_counter()
    with torch.no_grad():
        for frame_idx in range(total_frames):
            frame_start = time.perf_counter()
            ref_index = ref_indices[frame_idx] + args.start_frame
            ref_index = max(0, min(ref_index, len(jpgs) - 1))

            img_path = img_dir / f"{ref_index}.jpg"
            lms_path = lms_dir / f"{ref_index}.lms"
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Could not read frame: {img_path}")

            lms = load_landmarks(lms_path, np)
            xmin, ymin, xmax, ymax = crop_bounds_from_landmarks(lms)
            crop = img[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                raise RuntimeError(f"Empty crop for frame {ref_index}")
            crop_h, crop_w = crop.shape[:2]
            crop_resized = cv2.resize(crop, (328, 328), interpolation=cv2.INTER_CUBIC)
            crop_original = crop_resized.copy()
            real = crop_resized[4:324, 4:324].copy()
            masked = real.copy()
            masked = cv2.rectangle(masked, (5, 5), (310, 305), (0, 0, 0), -1)

            real_t = torch.from_numpy(real.transpose(2, 0, 1).astype(np.float32) / 255.0)
            masked_t = torch.from_numpy(masked.transpose(2, 0, 1).astype(np.float32) / 255.0)
            input_t = torch.cat([real_t, masked_t], dim=0).unsqueeze(0).to(device)
            audio_window = get_audio_window(torch, audio_features, frame_idx)
            audio_t = reshape_audio_feature(audio_window, args.mode).unsqueeze(0).to(device)

            pred = model(input_t, audio_t)[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0
            pred = np.array(np.clip(pred, 0, 255), dtype=np.uint8)
            crop_original[4:324, 4:324] = pred
            crop_original = cv2.resize(crop_original, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
            img[ymin:ymax, xmin:xmax] = crop_original
            writer.write(img)

            per_frame_rows.append({
                "frame": frame_idx,
                "reference_frame": ref_index,
                "generation_seconds": time.perf_counter() - frame_start,
            })
    writer.release()
    timing["frame_generation_seconds"] = time.perf_counter() - generation_start

    encode_seconds = None
    if not args.skip_encode:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg was not found. Re-run with --skip-encode or install ffmpeg.")
        encode_start = time.perf_counter()
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(temp_video),
                "-i",
                str(audio_path),
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-crf",
                "20",
                str(final_video),
            ],
            check=True,
        )
        encode_seconds = time.perf_counter() - encode_start
    timing["encoding_seconds"] = encode_seconds

    total_wall = sum(v for v in timing.values() if isinstance(v, (int, float)))
    generated_duration = total_frames / 25.0
    frame_times = [row["generation_seconds"] for row in per_frame_rows]
    metrics = {
        "type": "benchmark_inference_328",
        "dataset_name": args.dataset_name,
        "mode": args.mode,
        "checkpoint": str(checkpoint),
        "audio_path": str(audio_path),
        "device": str(device),
        "frames_generated": total_frames,
        "generated_duration_seconds": generated_duration,
        "timing": timing,
        "frame_generation": summarize(frame_times),
        "generated_fps_excluding_encode": total_frames / timing["frame_generation_seconds"] if timing["frame_generation_seconds"] > 0 else None,
        "real_time_factor_total": generated_duration / total_wall if total_wall > 0 else None,
        "temp_video": str(temp_video),
        "final_video": str(final_video) if final_video.exists() else None,
        "peak_cuda_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if device.type == "cuda"
            else None
        ),
    }

    write_csv(output_dir / "per_frame_timing.csv", per_frame_rows, ["frame", "reference_frame", "generation_seconds"])
    write_json(output_dir / "metrics.json", metrics)
    print(f"Wrote benchmark results to: {output_dir}")


if __name__ == "__main__":
    main()
