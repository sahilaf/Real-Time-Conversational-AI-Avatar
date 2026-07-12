import argparse
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


def process_audio(torch, AudioEncoder, AudDataset, audio_path: Path, device):
    from torch.utils.data import DataLoader

    encoder = AudioEncoder().to(device).eval()
    ckpt = torch.load(project_root() / "model" / "checkpoints" / "audio_visual_encoder.pth", map_location=device)
    encoder.load_state_dict({f"audio_encoder.{k}": v for k, v in ckpt.items()})

    dataset = AudDataset(str(audio_path))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    outputs = []
    with torch.no_grad():
        for mel in loader:
            mel = mel.to(device)
            outputs.append(encoder(mel).detach().cpu())
    if not outputs:
        raise RuntimeError("Audio feature extraction produced no frames.")
    feats = torch.cat(outputs, dim=0)
    first_frame, last_frame = feats[:1], feats[-1:]
    return torch.cat([first_frame, feats, last_frame], dim=0).numpy()


def main():
    parser = argparse.ArgumentParser(description="Evaluate SyncTalk lip-sync score with local SyncNet.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory with full_body_img and landmarks.")
    parser.add_argument("--video-path", required=True, help="Generated mp4 to score.")
    parser.add_argument("--audio-path", required=True, help="Audio wav used for the generated video.")
    parser.add_argument("--syncnet-checkpoint", required=True, help="SyncNet .pth, checkpoint dir, or .../latest.")
    parser.add_argument("--mode", default="ave", choices=["ave", "hubert", "wenet"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--offset-min", type=int, default=-15)
    parser.add_argument("--offset-max", type=int, default=15)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    add_project_to_path()
    import cv2
    import numpy as np
    import torch
    from syncnet_328 import SyncNet_color
    from utils import AudioEncoder, AudDataset

    root = project_root()
    device = choose_device(torch, args.device)
    dataset_dir = resolve_from_project(args.dataset_dir, root)
    video_path = resolve_from_project(args.video_path, root)
    audio_path = resolve_from_project(args.audio_path, root)
    syncnet_checkpoint = latest_checkpoint(args.syncnet_checkpoint, root)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir(args.dataset_name, syncnet_checkpoint, "sync")
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_features = process_audio(torch, AudioEncoder, AudDataset, audio_path, device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open generated video: {video_path}")
    video_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        video_frames.append(frame)
        if args.max_frames and len(video_frames) >= args.max_frames:
            break
    cap.release()

    if not video_frames:
        raise RuntimeError("No frames read from generated video.")

    image_dir = dataset_dir / "full_body_img"
    landmark_dir = dataset_dir / "landmarks"
    available_count = len(list(image_dir.glob("*.jpg")))
    ref_indices = pingpong_indices(len(video_frames), available_count)

    syncnet = SyncNet_color(args.mode).to(device).eval()
    syncnet.load_state_dict(torch.load(syncnet_checkpoint, map_location=device))

    offset_rows = []
    frame_rows = []
    with torch.no_grad():
        for offset in range(args.offset_min, args.offset_max + 1):
            scores = []
            for frame_idx, frame in enumerate(video_frames):
                audio_idx = frame_idx + offset
                if audio_idx < 0 or audio_idx >= audio_features.shape[0]:
                    continue
                ref_idx = ref_indices[frame_idx] + args.start_frame
                ref_idx = max(0, min(ref_idx, available_count - 1))
                lms_path = landmark_dir / f"{ref_idx}.lms"
                if not lms_path.exists():
                    continue
                lms = load_landmarks(lms_path, np)
                xmin, ymin, xmax, ymax = crop_bounds_from_landmarks(lms)
                crop = frame[ymin:ymax, xmin:xmax]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (328, 328), interpolation=cv2.INTER_CUBIC)
                face = crop[4:324, 4:324].copy()
                face_t = torch.from_numpy(face.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(device)

                audio_window = get_audio_window(torch, audio_features, audio_idx)
                audio_t = reshape_audio_feature(audio_window, args.mode).unsqueeze(0).to(device)
                audio_embedding, face_embedding = syncnet(face_t, audio_t)
                score = torch.nn.functional.cosine_similarity(audio_embedding, face_embedding).item()
                scores.append(score)
                if offset == 0:
                    frame_rows.append({"frame": frame_idx, "audio_frame": audio_idx, "sync_score": score})

            summary = summarize(scores)
            offset_rows.append({
                "offset": offset,
                "count": summary["count"],
                "mean_sync_score": summary["mean"],
                "median_sync_score": summary["median"],
                "min_sync_score": summary["min"],
                "max_sync_score": summary["max"],
            })

    valid_offsets = [r for r in offset_rows if r["mean_sync_score"] is not None]
    best = max(valid_offsets, key=lambda r: r["mean_sync_score"]) if valid_offsets else None
    zero = next((r for r in offset_rows if r["offset"] == 0), None)
    metrics = {
        "type": "sync_328",
        "dataset_name": args.dataset_name,
        "mode": args.mode,
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "syncnet_checkpoint": str(syncnet_checkpoint),
        "device": str(device),
        "frames_read": len(video_frames),
        "zero_offset": zero,
        "best_offset": best,
        "acceptance": {
            "best_offset_within_2_frames": bool(best and abs(int(best["offset"])) <= 2)
        },
    }

    write_csv(output_dir / "sync_offset.csv", offset_rows, [
        "offset",
        "count",
        "mean_sync_score",
        "median_sync_score",
        "min_sync_score",
        "max_sync_score",
    ])
    write_csv(output_dir / "per_frame_sync_zero_offset.csv", frame_rows, ["frame", "audio_frame", "sync_score"])
    write_json(output_dir / "metrics.json", metrics)
    print(f"Wrote sync results to: {output_dir}")


if __name__ == "__main__":
    main()
