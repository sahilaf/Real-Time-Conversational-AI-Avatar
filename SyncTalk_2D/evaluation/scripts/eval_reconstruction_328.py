import argparse
import math
from pathlib import Path

from common_eval import (
    add_project_to_path,
    audio_feature_path,
    choose_device,
    crop_bounds_from_landmarks,
    get_audio_window,
    latest_checkpoint,
    load_landmarks,
    manifest_split_indices,
    psnr_from_mse,
    read_json,
    reshape_audio_feature,
    run_dir,
    simple_ssim,
    summarize,
    write_csv,
    write_json,
)


def load_image_or_raise(cv2, path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def main():
    parser = argparse.ArgumentParser(description="Evaluate 328 SyncTalk reconstruction quality.")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON from create_manifest.py.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth, checkpoint dir, or .../latest.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap for quick manual checks.")
    parser.add_argument("--output-dir", default=None, help="Optional output run directory.")
    args = parser.parse_args()

    add_project_to_path()
    import cv2
    import numpy as np
    import torch
    from unet_328 import Model

    manifest = read_json(Path(args.manifest))
    dataset_dir = Path(manifest["dataset_dir"])
    mode = manifest["mode"]
    checkpoint = latest_checkpoint(args.checkpoint)
    device = choose_device(torch, args.device)

    image_dir = dataset_dir / "full_body_img"
    landmark_dir = dataset_dir / "landmarks"
    audio_features = np.load(audio_feature_path(dataset_dir, mode)).astype(np.float32)

    indices = manifest_split_indices(manifest, args.split)
    if args.max_frames and args.max_frames > 0:
        indices = indices[: args.max_frames]

    output_dir = Path(args.output_dir) if args.output_dir else run_dir(manifest["dataset_name"], checkpoint, f"reconstruction_{args.split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Model(6, mode=mode).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    rows = []
    with torch.no_grad():
        for idx in indices:
            img_path = image_dir / f"{idx}.jpg"
            lms_path = landmark_dir / f"{idx}.lms"
            if not img_path.exists() or not lms_path.exists() or idx >= audio_features.shape[0]:
                continue

            img = load_image_or_raise(cv2, img_path)
            lms = load_landmarks(lms_path, np)
            xmin, ymin, xmax, ymax = crop_bounds_from_landmarks(lms)
            crop = img[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue

            crop_resized = cv2.resize(crop, (328, 328), interpolation=cv2.INTER_CUBIC)
            real = crop_resized[4:324, 4:324].copy()
            masked = real.copy()
            masked = cv2.rectangle(masked, (5, 5), (310, 305), (0, 0, 0), -1)

            real_t = torch.from_numpy(real.transpose(2, 0, 1).astype(np.float32) / 255.0)
            masked_t = torch.from_numpy(masked.transpose(2, 0, 1).astype(np.float32) / 255.0)
            input_t = torch.cat([real_t, masked_t], dim=0).unsqueeze(0).to(device)

            audio_window = get_audio_window(torch, audio_features, idx)
            audio_t = reshape_audio_feature(audio_window, mode).unsqueeze(0).to(device)
            pred = model(input_t, audio_t)[0].detach().cpu().numpy().transpose(1, 2, 0)
            pred = np.clip(pred, 0.0, 1.0)
            real_float = real.astype(np.float32) / 255.0

            diff = pred - real_float
            mae = float(np.mean(np.abs(diff)))
            mse = float(np.mean(diff * diff))
            psnr = psnr_from_mse(mse)
            ssim = simple_ssim(cv2, np, (pred * 255.0).astype(np.uint8), real)

            rows.append({
                "frame": idx,
                "mae": mae,
                "mse": mse,
                "psnr": psnr,
                "ssim": ssim,
            })

    summary = {
        "type": "reconstruction_328",
        "dataset_name": manifest["dataset_name"],
        "split": args.split,
        "mode": mode,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "frames_evaluated": len(rows),
        "mae": summarize([r["mae"] for r in rows]),
        "mse": summarize([r["mse"] for r in rows]),
        "psnr": summarize([r["psnr"] for r in rows if not math.isinf(r["psnr"])]),
        "ssim": summarize([r["ssim"] for r in rows]),
    }

    write_csv(output_dir / "per_frame.csv", rows, ["frame", "mae", "mse", "psnr", "ssim"])
    write_json(output_dir / "metrics.json", summary)
    print(f"Wrote reconstruction results to: {output_dir}")


if __name__ == "__main__":
    main()
