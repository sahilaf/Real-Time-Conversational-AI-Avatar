import argparse
from pathlib import Path

from common_eval import (
    audio_feature_path,
    evaluation_root,
    list_numbered_files,
    project_root,
    resolve_from_project,
    write_json,
)


def make_split(count: int, train_ratio: float, val_ratio: float):
    train_count = int(count * train_ratio)
    val_count = int(count * val_ratio)
    test_count = max(0, count - train_count - val_count)

    train_end = train_count - 1
    val_start = train_count
    val_end = train_count + val_count - 1
    test_start = train_count + val_count
    test_end = train_count + val_count + test_count - 1

    return {
        "train": {"start": 0, "end": max(-1, train_end), "count": max(0, train_count)},
        "val": {"start": val_start, "end": max(val_start - 1, val_end), "count": max(0, val_count)},
        "test": {"start": test_start, "end": max(test_start - 1, test_end), "count": max(0, test_count)},
    }


def main():
    parser = argparse.ArgumentParser(description="Create a SyncTalk_2D evaluation manifest.")
    parser.add_argument("--dataset-name", required=True, help="Dataset/person name, e.g. May.")
    parser.add_argument("--dataset-dir", default=None, help="Defaults to dataset/<dataset-name>.")
    parser.add_argument("--mode", default="ave", choices=["ave", "hubert", "wenet"])
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--output", default=None, help="Defaults to evaluation/manifests/<name>_splits.json.")
    args = parser.parse_args()

    root = project_root()
    dataset_dir = resolve_from_project(args.dataset_dir or f"dataset/{args.dataset_name}", root)
    image_dir = dataset_dir / "full_body_img"
    landmark_dir = dataset_dir / "landmarks"
    audio_path = audio_feature_path(dataset_dir, args.mode)

    issues = []
    frames = list_numbered_files(image_dir, ".jpg")
    landmarks = list_numbered_files(landmark_dir, ".lms")
    audio_feature_count = None

    if not dataset_dir.exists():
        issues.append(f"Dataset directory does not exist: {dataset_dir}")
    if not image_dir.exists():
        issues.append(f"Frame directory does not exist: {image_dir}")
    if not landmark_dir.exists():
        issues.append(f"Landmark directory does not exist: {landmark_dir}")
    if not audio_path.exists():
        issues.append(f"Audio feature file does not exist: {audio_path}")

    if audio_path.exists():
        import numpy as np

        audio_features = np.load(audio_path)
        audio_feature_count = int(audio_features.shape[0])

    frame_ids = [int(p.stem) for p in frames if p.stem.isdigit()]
    landmark_ids = [int(p.stem) for p in landmarks if p.stem.isdigit()]
    common_ids = sorted(set(frame_ids).intersection(landmark_ids))

    usable_count = len(common_ids)
    if audio_feature_count is not None:
        usable_count = min(usable_count, max(0, audio_feature_count - 1))

    if usable_count <= 0:
        issues.append("No usable aligned frames found.")
    if len(frames) != len(landmarks):
        issues.append(f"Frame/landmark count mismatch: {len(frames)} jpg vs {len(landmarks)} lms.")
    if audio_feature_count is not None and audio_feature_count - 1 != len(frames):
        issues.append(
            f"Audio feature length diagnostic: aud count - 1 = {audio_feature_count - 1}, frame count = {len(frames)}."
        )

    splits = make_split(usable_count, args.train_ratio, args.val_ratio)
    output_path = Path(args.output) if args.output else evaluation_root() / "manifests" / f"{args.dataset_name}_splits.json"
    if not output_path.is_absolute():
        output_path = root / output_path

    manifest = {
        "dataset_name": args.dataset_name,
        "dataset_dir": str(dataset_dir),
        "mode": args.mode,
        "fps": args.fps,
        "created_by": "evaluation/scripts/create_manifest.py",
        "counts": {
            "frames": len(frames),
            "landmarks": len(landmarks),
            "audio_features": audio_feature_count,
            "usable_aligned_frames": usable_count,
        },
        "splits": splits,
        "issues": issues,
    }
    write_json(output_path, manifest)

    print(f"Wrote manifest: {output_path}")
    if issues:
        print("Diagnostics:")
        for issue in issues:
            print(f"- {issue}")


if __name__ == "__main__":
    main()
