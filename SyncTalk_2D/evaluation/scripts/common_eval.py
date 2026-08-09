import csv
import json
import math
import os
import re
import statistics
from datetime import datetime
from pathlib import Path


MODE_AUDIO_FILES = {
    "ave": "aud_ave.npy",
    "hubert": "aud_hu.npy",
    "wenet": "aud_wenet.npy",
    "ssl": "aud_ssl.npy",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def evaluation_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_from_project(path_value, root=None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (root or project_root()) / path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def write_csv(path: Path, rows, fieldnames) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def numeric_sort_key(path_or_name):
    name = Path(path_or_name).stem
    match = re.search(r"-?\d+", name)
    if match:
        return int(match.group(0))
    return name


def list_numbered_files(directory: Path, suffix: str):
    if not directory.exists():
        return []
    # Only epoch-numbered files. Skips last.pth (a resume bundle, not a plain
    # state_dict) and best_val.pth - and avoids sorting str against int, which
    # numeric_sort_key would otherwise produce for non-numeric names.
    files = [p for p in directory.glob(f"*{suffix}") if re.search(r"-?\d+", p.stem)]
    return sorted(files, key=numeric_sort_key)


def audio_feature_path(dataset_dir: Path, mode: str) -> Path:
    if mode not in MODE_AUDIO_FILES:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of {sorted(MODE_AUDIO_FILES)}")
    return dataset_dir / MODE_AUDIO_FILES[mode]


def latest_checkpoint(path_value: str, root=None) -> Path:
    raw = str(path_value)
    if raw.endswith("/latest") or raw.endswith("\\latest"):
        base = resolve_from_project(raw[:-7], root)
        candidates = list_numbered_files(base, ".pth")
        if not candidates:
            raise FileNotFoundError(f"No .pth checkpoints found in {base}")
        return candidates[-1]
    path = resolve_from_project(path_value, root)
    if path.is_dir():
        candidates = list_numbered_files(path, ".pth")
        if not candidates:
            raise FileNotFoundError(f"No .pth checkpoints found in {path}")
        return candidates[-1]
    return path


def summarize(values):
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "median": None, "p95": None, "p99": None}
    ordered = sorted(values)

    def percentile(p):
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * p
        low = math.floor(rank)
        high = math.ceil(rank)
        if low == high:
            return ordered[int(rank)]
        return ordered[low] * (high - rank) + ordered[high] * (rank - low)

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def split_range(start: int, end: int):
    if end < start:
        return []
    return list(range(start, end + 1))


def manifest_split_indices(manifest, split_name: str):
    split = manifest["splits"][split_name]
    return split_range(int(split["start"]), int(split["end"]))


def run_dir(dataset_name: str, checkpoint_path: Path, label: str) -> Path:
    ckpt_name = checkpoint_path.stem if checkpoint_path else "no_checkpoint"
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    return ensure_dir(evaluation_root() / "runs" / f"{timestamp()}_{dataset_name}_{ckpt_name}_{safe_label}")


def add_project_to_path() -> None:
    import sys

    root = str(project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def choose_device(torch_module, requested: str):
    if requested == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(requested)


def reshape_audio_feature(audio_feat, mode: str):
    if mode == "hubert":
        return audio_feat.reshape(32, 32, 32)
    if mode == "wenet":
        return audio_feat.reshape(256, 16, 32)
    if mode == "ave":
        return audio_feat.reshape(32, 16, 16)
    if mode == "ssl":
        # 16 frames x 1024 dims = 16384 = 16*32*32
        return audio_feat.reshape(16, 32, 32)
    raise ValueError(f"Unsupported mode: {mode}")


def get_audio_window(torch_module, features, index: int):
    left = index - 8
    right = index + 8
    pad_left = 0
    pad_right = 0
    if left < 0:
        pad_left = -left
        left = 0
    if right > features.shape[0]:
        pad_right = right - features.shape[0]
        right = features.shape[0]
    auds = torch_module.from_numpy(features[left:right])
    if pad_left > 0:
        auds = torch_module.cat([torch_module.zeros_like(auds[:pad_left]), auds], dim=0)
    if pad_right > 0:
        auds = torch_module.cat([auds, torch_module.zeros_like(auds[:pad_right])], dim=0)
    return auds


def load_landmarks(path: Path, np_module):
    points = []
    with path.open("r", encoding="utf-8") as f:
        for line in f.read().splitlines():
            if line.strip():
                points.append(np_module.array(line.split(" "), dtype=np_module.float32))
    return np_module.array(points, dtype=np_module.int32)


def crop_bounds_from_landmarks(lms):
    xmin = int(lms[1][0])
    ymin = int(lms[52][1])
    xmax = int(lms[31][0])
    width = xmax - xmin
    ymax = ymin + width
    return xmin, ymin, xmax, ymax


def pingpong_indices(total_frames: int, available_count: int):
    if available_count <= 1:
        return [0 for _ in range(total_frames)]
    len_img = available_count - 1
    img_idx = 0
    step_stride = 0
    indices = []
    for _ in range(total_frames):
        if img_idx > len_img - 1:
            step_stride = -1
        if img_idx < 1:
            step_stride = 1
        img_idx += step_stride
        indices.append(max(0, min(img_idx, available_count - 1)))
    return indices


def simple_ssim(cv2_module, np_module, img_a, img_b):
    a = img_a.astype(np_module.float64)
    b = img_b.astype(np_module.float64)
    c1 = 6.5025
    c2 = 58.5225
    kernel = (11, 11)
    sigma = 1.5
    mu_a = cv2_module.GaussianBlur(a, kernel, sigma)
    mu_b = cv2_module.GaussianBlur(b, kernel, sigma)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a_sq = cv2_module.GaussianBlur(a * a, kernel, sigma) - mu_a_sq
    sigma_b_sq = cv2_module.GaussianBlur(b * b, kernel, sigma) - mu_b_sq
    sigma_ab = cv2_module.GaussianBlur(a * b, kernel, sigma) - mu_ab
    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    return float((numerator / denominator).mean())


def psnr_from_mse(mse: float):
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))
