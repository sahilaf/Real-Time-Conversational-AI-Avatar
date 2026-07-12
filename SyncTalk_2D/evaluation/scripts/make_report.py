import argparse
from pathlib import Path

from common_eval import evaluation_root, read_json


def fmt(value, digits=4):
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def metric_mean(metrics, key):
    value = metrics.get(key)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def report_section(metrics_path: Path, metrics: dict):
    kind = metrics.get("type", "unknown")
    lines = [
        f"### {metrics_path.parent.name}",
        "",
        f"- Type: `{kind}`",
        f"- Metrics file: `{metrics_path}`",
    ]

    if "dataset_name" in metrics:
        lines.append(f"- Dataset: `{metrics['dataset_name']}`")
    if "mode" in metrics:
        lines.append(f"- Mode: `{metrics['mode']}`")
    if "checkpoint" in metrics:
        lines.append(f"- Checkpoint: `{metrics['checkpoint']}`")
    if "device" in metrics:
        lines.append(f"- Device: `{metrics['device']}`")

    lines.append("")
    lines.append("| Area | Metric | Value |")
    lines.append("| --- | --- | ---: |")

    if kind == "reconstruction_328":
        lines.append(f"| Reconstruction | frames evaluated | {fmt(metrics.get('frames_evaluated'))} |")
        lines.append(f"| Reconstruction | MAE mean | {fmt(metric_mean(metrics, 'mae'))} |")
        lines.append(f"| Reconstruction | MSE mean | {fmt(metric_mean(metrics, 'mse'))} |")
        lines.append(f"| Reconstruction | PSNR mean | {fmt(metric_mean(metrics, 'psnr'))} |")
        lines.append(f"| Reconstruction | SSIM mean | {fmt(metric_mean(metrics, 'ssim'))} |")
    elif kind == "sync_328":
        best = metrics.get("best_offset") or {}
        zero = metrics.get("zero_offset") or {}
        acceptance = metrics.get("acceptance") or {}
        lines.append(f"| Sync | frames read | {fmt(metrics.get('frames_read'))} |")
        lines.append(f"| Sync | zero-offset mean score | {fmt(zero.get('mean_sync_score'))} |")
        lines.append(f"| Sync | best offset | {fmt(best.get('offset'))} frames |")
        lines.append(f"| Sync | best mean score | {fmt(best.get('mean_sync_score'))} |")
        lines.append(f"| Sync | best offset within +/-2 | {fmt(acceptance.get('best_offset_within_2_frames'))} |")
    elif kind == "benchmark_inference_328":
        timing = metrics.get("timing") or {}
        frame_generation = metrics.get("frame_generation") or {}
        lines.append(f"| Performance | frames generated | {fmt(metrics.get('frames_generated'))} |")
        lines.append(f"| Performance | generated FPS excluding encode | {fmt(metrics.get('generated_fps_excluding_encode'))} |")
        lines.append(f"| Performance | real-time factor total | {fmt(metrics.get('real_time_factor_total'))} |")
        lines.append(f"| Performance | audio feature seconds | {fmt(timing.get('audio_feature_seconds'))} |")
        lines.append(f"| Performance | generation seconds | {fmt(timing.get('frame_generation_seconds'))} |")
        lines.append(f"| Performance | encoding seconds | {fmt(timing.get('encoding_seconds'))} |")
        lines.append(f"| Performance | per-frame p50 seconds | {fmt(frame_generation.get('median'))} |")
        lines.append(f"| Performance | per-frame p95 seconds | {fmt(frame_generation.get('p95'))} |")
        lines.append(f"| Performance | per-frame p99 seconds | {fmt(frame_generation.get('p99'))} |")
        lines.append(f"| Performance | peak CUDA memory MB | {fmt(metrics.get('peak_cuda_memory_mb'))} |")
    else:
        for key, value in sorted(metrics.items()):
            if isinstance(value, (str, int, float, bool)) or value is None:
                lines.append(f"| Raw | {key} | {fmt(value)} |")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Create a Markdown evaluation report from run metrics.")
    parser.add_argument("--run-dir", action="append", default=[], help="Run directory containing metrics.json. Can repeat.")
    parser.add_argument("--all", action="store_true", help="Include every evaluation/runs/*/metrics.json.")
    parser.add_argument(
        "--include-sync",
        action="store_true",
        help="Include sync_328 runs. Disabled by default because the local SyncNet evaluator can saturate.",
    )
    parser.add_argument("--output", default=None, help="Defaults to evaluation/reports/evaluation_report.md.")
    args = parser.parse_args()

    metric_paths = []
    if args.all:
        metric_paths.extend(sorted((evaluation_root() / "runs").glob("*/metrics.json")))
    for run_dir in args.run_dir:
        path = Path(run_dir)
        metric_paths.append(path / "metrics.json" if path.is_dir() else path)

    metric_paths = [path for path in metric_paths if path.exists()]
    if not args.include_sync:
        filtered_paths = []
        for path in metric_paths:
            metrics = read_json(path)
            if metrics.get("type") != "sync_328":
                filtered_paths.append(path)
        metric_paths = filtered_paths

    if not metric_paths:
        raise FileNotFoundError(
            "No non-sync metrics.json files found. Pass --run-dir or --all after running reconstruction/benchmark, "
            "or add --include-sync to include diagnostic sync runs."
        )

    sections = []
    for path in metric_paths:
        sections.append(report_section(path, read_json(path)))

    output = Path(args.output) if args.output else evaluation_root() / "reports" / "evaluation_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    content = [
        "# SyncTalk 2D Evaluation Report",
        "",
        "This report was generated from evaluation run outputs.",
        "",
        "SyncNet-based sync runs are excluded by default because the current local evaluator can saturate near 1.0 across offsets.",
        "",
        "## Runs",
        "",
        "\n".join(sections),
    ]
    output.write_text("\n".join(content), encoding="utf-8")
    print(f"Wrote report: {output}")


if __name__ == "__main__":
    main()
