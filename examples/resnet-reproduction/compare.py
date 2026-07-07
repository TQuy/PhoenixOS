#!/usr/bin/env python3
import argparse
import csv
import html
import json
from pathlib import Path
import sys

from analyze import analyze


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PHOS_LOG = SCRIPT_DIR / "log" / "moti-ckpt" / "phos-trans-resnet"
DEFAULT_CUDA_LOG = SCRIPT_DIR / "log" / "moti-ckpt" / "cuda-trans-resnet"
DEFAULT_CUDA_GPU_LOG = SCRIPT_DIR / "log" / "moti-ckpt" / "cuda-gpu-trans-resnet"
DEFAULT_PLOT = SCRIPT_DIR / "comparison.svg"
DEFAULT_GPU_PLOT = SCRIPT_DIR / "comparison_gpu.svg"
DEFAULT_GPU_CR_PLOT = SCRIPT_DIR / "comparison_gpu_cr.svg"


def fmt(value):
    return "" if value is None else value


def build_rows(phos, cuda):
    rows = []
    rows.append(
        {
            "method": "phos",
            "total_checkpoint_time_s": phos["raw_dump_time_s"],
            "gpu_checkpoint_time_s": phos["data_time_s"],
            "cpu_checkpoint_time_s": phos["freeze_total_s"],
            "checkpoint_size_bytes": phos["process_checkpoint_size_bytes"],
            "total_restore_time_s": phos["raw_restore_time_s"],
            "gpu_restore_time_s": phos["buffer_restore_time_s"],
            "cpu_restore_time_s": None,
            "gpu_memory_bytes": phos["buffer_checkpoint_size_bytes"],
            "checkpoint_bandwidth_gib_s": phos["buffer_checkpoint_bandwidth_gib_s"],
            "notes": "phos_raw_dump_includes_cpu_gpu_overlap",
        }
    )

    rows.append(
        {
            "method": "cuda",
            "total_checkpoint_time_s": cuda["cuda_total_checkpoint_time_s"],
            "gpu_checkpoint_time_s": cuda["cuda_gpu_checkpoint_time_s"],
            "cpu_checkpoint_time_s": cuda["criu_dump_time_s"],
            "checkpoint_size_bytes": cuda["process_checkpoint_size_bytes"],
            "total_restore_time_s": cuda["cuda_total_restore_time_s"],
            "gpu_restore_time_s": cuda["cuda_gpu_restore_time_s"],
            "cpu_restore_time_s": cuda["criu_restore_time_s"],
            "gpu_memory_bytes": cuda["gpu_memory_used_before_checkpoint_bytes"],
            "checkpoint_bandwidth_gib_s": cuda["cuda_total_bandwidth_gib_s"],
            "notes": "cuda_checkpoint_plus_criu",
        }
    )
    return rows


def build_gpu_rows(phos, cuda_gpu):
    if cuda_gpu["cuda_gpu_restore_time_s"] is None:
        print(
            "warning: cuda-gpu log has no GPU restore time; run METHOD=cuda-gpu again to collect it",
            file=sys.stderr,
        )

    rows = []
    rows.append(
        {
            "method": "phos",
            "total_checkpoint_time_s": None,
            "gpu_checkpoint_time_s": phos["data_time_s"],
            "cpu_checkpoint_time_s": None,
            "checkpoint_size_bytes": phos["buffer_checkpoint_size_bytes"],
            "total_restore_time_s": None,
            "gpu_restore_time_s": phos["buffer_restore_time_s"],
            "cpu_restore_time_s": None,
            "gpu_memory_bytes": phos["buffer_checkpoint_size_bytes"],
            "checkpoint_bandwidth_gib_s": phos["buffer_checkpoint_bandwidth_gib_s"],
            "notes": "phos_gpu_fields",
        }
    )
    rows.append(
        {
            "method": "cuda",
            "total_checkpoint_time_s": None,
            "gpu_checkpoint_time_s": cuda_gpu["cuda_gpu_checkpoint_time_s"],
            "cpu_checkpoint_time_s": None,
            "checkpoint_size_bytes": None,
            "total_restore_time_s": None,
            "gpu_restore_time_s": cuda_gpu["cuda_gpu_restore_time_s"],
            "cpu_restore_time_s": None,
            "gpu_memory_bytes": cuda_gpu["gpu_memory_used_before_checkpoint_bytes"],
            "checkpoint_bandwidth_gib_s": cuda_gpu["cuda_total_bandwidth_gib_s"],
            "notes": "cuda_gpu_only_log",
        }
    )
    return rows


def build_gpu_checkpoint_rows(phos, cuda_gpu):
    phos_data = phos["data_time_s"]
    phos_control = phos["control_time_s"]
    phos_total = phos["adjusted_checkpoint_time_s"]
    if phos_total is None and phos_data is not None:
        phos_total = phos_data + (phos_control or 0.0)

    cuda_total = cuda_gpu["cuda_gpu_checkpoint_time_s"]
    cuda_control = cuda_gpu["cuda_control_time_s"]
    cuda_data = cuda_gpu["cuda_data_time_s"]
    cuda_notes = "cuda_gpu_total_as_data_no_control_split"
    if cuda_data is None:
        cuda_data = cuda_total
        cuda_control = 0.0 if cuda_total is not None else None
    else:
        cuda_notes = "cuda_gpu_data_control_split"

    return [
        {
            "method": "phos",
            "gpu_checkpoint_total_s": phos_total,
            "gpu_checkpoint_data_s": phos_data,
            "gpu_checkpoint_control_s": phos_control,
            "notes": "phos_adjusted_checkpoint_data_plus_control",
        },
        {
            "method": "cuda",
            "gpu_checkpoint_total_s": cuda_total,
            "gpu_checkpoint_data_s": cuda_data,
            "gpu_checkpoint_control_s": cuda_control,
            "notes": cuda_notes,
        },
    ]


def print_table(rows):
    print(
        "method total_checkpoint_time_s gpu_checkpoint_time_s cpu_checkpoint_time_s "
        "total_restore_time_s gpu_restore_time_s cpu_restore_time_s checkpoint_size_bytes notes"
    )
    for row in rows:
        print(
            f"{row['method']:>6s} "
            f"{fmt(row['total_checkpoint_time_s'])!s:>23s} "
            f"{fmt(row['gpu_checkpoint_time_s'])!s:>21s} "
            f"{fmt(row['cpu_checkpoint_time_s'])!s:>21s} "
            f"{fmt(row['total_restore_time_s'])!s:>20s} "
            f"{fmt(row['gpu_restore_time_s'])!s:>18s} "
            f"{fmt(row['cpu_restore_time_s'])!s:>18s} "
            f"{fmt(row['checkpoint_size_bytes'])!s:>21s} "
            f"{row['notes']}"
        )


def print_gpu_checkpoint_table(rows):
    print("method gpu_checkpoint_total_s gpu_checkpoint_data_s gpu_checkpoint_control_s notes")
    for row in rows:
        print(
            f"{row['method']:>6s} "
            f"{fmt(row['gpu_checkpoint_total_s'])!s:>22s} "
            f"{fmt(row['gpu_checkpoint_data_s'])!s:>21s} "
            f"{fmt(row['gpu_checkpoint_control_s'])!s:>24s} "
            f"{row['notes']}"
        )


def number(value):
    return 0.0 if value is None else float(value)


def svg_text(x, y, text, size=14, anchor="middle", weight="400", fill="#111827"):
    escaped = html.escape(str(text))
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{escaped}</text>'
    )


def write_svg_plot(rows, output_path, metrics, title, subtitle):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "phos": "#2563eb",
        "cuda": "#f97316",
    }

    values = [number(row[key]) for row in rows for _, key in metrics]
    max_value = max(values) if values else 1.0
    y_max = max(1.0, max_value * 1.20)

    width = 920
    height = 520
    left = 92
    right = 44
    top = 92
    bottom = 88
    plot_width = width - left - right
    plot_height = height - top - bottom
    baseline = top + plot_height
    group_width = plot_width / len(metrics)
    bar_width = 82
    bar_gap = 18

    def y_pos(value):
        return baseline - (number(value) / y_max) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 34, title, size=24, weight="700"),
        svg_text(
            width / 2,
            58,
            subtitle,
            size=13,
            fill="#4b5563",
        ),
    ]

    for tick in range(6):
        value = y_max * tick / 5.0
        y = y_pos(value)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        svg.append(svg_text(left - 12, y + 4, f"{value:.1f}", size=12, anchor="end", fill="#4b5563"))

    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{baseline}" stroke="#111827" stroke-width="1.2"/>')
    svg.append(f'<line x1="{left}" y1="{baseline}" x2="{width - right}" y2="{baseline}" stroke="#111827" stroke-width="1.2"/>')
    svg.append(svg_text(24, top + plot_height / 2, "Latency (s)", size=13, fill="#374151"))

    for metric_index, (label, key) in enumerate(metrics):
        group_center = left + group_width * (metric_index + 0.5)
        bars_total_width = len(rows) * bar_width + (len(rows) - 1) * bar_gap
        start_x = group_center - bars_total_width / 2

        for row_index, row in enumerate(rows):
            method = row["method"]
            value = number(row[key])
            x = start_x + row_index * (bar_width + bar_gap)
            y = y_pos(value)
            bar_height = baseline - y
            svg.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" '
                f'rx="3" fill="{colors.get(method, "#6b7280")}"/>'
            )
            svg.append(svg_text(x + bar_width / 2, y - 8, f"{value:.2f}s", size=12, weight="700"))
            svg.append(svg_text(x + bar_width / 2, baseline + 22, method, size=13, fill="#374151"))

        svg.append(svg_text(group_center, baseline + 54, label, size=16, weight="700"))

    legend_x = width - right - 188
    legend_y = 76
    for index, row in enumerate(rows):
        method = row["method"]
        y = legend_y + index * 24
        svg.append(f'<rect x="{legend_x}" y="{y - 12}" width="14" height="14" fill="{colors.get(method, "#6b7280")}"/>')
        svg.append(svg_text(legend_x + 22, y, method, size=13, anchor="start", fill="#374151"))

    svg.append(svg_text(width / 2, height - 18, "Generated by examples/resnet-reproduction/compare.py", size=11, fill="#6b7280"))
    svg.append("</svg>")

    output_path.write_text("\n".join(svg) + "\n")
    return output_path


def write_gpu_checkpoint_svg_plot(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        ("phos", "data"): "#fb923c",
        ("phos", "control"): "#f97316",
        ("cuda", "data"): "#93c5fd",
        ("cuda", "control"): "#2563eb",
    }

    max_value = max((number(row["gpu_checkpoint_total_s"]) for row in rows), default=1.0)
    y_max = max(1.0, max_value * 1.22)

    width = 820
    height = 500
    left = 92
    right = 44
    top = 100
    bottom = 86
    plot_width = width - left - right
    plot_height = height - top - bottom
    baseline = top + plot_height
    group_width = plot_width / len(rows)
    bar_width = 118

    def y_pos(value):
        return baseline - (number(value) / y_max) * plot_height

    cuda_has_split = any(
        row["method"] == "cuda" and number(row["gpu_checkpoint_control_s"]) > 0.0
        for row in rows
    )
    subtitle = (
        "Checkpoint only, ResNet152. Bars are data + control; "
        "CUDA is aggregate cuda-checkpoint time unless --cuda-control-time-s is supplied."
        if not cuda_has_split
        else "Checkpoint only, ResNet152. Bars are split into data and control time."
    )

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 34, "ResNet GPU Checkpoint Latency", size=24, weight="700"),
        svg_text(width / 2, 58, subtitle, size=13, fill="#4b5563"),
    ]

    for tick in range(6):
        value = y_max * tick / 5.0
        y = y_pos(value)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        svg.append(svg_text(left - 12, y + 4, f"{value:.1f}", size=12, anchor="end", fill="#4b5563"))

    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{baseline}" stroke="#111827" stroke-width="1.2"/>')
    svg.append(f'<line x1="{left}" y1="{baseline}" x2="{width - right}" y2="{baseline}" stroke="#111827" stroke-width="1.2"/>')
    svg.append(svg_text(24, top + plot_height / 2, "Latency (s)", size=13, fill="#374151"))

    for row_index, row in enumerate(rows):
        method = row["method"]
        group_center = left + group_width * (row_index + 0.5)
        x = group_center - bar_width / 2
        data = number(row["gpu_checkpoint_data_s"])
        control = number(row["gpu_checkpoint_control_s"])
        total = number(row["gpu_checkpoint_total_s"])

        data_height = (data / y_max) * plot_height
        control_height = (control / y_max) * plot_height
        data_y = baseline - data_height
        svg.append(
            f'<rect x="{x:.1f}" y="{data_y:.1f}" width="{bar_width}" height="{data_height:.1f}" '
            f'rx="3" fill="{colors[(method, "data")]}"/>'
        )
        if control_height > 0.0:
            control_y = data_y - control_height
            svg.append(
                f'<rect x="{x:.1f}" y="{control_y:.1f}" width="{bar_width}" height="{control_height:.1f}" '
                f'rx="3" fill="{colors[(method, "control")]}"/>'
            )
        else:
            control_y = data_y

        svg.append(svg_text(group_center, control_y - 8, f"{total:.2f}s", size=12, weight="700"))
        svg.append(svg_text(group_center, baseline + 26, method, size=15, weight="700", fill="#374151"))
        svg.append(svg_text(group_center, baseline + 50, f"data {data:.2f}s / control {control:.2f}s", size=12, fill="#4b5563"))

    legend_items = [
        ("cuda data", colors[("cuda", "data")]),
        ("cuda control", colors[("cuda", "control")]),
        ("PhOS data", colors[("phos", "data")]),
        ("PhOS control", colors[("phos", "control")]),
    ]
    legend_x = width - right - 188
    legend_y = 86
    for index, (label, color) in enumerate(legend_items):
        y = legend_y + index * 22
        svg.append(f'<rect x="{legend_x}" y="{y - 12}" width="14" height="14" fill="{color}"/>')
        svg.append(svg_text(legend_x + 22, y, label, size=13, anchor="start", fill="#374151"))

    svg.append(svg_text(width / 2, height - 18, "Generated by examples/resnet-reproduction/compare.py", size=11, fill="#6b7280"))
    svg.append("</svg>")

    output_path.write_text("\n".join(svg) + "\n")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Export PhOS vs cuda-checkpoint ResNet comparison metrics.")
    parser.add_argument("--phos-log-dir", default=str(DEFAULT_PHOS_LOG))
    parser.add_argument("--cuda-log-dir", default=str(DEFAULT_CUDA_LOG))
    parser.add_argument(
        "--cuda-gpu-log-dir",
        default=str(DEFAULT_CUDA_GPU_LOG),
        help="cuda-gpu log directory used by --plot-gpu",
    )
    parser.add_argument(
        "--cuda-control-time-s",
        type=float,
        default=None,
        help="accepted for compatibility with older cuda-gpu logs; not needed for full cuda-checkpoint+CRIU",
    )
    parser.add_argument("--csv", action="store_true", help="write CSV to stdout")
    parser.add_argument("--json", action="store_true", help="write JSON to stdout")
    parser.add_argument(
        "--plot",
        nargs="?",
        const=str(DEFAULT_PLOT),
        default=None,
        help="write a dependency-free SVG total-latency bar chart; default path is comparison.svg",
    )
    parser.add_argument(
        "--plot-gpu",
        nargs="?",
        const=str(DEFAULT_GPU_PLOT),
        default=None,
        help="write a dependency-free SVG GPU checkpoint-only chart; default path is comparison_gpu.svg",
    )
    parser.add_argument(
        "--plot-gpu-cr",
        nargs="?",
        const=str(DEFAULT_GPU_CR_PLOT),
        default=None,
        help="write the older GPU checkpoint+restore chart; default path is comparison_gpu_cr.svg",
    )
    parser.add_argument(
        "--plot-mode",
        choices=("total", "gpu", "gpu-cr", "both", "all"),
        default=None,
        help="write total, GPU checkpoint-only, GPU C/R, both total+GPU checkpoint, or all default SVG plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    phos = analyze(args.phos_log_dir, method="phos")
    cuda = analyze(args.cuda_log_dir, method="cuda", cuda_control_time_s=args.cuda_control_time_s)
    rows = build_rows(phos, cuda)
    plot_paths = []
    gpu_checkpoint_rows = None
    gpu_cr_rows = None
    plot_total = args.plot
    plot_gpu = args.plot_gpu
    plot_gpu_cr = args.plot_gpu_cr
    if args.plot_mode in ("total", "both", "all"):
        plot_total = plot_total or str(DEFAULT_PLOT)
    if args.plot_mode in ("gpu", "both", "all"):
        plot_gpu = plot_gpu or str(DEFAULT_GPU_PLOT)
    if args.plot_mode in ("gpu-cr", "all"):
        plot_gpu_cr = plot_gpu_cr or str(DEFAULT_GPU_CR_PLOT)

    if plot_total:
        plot_paths.append(
            write_svg_plot(
                rows,
                plot_total,
                metrics=[
                    ("Checkpoint", "total_checkpoint_time_s"),
                    ("Restore", "total_restore_time_s"),
                ],
                title="ResNet C/R Total Latency",
                subtitle=(
                    "Totals only. PhOS checkpoint overlaps CPU/GPU work; "
                    "cuda-checkpoint path runs GPU checkpoint then CRIU."
                ),
            )
        )
    if plot_gpu:
        cuda_gpu = analyze(args.cuda_gpu_log_dir, method="cuda-gpu", cuda_control_time_s=args.cuda_control_time_s)
        gpu_checkpoint_rows = build_gpu_checkpoint_rows(phos, cuda_gpu)
        plot_paths.append(write_gpu_checkpoint_svg_plot(gpu_checkpoint_rows, plot_gpu))
    if plot_gpu_cr:
        cuda_gpu = analyze(args.cuda_gpu_log_dir, method="cuda-gpu", cuda_control_time_s=args.cuda_control_time_s)
        gpu_cr_rows = build_gpu_rows(phos, cuda_gpu)
        plot_paths.append(
            write_svg_plot(
                gpu_cr_rows,
                plot_gpu_cr,
                metrics=[
                    ("GPU Checkpoint", "gpu_checkpoint_time_s"),
                    ("GPU Restore", "gpu_restore_time_s"),
                ],
                title="ResNet GPU C/R Latency",
                subtitle=(
                    "Uses cuda-gpu-trans-resnet, equivalent to analyzing that log with --method cuda-gpu. "
                    "PhOS restore GPU timing depends on the POS restore timing hook."
                ),
            )
        )

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    else:
        if args.plot_mode == "gpu" and gpu_checkpoint_rows is not None:
            print_gpu_checkpoint_table(gpu_checkpoint_rows)
        elif args.plot_mode == "gpu-cr" and gpu_cr_rows is not None:
            print_table(gpu_cr_rows)
        else:
            print_table(rows)
        for plot_path in plot_paths:
            print(f"wrote plot to {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
