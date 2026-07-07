#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import re
import sys


FLOAT_RE = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"


def read_text(path):
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


def extract_float(pattern, text):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def extract_last_float(pattern, text):
    matches = re.findall(pattern, text)
    return float(matches[-1]) if matches else None


def extract_floats(pattern, text):
    return [float(value) for value in re.findall(pattern, text)]


def extract_int(pattern, text):
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def extract_last_int(pattern, text):
    matches = re.findall(pattern, text)
    return int(matches[-1]) if matches else None


def infer_method(log_dir, time_text, ckpt_text):
    name = Path(log_dir).name
    if name.startswith("cuda-gpu-"):
        return "cuda-gpu"
    if name.startswith("cuda-") or "criu dump time" in time_text:
        return "cuda"
    if "gpu checkpoint time" in time_text or "cuda-checkpoint" in ckpt_text:
        return "cuda-gpu"
    return "phos"


def analyze_phos(log_dir, time_text, ckpt_text):
    raw_dump_time = extract_float(r"process dump time:\s*" + FLOAT_RE + r"\s*s", time_text)
    process_checkpoint_size = extract_int(r"process checkpoint size:\s*(\d+)\s*bytes", time_text)
    buffer_checkpoint_size = extract_int(r"buffer checkpoint size:\s*(\d+)\s*bytes", time_text)
    buffer_checkpoint_time = extract_float(r"buffer checkpoint time:\s*" + FLOAT_RE + r"\s*seconds", time_text)
    buffer_checkpoint_memcpy_time = extract_float(
        r"buffer checkpoint memcpy time:\s*" + FLOAT_RE + r"\s*seconds",
        time_text,
    )
    buffer_checkpoint_bandwidth = extract_float(r"buffer checkpoint bandwidth:\s*" + FLOAT_RE + r"\s*GB/s", time_text)
    buffer_checkpoint_memcpy_bandwidth = extract_float(
        r"buffer checkpoint memcpy bandwidth:\s*" + FLOAT_RE + r"\s*GB/s",
        time_text,
    )
    raw_restore_time = extract_float(r"process restore time:\s*" + FLOAT_RE + r"\s*s", time_text)
    buffer_restore_size = extract_last_int(r"buffer restore size:\s*(\d+)\s*bytes", time_text)
    buffer_restore_time = extract_last_float(r"buffer restore time:\s*" + FLOAT_RE + r"\s*seconds", time_text)
    buffer_restore_memcpy_time = extract_last_float(
        r"buffer restore memcpy time:\s*" + FLOAT_RE + r"\s*seconds",
        time_text,
    )
    buffer_restore_bandwidth = extract_last_float(r"buffer restore bandwidth:\s*" + FLOAT_RE + r"\s*GB/s", time_text)
    buffer_restore_memcpy_bandwidth = extract_last_float(
        r"buffer restore memcpy bandwidth:\s*" + FLOAT_RE + r"\s*GB/s",
        time_text,
    )
    freezing_us = extract_int(r"Freezing time:\s*(\d+)\s*us", ckpt_text)
    frozen_us = extract_int(r"Frozen time:\s*(\d+)\s*us", ckpt_text)

    freeze_total_s = None
    if freezing_us is not None and frozen_us is not None:
        freeze_total_s = (freezing_us + frozen_us) / 1000000.0

    adjusted_checkpoint_time = None
    if raw_dump_time is not None:
        adjusted_checkpoint_time = raw_dump_time - (freeze_total_s or 0.0)

    control_time = None
    if adjusted_checkpoint_time is not None and buffer_checkpoint_time is not None:
        control_time = adjusted_checkpoint_time - buffer_checkpoint_time

    reconciled = None
    if adjusted_checkpoint_time is not None and buffer_checkpoint_time is not None and control_time is not None:
        reconciled = abs((buffer_checkpoint_time + control_time) - adjusted_checkpoint_time) < 1e-6

    return {
        "method": "phos",
        "log_dir": str(log_dir),
        "raw_dump_time_s": raw_dump_time,
        "freezing_time_us": freezing_us,
        "frozen_time_us": frozen_us,
        "freeze_total_s": freeze_total_s,
        "adjusted_checkpoint_time_s": adjusted_checkpoint_time,
        "data_time_s": buffer_checkpoint_time,
        "control_time_s": control_time,
        "buffer_checkpoint_memcpy_time_s": buffer_checkpoint_memcpy_time,
        "process_checkpoint_size_bytes": process_checkpoint_size,
        "buffer_checkpoint_size_bytes": buffer_checkpoint_size,
        "buffer_checkpoint_bandwidth_gib_s": buffer_checkpoint_bandwidth,
        "buffer_checkpoint_memcpy_bandwidth_gib_s": buffer_checkpoint_memcpy_bandwidth,
        "raw_restore_time_s": raw_restore_time,
        "buffer_restore_size_bytes": buffer_restore_size,
        "buffer_restore_time_s": buffer_restore_time,
        "buffer_restore_memcpy_time_s": buffer_restore_memcpy_time,
        "buffer_restore_bandwidth_gib_s": buffer_restore_bandwidth,
        "buffer_restore_memcpy_bandwidth_gib_s": buffer_restore_memcpy_bandwidth,
        "reconciled": reconciled,
    }


def analyze_cuda(log_dir, time_text, ckpt_text, method="cuda", cuda_control_time_s=None):
    gpu_checkpoint_times = extract_floats(
        r"process\s+\d+\s+gpu checkpoint time:\s*" + FLOAT_RE + r"\s*s",
        time_text,
    )
    gpu_checkpoint_time = extract_float(r"cuda checkpoint aggregate time:\s*" + FLOAT_RE + r"\s*s", time_text)
    if gpu_checkpoint_time is None and gpu_checkpoint_times:
        gpu_checkpoint_time = sum(gpu_checkpoint_times)

    criu_dump_time = extract_float(r"criu dump time:\s*" + FLOAT_RE + r"\s*s", time_text)
    checkpoint_total = extract_float(r"cuda-criu checkpoint total time:\s*" + FLOAT_RE + r"\s*s", time_text)
    if checkpoint_total is None:
        checkpoint_total = gpu_checkpoint_time

    gpu_restore_times = extract_floats(
        r"process\s+\d+\s+gpu restore time:\s*" + FLOAT_RE + r"\s*s",
        time_text,
    )
    gpu_restore_time = extract_float(r"cuda restore aggregate time:\s*" + FLOAT_RE + r"\s*s", time_text)
    if gpu_restore_time is None and gpu_restore_times:
        gpu_restore_time = sum(gpu_restore_times)
    criu_restore_time = extract_float(r"criu restore time:\s*" + FLOAT_RE + r"\s*s", time_text)
    restore_total = extract_float(r"cuda-criu restore total time:\s*" + FLOAT_RE + r"\s*s", time_text)
    if restore_total is None:
        restore_total = gpu_restore_time

    gpu_memory_before = extract_int(r"cuda checkpoint gpu memory before total:\s*(\d+)\s*bytes", time_text)
    gpu_memory_after = extract_int(r"cuda checkpoint gpu memory after total:\s*(\d+)\s*bytes", time_text)
    aggregate_bandwidth = extract_float(r"cuda checkpoint aggregate bandwidth:\s*" + FLOAT_RE + r"\s*GB/s", time_text)
    process_checkpoint_size = extract_int(r"process checkpoint size:\s*(\d+)\s*bytes", time_text)

    cuda_data_time = None
    if checkpoint_total is not None and cuda_control_time_s is not None:
        cuda_data_time = checkpoint_total - cuda_control_time_s

    cuda_data_bandwidth = None
    if gpu_memory_before is not None and cuda_data_time is not None and cuda_data_time > 0:
        cuda_data_bandwidth = gpu_memory_before / (1024.0 * 1024.0 * 1024.0) / cuda_data_time

    reconciled = None
    if checkpoint_total is not None and cuda_control_time_s is not None and cuda_data_time is not None:
        reconciled = abs((cuda_control_time_s + cuda_data_time) - checkpoint_total) < 1e-6
    elif checkpoint_total is not None and gpu_checkpoint_time is not None and criu_dump_time is not None:
        reconciled = abs((gpu_checkpoint_time + criu_dump_time) - checkpoint_total) < 0.1

    return {
        "method": method,
        "log_dir": str(log_dir),
        "cuda_total_checkpoint_time_s": checkpoint_total,
        "cuda_gpu_checkpoint_time_s": gpu_checkpoint_time,
        "criu_dump_time_s": criu_dump_time,
        "cuda_per_pid_checkpoint_times_s": gpu_checkpoint_times,
        "cuda_control_time_s": cuda_control_time_s,
        "cuda_data_time_s": cuda_data_time,
        "cuda_total_restore_time_s": restore_total,
        "cuda_gpu_restore_time_s": gpu_restore_time,
        "criu_restore_time_s": criu_restore_time,
        "gpu_memory_used_before_checkpoint_bytes": gpu_memory_before,
        "gpu_memory_used_after_checkpoint_bytes": gpu_memory_after,
        "process_checkpoint_size_bytes": process_checkpoint_size,
        "cuda_total_bandwidth_gib_s": aggregate_bandwidth,
        "cuda_data_bandwidth_gib_s": cuda_data_bandwidth,
        "cuda_split_source": "provided_control_time" if cuda_control_time_s is not None else "total_only",
        "reconciled": reconciled,
    }


def analyze(log_dir, method="auto", cuda_control_time_s=None):
    log_dir = Path(log_dir)
    time_text = read_text(log_dir / "time.log")
    ckpt_text = read_text(log_dir / "ckpt.log")
    selected_method = infer_method(log_dir, time_text, ckpt_text) if method == "auto" else method
    if selected_method in ("cuda", "cuda-gpu"):
        return analyze_cuda(log_dir, time_text, ckpt_text, method=selected_method, cuda_control_time_s=cuda_control_time_s)
    return analyze_phos(log_dir, time_text, ckpt_text)


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def print_table(result):
    if result["method"] in ("cuda", "cuda-gpu"):
        print_cuda_table(result)
        return

    rows = [
        ("raw_dump_time_s", result["raw_dump_time_s"]),
        ("freeze_total_s", result["freeze_total_s"]),
        ("adjusted_checkpoint_time_s", result["adjusted_checkpoint_time_s"]),
        ("data_time_s", result["data_time_s"]),
        ("control_time_s", result["control_time_s"]),
        ("buffer_checkpoint_memcpy_time_s", result["buffer_checkpoint_memcpy_time_s"]),
        ("buffer_checkpoint_size_bytes", result["buffer_checkpoint_size_bytes"]),
        ("process_checkpoint_size_bytes", result["process_checkpoint_size_bytes"]),
        ("buffer_checkpoint_bandwidth_gib_s", result["buffer_checkpoint_bandwidth_gib_s"]),
        ("buffer_checkpoint_memcpy_bandwidth_gib_s", result["buffer_checkpoint_memcpy_bandwidth_gib_s"]),
        ("raw_restore_time_s", result["raw_restore_time_s"]),
        ("buffer_restore_time_s", result["buffer_restore_time_s"]),
        ("buffer_restore_memcpy_time_s", result["buffer_restore_memcpy_time_s"]),
        ("buffer_restore_size_bytes", result["buffer_restore_size_bytes"]),
        ("buffer_restore_bandwidth_gib_s", result["buffer_restore_bandwidth_gib_s"]),
        ("reconciled", result["reconciled"]),
    ]
    print(f"Checkpoint analysis for {result['log_dir']}")
    for key, value in rows:
        print(f"{key:40s} {fmt(value)}")


def print_cuda_table(result):
    rows = [
        ("cuda_total_checkpoint_time_s", result["cuda_total_checkpoint_time_s"]),
        ("cuda_gpu_checkpoint_time_s", result["cuda_gpu_checkpoint_time_s"]),
        ("criu_dump_time_s", result["criu_dump_time_s"]),
        ("cuda_control_time_s", result["cuda_control_time_s"]),
        ("cuda_data_time_s", result["cuda_data_time_s"]),
        ("cuda_total_restore_time_s", result["cuda_total_restore_time_s"]),
        ("cuda_gpu_restore_time_s", result["cuda_gpu_restore_time_s"]),
        ("criu_restore_time_s", result["criu_restore_time_s"]),
        ("gpu_memory_used_before_checkpoint_bytes", result["gpu_memory_used_before_checkpoint_bytes"]),
        ("gpu_memory_used_after_checkpoint_bytes", result["gpu_memory_used_after_checkpoint_bytes"]),
        ("process_checkpoint_size_bytes", result["process_checkpoint_size_bytes"]),
        ("cuda_total_bandwidth_gib_s", result["cuda_total_bandwidth_gib_s"]),
        ("cuda_data_bandwidth_gib_s", result["cuda_data_bandwidth_gib_s"]),
        ("cuda_split_source", result["cuda_split_source"]),
        ("reconciled", result["reconciled"]),
    ]
    print(f"CUDA checkpoint analysis for {result['log_dir']}")
    for key, value in rows:
        print(f"{key:40s} {fmt(value)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze ResNet Figure 11 reproduction logs.")
    parser.add_argument("log_dir", nargs="?", default="./log/moti-ckpt/phos-trans-resnet")
    parser.add_argument("--method", choices=("auto", "phos", "cuda", "cuda-gpu"), default="auto")
    parser.add_argument(
        "--cuda-control-time-s",
        type=float,
        default=None,
        help="optional control-only cuda time; data_time becomes total-control, matching GCR's split",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="return non-zero when required metrics are missing")
    return parser.parse_args()


def main():
    args = parse_args()
    result = analyze(args.log_dir, method=args.method, cuda_control_time_s=args.cuda_control_time_s)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_table(result)

    if args.strict and result["method"] == "phos":
        required = [
            result["raw_dump_time_s"],
            result["adjusted_checkpoint_time_s"],
            result["data_time_s"],
            result["control_time_s"],
            result["buffer_checkpoint_size_bytes"],
        ]
        if any(value is None for value in required) or result["reconciled"] is not True:
            return 1
    elif args.strict and result["method"] == "cuda":
        required = [
            result["cuda_total_checkpoint_time_s"],
            result["cuda_gpu_checkpoint_time_s"],
            result["criu_dump_time_s"],
            result["gpu_memory_used_before_checkpoint_bytes"],
            result["process_checkpoint_size_bytes"],
        ]
        if any(value is None for value in required):
            return 1
    elif args.strict and result["method"] == "cuda-gpu":
        required = [
            result["cuda_total_checkpoint_time_s"],
            result["gpu_memory_used_before_checkpoint_bytes"],
        ]
        if any(value is None for value in required):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
