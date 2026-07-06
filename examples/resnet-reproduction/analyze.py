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


def extract_int(pattern, text):
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def analyze(log_dir):
    log_dir = Path(log_dir)
    time_text = read_text(log_dir / "time.log")
    ckpt_text = read_text(log_dir / "ckpt.log")

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
        "reconciled": reconciled,
    }


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def print_table(result):
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
        ("reconciled", result["reconciled"]),
    ]
    print(f"Checkpoint analysis for {result['log_dir']}")
    for key, value in rows:
        print(f"{key:40s} {fmt(value)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze PhOS ResNet Figure 11 reproduction logs.")
    parser.add_argument("log_dir", nargs="?", default="./log/moti-ckpt/phos-trans-resnet")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="return non-zero when required metrics are missing")
    return parser.parse_args()


def main():
    args = parse_args()
    result = analyze(args.log_dir)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_table(result)

    if args.strict:
        required = [
            result["raw_dump_time_s"],
            result["adjusted_checkpoint_time_s"],
            result["data_time_s"],
            result["control_time_s"],
            result["buffer_checkpoint_size_bytes"],
        ]
        if any(value is None for value in required) or result["reconciled"] is not True:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

