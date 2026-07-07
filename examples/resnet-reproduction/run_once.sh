#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

if [ -f /etc/profile ]; then
    source /etc/profile
fi

FREQ=${FREQ:-10}
REPEATS=${REPEATS:-1}
WARMUP_ITERS=${WARMUP_ITERS:-2}
SKIP_WARMUP=${SKIP_WARMUP:-0}
METHOD=${METHOD:-both}
CUDA_CONTROL_TIME_S=${CUDA_CONTROL_TIME_S:-}
SKIP_RESTORE=${SKIP_RESTORE:-0}

RESTORE_ARGS=()
if [ "$SKIP_RESTORE" = "1" ]; then
    RESTORE_ARGS+=(--skip-restore)
fi

if [ "$SKIP_WARMUP" != "1" ]; then
    python3 train_resnet.py --freq 0 --max-iters "$WARMUP_ITERS" --no-checkpoint-signal --no-wait-after-signal
fi

if [ "$METHOD" = "phos" ] || [ "$METHOD" = "both" ]; then
    python3 runner.py --method phos --freq "$FREQ" --repeats "$REPEATS" "${RESTORE_ARGS[@]}"
    python3 analyze.py ./log/moti-ckpt/phos-trans-resnet --method phos
fi

if [ "$METHOD" = "cuda" ] || [ "$METHOD" = "both" ]; then
    python3 runner.py --method cuda --freq "$FREQ" --repeats "$REPEATS" "${RESTORE_ARGS[@]}"
    if [ -n "$CUDA_CONTROL_TIME_S" ]; then
        python3 analyze.py ./log/moti-ckpt/cuda-trans-resnet --method cuda --cuda-control-time-s "$CUDA_CONTROL_TIME_S"
    else
        python3 analyze.py ./log/moti-ckpt/cuda-trans-resnet --method cuda
    fi
fi

if [ "$METHOD" = "cuda-gpu" ]; then
    python3 runner.py --method cuda-gpu --freq "$FREQ" --repeats "$REPEATS"
    python3 analyze.py ./log/moti-ckpt/cuda-gpu-trans-resnet --method cuda-gpu
fi

if [ "$METHOD" = "both" ]; then
    if [ -n "$CUDA_CONTROL_TIME_S" ]; then
        python3 compare.py --cuda-control-time-s "$CUDA_CONTROL_TIME_S"
    else
        python3 compare.py
    fi
fi
