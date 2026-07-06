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

if [ "$SKIP_WARMUP" != "1" ]; then
    python3 train_resnet.py --freq 0 --max-iters "$WARMUP_ITERS" --no-checkpoint-signal --no-wait-after-signal
fi

python3 runner.py --freq "$FREQ" --repeats "$REPEATS"
python3 analyze.py ./log/moti-ckpt/phos-trans-resnet

