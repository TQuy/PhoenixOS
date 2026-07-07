#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

if [ -f /etc/profile ]; then
    set +u
    source /etc/profile
    set -u
fi

FREQ=${FREQ:-10}
REPEATS=${REPEATS:-1}
WARMUP_ITERS=${WARMUP_ITERS:-2}
SKIP_WARMUP=${SKIP_WARMUP:-0}
METHOD=${METHOD:-both}
CUDA_CONTROL_TIME_S=${CUDA_CONTROL_TIME_S:-}
SKIP_RESTORE=${SKIP_RESTORE:-0}
PLOT_MODE=${PLOT_MODE:-none}

usage() {
    cat <<'EOF'
Usage: bash run_once.sh [options]

Options:
  --method phos|cuda|cuda-gpu|both  Workload method(s) to run. Default: both.
  --plot none|total|gpu|gpu-cr|both|all
                                    Generate no plot, total C/R plot, GPU checkpoint plot,
                                    older GPU C/R plot, total+GPU checkpoint, or all plots.
                                    Default: none.
  --freq N                          Checkpoint request frequency. Default: 10.
  --repeats N                       Number of repeats. Default: 1.
  --warmup-iters N                  Warmup iterations. Default: 2.
  --skip-warmup                     Skip the warmup run.
  --skip-restore                    Run checkpoint only.
  -h, --help                        Show this help.

Environment variables with the same names still work: METHOD, PLOT_MODE, FREQ,
REPEATS, WARMUP_ITERS, SKIP_WARMUP, SKIP_RESTORE.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --method)
            METHOD=${2:?missing value for --method}
            shift 2
            ;;
        --plot)
            PLOT_MODE=${2:?missing value for --plot}
            shift 2
            ;;
        --freq)
            FREQ=${2:?missing value for --freq}
            shift 2
            ;;
        --repeats)
            REPEATS=${2:?missing value for --repeats}
            shift 2
            ;;
        --warmup-iters)
            WARMUP_ITERS=${2:?missing value for --warmup-iters}
            shift 2
            ;;
        --skip-warmup)
            SKIP_WARMUP=1
            shift
            ;;
        --skip-restore)
            SKIP_RESTORE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$METHOD" in
    phos|cuda|cuda-gpu|both) ;;
    *)
        echo "Invalid --method: $METHOD" >&2
        exit 1
        ;;
esac

case "$PLOT_MODE" in
    none|total|gpu|gpu-cr|both|all) ;;
    *)
        echo "Invalid --plot: $PLOT_MODE" >&2
        exit 1
        ;;
esac

RESTORE_ARGS=()
if [ "$SKIP_RESTORE" = "1" ]; then
    RESTORE_ARGS+=(--skip-restore)
fi

RUN_PHOS=0
RUN_CUDA=0
RUN_CUDA_GPU=0

case "$METHOD" in
    phos)
        RUN_PHOS=1
        ;;
    cuda)
        RUN_CUDA=1
        ;;
    cuda-gpu)
        RUN_CUDA_GPU=1
        ;;
    both)
        RUN_PHOS=1
        RUN_CUDA=1
        ;;
esac

case "$PLOT_MODE" in
    total)
        RUN_PHOS=1
        RUN_CUDA=1
        ;;
    gpu)
        RUN_PHOS=1
        RUN_CUDA_GPU=1
        ;;
    gpu-cr)
        RUN_PHOS=1
        RUN_CUDA_GPU=1
        ;;
    both)
        RUN_PHOS=1
        RUN_CUDA=1
        RUN_CUDA_GPU=1
        ;;
    all)
        RUN_PHOS=1
        RUN_CUDA=1
        RUN_CUDA_GPU=1
        ;;
esac

if [ "$SKIP_WARMUP" != "1" ]; then
    python3 train_resnet.py --freq 0 --max-iters "$WARMUP_ITERS" --no-checkpoint-signal --no-wait-after-signal
fi

if [ "$RUN_PHOS" = "1" ]; then
    python3 runner.py --method phos --freq "$FREQ" --repeats "$REPEATS" "${RESTORE_ARGS[@]}"
    python3 analyze.py ./log/moti-ckpt/phos-trans-resnet --method phos
fi

if [ "$RUN_CUDA" = "1" ]; then
    python3 runner.py --method cuda --freq "$FREQ" --repeats "$REPEATS" "${RESTORE_ARGS[@]}"
    if [ -n "$CUDA_CONTROL_TIME_S" ]; then
        python3 analyze.py ./log/moti-ckpt/cuda-trans-resnet --method cuda --cuda-control-time-s "$CUDA_CONTROL_TIME_S"
    else
        python3 analyze.py ./log/moti-ckpt/cuda-trans-resnet --method cuda
    fi
fi

if [ "$RUN_CUDA_GPU" = "1" ]; then
    python3 runner.py --method cuda-gpu --freq "$FREQ" --repeats "$REPEATS" --cuda-resume-after-checkpoint
    python3 analyze.py ./log/moti-ckpt/cuda-gpu-trans-resnet --method cuda-gpu
fi

COMPARE_ARGS=()
if [ -n "$CUDA_CONTROL_TIME_S" ]; then
    COMPARE_ARGS+=(--cuda-control-time-s "$CUDA_CONTROL_TIME_S")
fi

if [ "$PLOT_MODE" != "none" ]; then
    python3 compare.py "${COMPARE_ARGS[@]}" --plot-mode "$PLOT_MODE"
elif [ "$METHOD" = "both" ]; then
    if [ -n "$CUDA_CONTROL_TIME_S" ]; then
        python3 compare.py --cuda-control-time-s "$CUDA_CONTROL_TIME_S"
    else
        python3 compare.py
    fi
fi
