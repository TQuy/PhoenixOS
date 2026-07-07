# ResNet PhOS vs cuda-checkpoint Reproduction

This folder runs the PhoenixOS ResNet C/R comparison with a GCR-style Python orchestration harness.

```bash
# PhOS path
pos_cli --dump --dir /root/ckpt --pid <pid>
pos_cli --restore --dir /root/ckpt

# cuda-checkpoint comparison path, equivalent to examples/resnet/readme.md
bash run_nvcr_ckpt.sh -s true -g
bash run_nvcr_restore.sh -g
```

The harness runs a ResNet152 CIFAR10 training process, waits for its checkpoint signal, and then performs full checkpoint/restore:

- `--method phos`: PhOS GPU checkpoint plus CRIU CPU checkpoint through `pos_cli --dump`, then PhOS/CRIU restore through `pos_cli --restore`.
- `--method cuda`: NVIDIA `cuda-checkpoint --toggle` for GPU checkpoint, `criu dump` for CPU checkpoint, `criu restore --restore-detached` for CPU restore, then `cuda-checkpoint --toggle` for GPU restore.
- `--method cuda-gpu`: old GPU-only cuda-checkpoint timing, kept only for reference.

## Files

- `train_resnet.py`: ResNet152 training workload with the GCR-style `SIGUSR2` checkpoint trigger.
- `runner.py`: runs `--method phos`, `--method cuda`, or `--method cuda-gpu`, performs checkpoint/restore, and writes logs.
- `analyze.py`: parses `time.log`, `ckpt.log`, and `restore.log` into checkpoint and restore latency fields.
- `compare.py`: emits a PhOS-vs-cuda table/CSV/JSON for plotting.
- `run_container.sh`: host-side helper for the PhoenixOS CUDA 11.3 container.
- `build_inside_container.sh`: container-side PhOS build and Python dependency setup.
- `run_once.sh`: warmup plus one measured checkpoint run per selected method.

## Host Setup

From the PhoenixOS repository root:

```bash
bash examples/resnet-reproduction/run_container.sh
sudo docker exec -it phos_resnet_repro /bin/bash
```

The container name can be overridden:

```bash
PHOS_CONTAINER_NAME=my_phos bash examples/resnet-reproduction/run_container.sh
```

## Container Setup

Inside the container:

```bash
bash /root/examples/resnet-reproduction/build_inside_container.sh
cd /root/examples/resnet-reproduction
python3 runner.py --method phos --preflight-only
python3 runner.py --method cuda --preflight-only
```

The build script downloads PhoenixOS assets, performs `bash build.sh -c -3`, then runs `bash build.sh -3 -i`, `source /etc/profile`, and `./pos_build -3 -i` before installing the Python modules used by the harness. It also exposes `/root/third_party/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint` as `/usr/local/bin/cuda-checkpoint` when present. It sources `/etc/profile` again at the end so the final shell state sees the installed PhOS environment.

## Run

Inside the container:

```bash
cd /root/examples/resnet-reproduction
bash run_once.sh
```

The default run uses `METHOD=both`:

- warms up two iterations so CIFAR10 and ResNet weights are cached,
- triggers each measured checkpoint at iteration 10,
- writes PhOS logs to `log/moti-ckpt/phos-trans-resnet`,
- writes cuda-checkpoint+CRIU logs to `log/moti-ckpt/cuda-trans-resnet`,
- prints analyzer output for both and then a two-row comparison table.

Useful overrides:

```bash
SKIP_WARMUP=1 bash run_once.sh
FREQ=20 bash run_once.sh
REPEATS=3 bash run_once.sh
METHOD=phos bash run_once.sh
METHOD=cuda bash run_once.sh
METHOD=cuda-gpu bash run_once.sh
SKIP_RESTORE=1 bash run_once.sh
```

`SKIP_RESTORE=1` is useful when debugging checkpoint-only failures. The default comparison includes restore.

## Output

The key files are:

- `task.log`: training output; should show at least 10 iterations.
- `ckpt.log`: checkpoint command output. For PhOS this is `pos_cli --dump`; for cuda this includes `cuda-checkpoint` and `criu dump`.
- `restore.log`: restore command output. For PhOS this is `pos_cli --restore`; for cuda this includes `criu restore` and `cuda-checkpoint`.
- `time.log`: normalized timing metrics for checkpoint and restore.

Run the analyzer directly with:

```bash
python3 analyze.py ./log/moti-ckpt/phos-trans-resnet --strict
python3 analyze.py ./log/moti-ckpt/cuda-trans-resnet --method cuda --strict
python3 compare.py --csv > comparison.csv
python3 compare.py --plot
python3 compare.py --plot-gpu
```
