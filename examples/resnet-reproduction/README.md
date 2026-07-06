# ResNet PhOS Figure 11 Reproduction

This folder reproduces the GCR paper's PhOS-style ResNet checkpoint latency path:

```bash
python moti.py phos trans-resnet python dnn-train-trans.py -m resnet -f 10
```

The harness runs a ResNet152 CIFAR10 training process, waits for its checkpoint signal, dumps it through `pos_cli --dump`, and records the same latency split used by GCR's Figure 11 analysis.

## Files

- `train_resnet.py`: ResNet152 training workload with the GCR-style `SIGUSR2` checkpoint trigger.
- `runner.py`: starts the PhOS daemon, launches training with `LD_PRELOAD=cricket-client.so`, performs the dump, and writes logs.
- `analyze.py`: parses `time.log` and `ckpt.log` into total, data, and control checkpoint latency.
- `run_container.sh`: host-side helper for the PhoenixOS CUDA 11.3 container.
- `build_inside_container.sh`: container-side PhOS build and Python dependency setup.
- `run_once.sh`: warmup plus one measured ResNet checkpoint run.

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
python3 runner.py --preflight-only
```

The build script downloads PhoenixOS assets, performs `bash build.sh -c -3`, then runs `bash build.sh -3 -i`, `source /etc/profile`, and `./pos_build -3 -i` before installing the Python modules used by the harness. It sources `/etc/profile` again at the end so the final shell state sees the installed PhOS environment.

## Run

Inside the container:

```bash
cd /root/examples/resnet-reproduction
bash run_once.sh
```

The default run:

- warms up two iterations so CIFAR10 and ResNet weights are cached,
- triggers the measured checkpoint at iteration 10,
- writes logs to `log/moti-ckpt/phos-trans-resnet`,
- prints the parsed Figure 11-style latency split.

Useful overrides:

```bash
SKIP_WARMUP=1 bash run_once.sh
FREQ=20 bash run_once.sh
REPEATS=3 bash run_once.sh
```

## Output

The key files are:

- `task.log`: training output; should show at least 10 iterations.
- `ckpt.log`: `pos_cli --dump` output; should include CRIU `Freezing time` and `Frozen time`.
- `time.log`: runner and PhOS memory timing metrics.

Run the analyzer directly with:

```bash
python3 analyze.py ./log/moti-ckpt/phos-trans-resnet --strict
```

The expected sanity shape is similar to GCR's archived PhOS ResNet run: around `3.91s` raw dump time and `1.23s` buffer checkpoint time on their A100 setup. Treat those values as a rough reference, not a pass/fail threshold.
