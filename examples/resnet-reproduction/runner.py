#!/usr/bin/env python3
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


try:
    import psutil
except ImportError:
    psutil = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_ROOT = SCRIPT_DIR / "log" / "moti-ckpt"
DEFAULT_WORKLOAD = "phos-trans-resnet"
POS_LOG_PATH = Path("/dev/shm/cr/pos_log")
PARENT_PID_PATH = Path("/dev/shm/cr/parent_pid")
CHILD_PID_PATH = Path("/dev/shm/cr/child_pid")

checkpoint_requested = False


class PosLog(ctypes.Structure):
    _fields_ = [
        ("tot_ckpt_size", ctypes.c_size_t),
        ("tot_ckpt_time", ctypes.c_double),
        ("ckpt_memcpy_time", ctypes.c_double),
        ("tot_restore_size", ctypes.c_size_t),
        ("tot_restore_time", ctypes.c_double),
        ("restore_memcpy_time", ctypes.c_double),
    ]


def checkpoint_signal_handler(signum, frame):
    global checkpoint_requested
    checkpoint_requested = True


def resolve_path(path, base=SCRIPT_DIR):
    path = Path(path)
    return path if path.is_absolute() else base / path


def command_output(cmd):
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def find_cricket_client():
    candidates = [
        Path("/usr/local/lib/cricket-client.so"),
        Path("/usr/lib/cricket-client.so"),
        Path("/usr/lib/x86_64-linux-gnu/cricket-client.so"),
        Path("/root/lib/cricket-client.so"),
    ]

    for item in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if item:
            candidates.append(Path(item) / "cricket-client.so")

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    ldconfig = shutil.which("ldconfig")
    if ldconfig:
        result = command_output([ldconfig, "-p"])
        if result.returncode == 0 and "cricket-client.so" in result.stdout:
            return "cricket-client.so"

    return None


def require_command(name):
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"required command not found in PATH: {name}")
    return path


def require_python_import(module):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"required Python module import failed: {module}\n{result.stderr}")


def require_torchvision_resnet_api():
    code = "from torchvision.models import ResNet152_Weights, resnet152"
    result = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "torchvision does not expose ResNet152_Weights/resnet152; "
            "run build_inside_container.sh or install the compatible torchvision build.\n"
            + result.stderr
        )


def check_cuda():
    code = (
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA is not available'; "
        "print(torch.cuda.get_device_name(0))"
    )
    result = subprocess.run([sys.executable, "-c", code], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"CUDA preflight failed:\n{result.stderr}")
    return result.stdout.strip()


def clear_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def init_control_files():
    POS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    POS_LOG_PATH.write_bytes(bytes(PosLog()))
    PARENT_PID_PATH.write_text(f"{os.getpid()}\n")
    if CHILD_PID_PATH.exists():
        CHILD_PID_PATH.unlink()


def cleanup_control_files():
    for path in (PARENT_PID_PATH, CHILD_PID_PATH):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def ensure_pos_yaml():
    content = 'job_name: "moti-trans-resnet"\ndaemon_addr: "127.0.0.1"\n'
    path = SCRIPT_DIR / "pos.yaml"
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def clear_client_marker():
    try:
        (SCRIPT_DIR / "client_exist.txt").unlink()
    except FileNotFoundError:
        pass


def read_pos_log():
    if not POS_LOG_PATH.exists() or POS_LOG_PATH.stat().st_size < ctypes.sizeof(PosLog):
        return PosLog()

    with POS_LOG_PATH.open("r+b") as f:
        return PosLog.from_buffer_copy(f.read(ctypes.sizeof(PosLog)))


def dir_size(path):
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def append_pos_log_metrics(time_output, ckpt_dir):
    plog = read_pos_log()
    time_output.write(f"process checkpoint size: {dir_size(ckpt_dir)} bytes\n")
    time_output.write(f"buffer checkpoint size: {plog.tot_ckpt_size} bytes\n")
    time_output.write(f"buffer checkpoint time: {plog.tot_ckpt_time} seconds\n")
    time_output.write(f"buffer checkpoint memcpy time: {plog.ckpt_memcpy_time} seconds\n")
    time_output.write(f"buffer restore size: {plog.tot_restore_size} bytes\n")
    time_output.write(f"buffer restore time: {plog.tot_restore_time} seconds\n")
    time_output.write(f"buffer restore memcpy time: {plog.restore_memcpy_time} seconds\n")
    if plog.tot_ckpt_time > 0:
        bw = plog.tot_ckpt_size / (1024.0 * 1024.0 * 1024.0) / plog.tot_ckpt_time
        time_output.write(f"buffer checkpoint bandwidth: {bw} GB/s\n")
    if plog.ckpt_memcpy_time > 0:
        bw = plog.tot_ckpt_size / (1024.0 * 1024.0 * 1024.0) / plog.ckpt_memcpy_time
        time_output.write(f"buffer checkpoint memcpy bandwidth: {bw} GB/s\n")
    if plog.tot_restore_time > 0:
        bw = plog.tot_restore_size / (1024.0 * 1024.0 * 1024.0) / plog.tot_restore_time
        time_output.write(f"buffer restore bandwidth: {bw} GB/s\n")
    if plog.restore_memcpy_time > 0:
        bw = plog.tot_restore_size / (1024.0 * 1024.0 * 1024.0) / plog.restore_memcpy_time
        time_output.write(f"buffer restore memcpy bandwidth: {bw} GB/s\n")
    time_output.flush()


def terminate_process_tree(proc, name, timeout=5):
    if proc is None or proc.poll() is not None:
        return

    if psutil is not None:
        try:
            parent = psutil.Process(proc.pid)
            children = parent.children(recursive=True)
            for child in children:
                child.terminate()
            parent.terminate()
            gone, alive = psutil.wait_procs(children + [parent], timeout=timeout)
            for item in alive:
                item.kill()
            return
        except psutil.NoSuchProcess:
            return

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=timeout)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


def preflight(args):
    require_command("pos_cli")
    require_python_import("psutil")
    require_python_import("pynvml")
    require_python_import("tqdm")
    require_python_import("torch")
    require_python_import("torchvision")
    require_torchvision_resnet_api()

    if find_cricket_client() is None:
        raise RuntimeError("could not find cricket-client.so through common library paths or ldconfig")

    ckpt_dir = resolve_path(args.ckpt_dir, Path("/"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    POS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    init_control_files()
    device = check_cuda()
    cleanup_control_files()
    print(json.dumps({"ok": True, "cuda_device": device, "ckpt_dir": str(ckpt_dir)}, indent=2))


def run_one(args, run_index):
    global checkpoint_requested
    checkpoint_requested = False

    workload = args.workload if args.repeats == 1 else f"{args.workload}-run-{run_index}"
    log_dir = resolve_path(args.log_root) / workload
    clear_directory(log_dir)

    ckpt_dir = Path(args.ckpt_dir)
    clear_directory(ckpt_dir)
    init_control_files()
    ensure_pos_yaml()
    clear_client_marker()
    stdin_path = log_dir / "stdin.txt"
    stdin_path.touch()

    daemon_proc = None
    child_proc = None
    signal.signal(signal.SIGUSR2, checkpoint_signal_handler)

    with (log_dir / "daemon.log").open("w") as daemon_out, \
            (log_dir / "daemon.err").open("w") as daemon_err, \
            (log_dir / "task.log").open("w") as task_out, \
            (log_dir / "task.err").open("w") as task_err, \
            (log_dir / "ckpt.log").open("w") as ckpt_out, \
            (log_dir / "ckpt.err").open("w") as ckpt_err, \
            stdin_path.open("rb") as task_stdin, \
            (log_dir / "time.log").open("w") as time_out:
        try:
            daemon_proc = subprocess.Popen(
                ["pos_cli", "--start", "--target", "daemon"],
                cwd=str(SCRIPT_DIR),
                stdout=daemon_out,
                stderr=daemon_err,
                start_new_session=True,
            )
            time.sleep(args.daemon_start_delay)

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            previous_preload = env.get("LD_PRELOAD")
            env["LD_PRELOAD"] = "cricket-client.so" if not previous_preload else f"cricket-client.so:{previous_preload}"

            child_cmd = [
                sys.executable,
                str(SCRIPT_DIR / "train_resnet.py"),
                "--freq",
                str(args.freq),
                "--udp-port",
                str(args.udp_port),
            ]
            child_proc = subprocess.Popen(
                child_cmd,
                cwd=str(SCRIPT_DIR),
                env=env,
                stdin=task_stdin,
                stdout=task_out,
                stderr=task_err,
                start_new_session=True,
            )

            start_wait = time.monotonic()
            while not checkpoint_requested:
                if child_proc.poll() is not None:
                    raise RuntimeError(f"training process exited before checkpoint request: {child_proc.returncode}")
                if time.monotonic() - start_wait > args.signal_timeout:
                    raise TimeoutError(f"timed out waiting {args.signal_timeout}s for checkpoint request")
                time.sleep(0.1)

            child_pid = child_proc.pid
            if CHILD_PID_PATH.exists():
                raw_child_pid = CHILD_PID_PATH.read_text().strip()
                if raw_child_pid:
                    child_pid = int(raw_child_pid)

            start_dump = time.monotonic()
            dump_error = None
            try:
                subprocess.run(
                    ["pos_cli", "--dump", "--dir", str(ckpt_dir), "--pid", str(child_pid)],
                    cwd=str(SCRIPT_DIR),
                    stdout=ckpt_out,
                    stderr=ckpt_err,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                dump_error = exc
            dump_elapsed = time.monotonic() - start_dump
            time_out.write(f"process dump time: {dump_elapsed} s\n")
            if dump_error is not None:
                time_out.write(f"process dump returncode: {dump_error.returncode}\n")
            append_pos_log_metrics(time_out, ckpt_dir)
            if dump_error is not None:
                raise dump_error
        finally:
            terminate_process_tree(child_proc, "training")
            terminate_process_tree(daemon_proc, "daemon")
            cleanup_control_files()

    return log_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run a PhOS ResNet checkpoint reproduction.")
    parser.add_argument("--freq", type=int, default=10, help="checkpoint request frequency in training iterations")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workload", default=DEFAULT_WORKLOAD)
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--ckpt-dir", default="/root/ckpt")
    parser.add_argument("--udp-port", type=int, default=10000)
    parser.add_argument("--signal-timeout", type=float, default=600.0)
    parser.add_argument("--daemon-start-delay", type=float, default=1.0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.preflight_only:
        preflight(args)
        return

    preflight(args)
    for index in range(1, args.repeats + 1):
        log_dir = run_one(args, index)
        print(f"wrote logs to {log_dir}")


if __name__ == "__main__":
    main()
