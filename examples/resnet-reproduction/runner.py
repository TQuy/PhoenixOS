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
DEFAULT_WORKLOADS = {
    "phos": "phos-trans-resnet",
    "cuda": "cuda-trans-resnet",
    "cuda-gpu": "cuda-gpu-trans-resnet",
}
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


def find_cuda_checkpoint(configured="cuda-checkpoint"):
    configured_path = Path(configured)
    if configured and (configured_path.is_absolute() or "/" in configured):
        if configured_path.exists():
            return str(configured_path)

    path = shutil.which(configured or "cuda-checkpoint")
    if path:
        return path

    candidates = [
        Path("/root/third_party/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint"),
        SCRIPT_DIR.parent.parent / "third_party" / "cuda-checkpoint" / "bin" / "x86_64_Linux" / "cuda-checkpoint",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


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


def import_pynvml():
    try:
        import pynvml
    except ImportError as exc:
        raise RuntimeError("pynvml is required for cuda-checkpoint GPU PID discovery") from exc
    return pynvml


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


def format_pid_list(pids):
    return ",".join(str(pid) for pid in pids)


def process_tree_pids(parent_pid):
    pids = {parent_pid}
    if psutil is None:
        return pids

    try:
        parent = psutil.Process(parent_pid)
        pids.update(child.pid for child in parent.children(recursive=True))
    except psutil.NoSuchProcess:
        pass
    return pids


def find_gpu_using_child_processes(parent_pid):
    candidate_pids = process_tree_pids(parent_pid)
    gpu_pids = set()
    pynvml = import_pynvml()

    try:
        pynvml.nvmlInit()
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                pid = int(proc.pid)
                if pid in candidate_pids:
                    gpu_pids.add(pid)
    except pynvml.NVMLError:
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    return sorted(gpu_pids)


def wait_for_gpu_pids(parent_pid, fallback_pid, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        gpu_pids = find_gpu_using_child_processes(parent_pid)
        if gpu_pids:
            return gpu_pids
        time.sleep(0.1)
    return [fallback_pid]


def snapshot_gpu_memory_by_pid(pids):
    target_pids = {int(pid) for pid in pids}
    memory_by_pid = {pid: 0 for pid in target_pids}
    pynvml = import_pynvml()

    try:
        pynvml.nvmlInit()
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                pid = int(proc.pid)
                if pid not in target_pids:
                    continue
                used = getattr(proc, "usedGpuMemory", None)
                if isinstance(used, int) and used >= 0:
                    memory_by_pid[pid] += used
    except pynvml.NVMLError:
        return {}
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    return {pid: used for pid, used in memory_by_pid.items() if used > 0}


def write_process_status(pid, label, stdout, stderr):
    stdout.write(f"=== ps {label} pid {pid} ===\n")
    stdout.flush()
    subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid,rss,vsz,comm"],
        text=True,
        check=False,
        stdout=stdout,
        stderr=stderr,
    )


def wait_for_dumped_child_exit(proc, time_output, timeout):
    if proc is None:
        return

    try:
        returncode = proc.wait(timeout=timeout)
        time_output.write(f"checkpointed child returncode: {returncode}\n")
        time_output.flush()
    except subprocess.TimeoutExpired as exc:
        time_output.write(f"checkpointed child wait timeout: {timeout} s\n")
        time_output.flush()
        raise RuntimeError("checkpointed child process did not exit after CRIU dump") from exc


def find_training_pids():
    if psutil is None:
        return []

    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        joined = " ".join(cmdline)
        if "train_resnet.py" in joined:
            found.append(int(proc.info["pid"]))
    return sorted(found)


def wait_for_restored_training_pid(original_pid, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if psutil is not None:
            try:
                proc = psutil.Process(original_pid)
                cmdline = " ".join(proc.cmdline())
                if "train_resnet.py" in cmdline:
                    return original_pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        pids = find_training_pids()
        if pids:
            return pids[0]
        time.sleep(0.1)
    return None


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


def append_pos_restore_metrics(time_output):
    plog = read_pos_log()
    time_output.write(f"buffer restore size: {plog.tot_restore_size} bytes\n")
    time_output.write(f"buffer restore time: {plog.tot_restore_time} seconds\n")
    time_output.write(f"buffer restore memcpy time: {plog.restore_memcpy_time} seconds\n")
    if plog.tot_restore_time > 0:
        bw = plog.tot_restore_size / (1024.0 * 1024.0 * 1024.0) / plog.tot_restore_time
        time_output.write(f"buffer restore bandwidth: {bw} GB/s\n")
    if plog.restore_memcpy_time > 0:
        bw = plog.tot_restore_size / (1024.0 * 1024.0 * 1024.0) / plog.restore_memcpy_time
        time_output.write(f"buffer restore memcpy bandwidth: {bw} GB/s\n")
    time_output.flush()


def run_phos_checkpoint(args, child_pid, ckpt_dir, ckpt_out, ckpt_err, time_out):
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


def run_phos_restore(args, child_pid, ckpt_dir, restore_out, restore_err, time_out):
    if args.skip_restore:
        return None

    start_restore = time.monotonic()
    restore_proc = subprocess.Popen(
        ["pos_cli", "--restore", "--dir", str(ckpt_dir)],
        cwd=str(SCRIPT_DIR),
        stdout=restore_out,
        stderr=restore_err,
        text=True,
        start_new_session=True,
    )

    restored_pid = None
    deadline = time.monotonic() + args.restore_timeout
    try:
        while time.monotonic() < deadline:
            restored_pid = wait_for_restored_training_pid(child_pid, 0.1)
            if restored_pid is not None:
                restore_elapsed = time.monotonic() - start_restore
                time_out.write(f"process restore time: {restore_elapsed} s\n")
                append_pos_restore_metrics(time_out)
                return restored_pid

            returncode = restore_proc.poll()
            if returncode is not None:
                if returncode != 0:
                    raise subprocess.CalledProcessError(returncode, restore_proc.args)
                restored_pid = wait_for_restored_training_pid(child_pid, args.restored_pid_timeout)
                restore_elapsed = time.monotonic() - start_restore
                time_out.write(f"process restore time: {restore_elapsed} s\n")
                append_pos_restore_metrics(time_out)
                return restored_pid

        raise TimeoutError(f"timed out waiting {args.restore_timeout}s for PhOS restore")
    finally:
        terminate_process_tree(restore_proc, "phos restore")


def run_cuda_gpu_checkpoint(args, child_proc, child_pid, ckpt_out, ckpt_err, restore_out, restore_err, time_out):
    cuda_checkpoint = find_cuda_checkpoint(args.cuda_checkpoint_bin)
    if cuda_checkpoint is None:
        raise RuntimeError("required command not found: cuda-checkpoint")

    gpu_pids = wait_for_gpu_pids(child_proc.pid, child_pid, args.gpu_pid_timeout)
    time_out.write(f"cuda checkpoint gpu pids: {format_pid_list(gpu_pids)}\n")

    memory_before = snapshot_gpu_memory_by_pid(gpu_pids)
    for pid in gpu_pids:
        if pid in memory_before:
            time_out.write(f"process {pid} gpu memory before checkpoint: {memory_before[pid]} bytes\n")

    checkpoint_total = 0.0
    for pid in gpu_pids:
        write_process_status(pid, "before cuda checkpoint", ckpt_out, ckpt_err)
        start_time = time.monotonic()
        subprocess.run(
            [cuda_checkpoint, "--toggle", "--pid", str(pid)],
            text=True,
            check=True,
            stdout=ckpt_out,
            stderr=ckpt_err,
        )
        elapsed = time.monotonic() - start_time
        checkpoint_total += elapsed
        ckpt_out.write(f"cuda-checkpoint checkpoint time: {elapsed:.6f} s\n")
        ckpt_out.flush()
        time_out.write(f"process {pid} gpu checkpoint time: {elapsed} s\n")
        write_process_status(pid, "after cuda checkpoint", ckpt_out, ckpt_err)

    memory_after = snapshot_gpu_memory_by_pid(gpu_pids)
    for pid in gpu_pids:
        if pid in memory_after:
            time_out.write(f"process {pid} gpu memory after checkpoint: {memory_after[pid]} bytes\n")

    memory_before_total = sum(memory_before.values())
    memory_after_total = sum(memory_after.values())
    time_out.write(f"cuda checkpoint aggregate time: {checkpoint_total} s\n")
    time_out.write(f"cuda checkpoint gpu memory before total: {memory_before_total} bytes\n")
    time_out.write(f"cuda checkpoint gpu memory after total: {memory_after_total} bytes\n")
    if checkpoint_total > 0 and memory_before_total > 0:
        bw = memory_before_total / (1024.0 * 1024.0 * 1024.0) / checkpoint_total
        time_out.write(f"cuda checkpoint aggregate bandwidth: {bw} GB/s\n")

    if args.cuda_resume_after_checkpoint:
        restore_total = 0.0
        for pid in gpu_pids:
            start_time = time.monotonic()
            subprocess.run(
                [cuda_checkpoint, "--toggle", "--pid", str(pid)],
                text=True,
                check=True,
                stdout=restore_out,
                stderr=restore_err,
            )
            elapsed = time.monotonic() - start_time
            restore_total += elapsed
            restore_out.write(f"cuda-checkpoint restore time: {elapsed:.6f} s\n")
            restore_out.flush()
            time_out.write(f"process {pid} gpu restore time: {elapsed} s\n")
        time_out.write(f"cuda restore aggregate time: {restore_total} s\n")

    time_out.flush()


def run_cuda_criu_checkpoint_restore(args, child_proc, child_pid, ckpt_dir, ckpt_out, ckpt_err, restore_out, restore_err, time_out):
    cuda_checkpoint = find_cuda_checkpoint(args.cuda_checkpoint_bin)
    if cuda_checkpoint is None:
        raise RuntimeError("required command not found: cuda-checkpoint")

    gpu_pids = wait_for_gpu_pids(child_proc.pid, child_pid, args.gpu_pid_timeout)
    time_out.write("cuda checkpoint mode: criu\n")
    time_out.write(f"cuda checkpoint gpu pids: {format_pid_list(gpu_pids)}\n")

    memory_before = snapshot_gpu_memory_by_pid(gpu_pids)
    for pid in gpu_pids:
        if pid in memory_before:
            time_out.write(f"process {pid} gpu memory before checkpoint: {memory_before[pid]} bytes\n")

    checkpoint_start = time.monotonic()
    gpu_checkpoint_total = 0.0
    for pid in gpu_pids:
        write_process_status(pid, "before cuda checkpoint", ckpt_out, ckpt_err)
        start_time = time.monotonic()
        subprocess.run(
            [cuda_checkpoint, "--toggle", "--pid", str(pid)],
            text=True,
            check=True,
            stdout=ckpt_out,
            stderr=ckpt_err,
        )
        elapsed = time.monotonic() - start_time
        gpu_checkpoint_total += elapsed
        ckpt_out.write(f"cuda-checkpoint checkpoint time: {elapsed:.6f} s\n")
        ckpt_out.flush()
        time_out.write(f"process {pid} gpu checkpoint time: {elapsed} s\n")
        write_process_status(pid, "after cuda checkpoint", ckpt_out, ckpt_err)

    start_criu_dump = time.monotonic()
    subprocess.run(
        [
            "criu",
            "dump",
            "--tree",
            str(child_pid),
            "--images-dir",
            str(ckpt_dir),
            "--shell-job",
            "--display-stats",
        ],
        text=True,
        check=True,
        stdout=ckpt_out,
        stderr=ckpt_err,
    )
    criu_dump_elapsed = time.monotonic() - start_criu_dump
    wait_for_dumped_child_exit(child_proc, time_out, args.child_exit_timeout)
    checkpoint_total = time.monotonic() - checkpoint_start

    memory_after = snapshot_gpu_memory_by_pid(gpu_pids)
    memory_before_total = sum(memory_before.values())
    memory_after_total = sum(memory_after.values())
    time_out.write(f"cuda checkpoint aggregate time: {gpu_checkpoint_total} s\n")
    time_out.write(f"criu dump time: {criu_dump_elapsed} s\n")
    time_out.write(f"cuda-criu checkpoint total time: {checkpoint_total} s\n")
    time_out.write(f"cuda checkpoint gpu memory before total: {memory_before_total} bytes\n")
    time_out.write(f"cuda checkpoint gpu memory after total: {memory_after_total} bytes\n")
    time_out.write(f"process checkpoint size: {dir_size(ckpt_dir)} bytes\n")
    if gpu_checkpoint_total > 0 and memory_before_total > 0:
        bw = memory_before_total / (1024.0 * 1024.0 * 1024.0) / gpu_checkpoint_total
        time_out.write(f"cuda checkpoint aggregate bandwidth: {bw} GB/s\n")

    restored_pid = None
    if not args.skip_restore:
        restore_start = time.monotonic()
        start_criu_restore = time.monotonic()
        subprocess.run(
            [
                "criu",
                "restore",
                "-D",
                str(ckpt_dir),
                "-j",
                "--display-stats",
                "--restore-detached",
            ],
            text=True,
            check=True,
            stdout=restore_out,
            stderr=restore_err,
            timeout=args.restore_timeout,
        )
        criu_restore_elapsed = time.monotonic() - start_criu_restore
        time_out.write(f"criu restore time: {criu_restore_elapsed} s\n")

        restored_pid = wait_for_restored_training_pid(child_pid, args.restored_pid_timeout)
        if restored_pid is None:
            raise RuntimeError("CRIU restore returned, but no restored train_resnet.py process was found")

        start_gpu_restore = time.monotonic()
        subprocess.run(
            [cuda_checkpoint, "--toggle", "--pid", str(restored_pid)],
            text=True,
            check=True,
            stdout=restore_out,
            stderr=restore_err,
            timeout=args.restore_timeout,
        )
        gpu_restore_elapsed = time.monotonic() - start_gpu_restore
        restore_total = time.monotonic() - restore_start
        restore_out.write(f"cuda-checkpoint restore time: {gpu_restore_elapsed:.6f} s\n")
        restore_out.flush()
        time_out.write(f"process {restored_pid} gpu restore time: {gpu_restore_elapsed} s\n")
        time_out.write(f"cuda restore aggregate time: {gpu_restore_elapsed} s\n")
        time_out.write(f"cuda-criu restore total time: {restore_total} s\n")

    time_out.flush()
    return restored_pid


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


def terminate_pid_tree(pid, timeout=5):
    if pid is None or psutil is None:
        return

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    for child in children:
        child.terminate()
    parent.terminate()
    gone, alive = psutil.wait_procs(children + [parent], timeout=timeout)
    for item in alive:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass


def preflight(args):
    require_python_import("psutil")
    require_python_import("tqdm")
    require_python_import("torch")
    require_python_import("torchvision")
    require_torchvision_resnet_api()

    result = {"ok": True, "method": args.method}
    if args.method == "phos":
        require_command("pos_cli")
        if find_cricket_client() is None:
            raise RuntimeError("could not find cricket-client.so through common library paths or ldconfig")
    elif args.method in ("cuda", "cuda-gpu"):
        require_python_import("pynvml")
        cuda_checkpoint = find_cuda_checkpoint(args.cuda_checkpoint_bin)
        if cuda_checkpoint is None:
            raise RuntimeError("required command not found: cuda-checkpoint")
        result["cuda_checkpoint"] = cuda_checkpoint
        if args.method == "cuda":
            require_command("criu")

    ckpt_dir = resolve_path(args.ckpt_dir, Path("/"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    POS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    init_control_files()
    device = check_cuda()
    cleanup_control_files()
    result["cuda_device"] = device
    result["ckpt_dir"] = str(ckpt_dir)
    print(json.dumps(result, indent=2))


def run_one(args, run_index):
    global checkpoint_requested
    checkpoint_requested = False

    base_workload = args.workload or DEFAULT_WORKLOADS[args.method]
    workload = base_workload if args.repeats == 1 else f"{base_workload}-run-{run_index}"
    log_dir = resolve_path(args.log_root) / workload
    clear_directory(log_dir)

    ckpt_dir = resolve_path(args.ckpt_dir, Path("/"))
    clear_directory(ckpt_dir)
    init_control_files()
    if args.method == "phos":
        ensure_pos_yaml()
        clear_client_marker()
    stdin_path = log_dir / "stdin.txt"
    stdin_path.touch()

    daemon_proc = None
    child_proc = None
    restored_pid = None
    signal.signal(signal.SIGUSR2, checkpoint_signal_handler)

    with (log_dir / "daemon.log").open("w") as daemon_out, \
            (log_dir / "daemon.err").open("w") as daemon_err, \
            (log_dir / "task.log").open("w") as task_out, \
            (log_dir / "task.err").open("w") as task_err, \
            (log_dir / "ckpt.log").open("w") as ckpt_out, \
            (log_dir / "ckpt.err").open("w") as ckpt_err, \
            (log_dir / "restore.log").open("w") as restore_out, \
            (log_dir / "restore.err").open("w") as restore_err, \
            stdin_path.open("rb") as task_stdin, \
            (log_dir / "time.log").open("w") as time_out:
        try:
            if args.method == "phos":
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
            if args.method == "phos":
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

            if args.method == "phos":
                run_phos_checkpoint(args, child_pid, ckpt_dir, ckpt_out, ckpt_err, time_out)
                wait_for_dumped_child_exit(child_proc, time_out, args.child_exit_timeout)
                restored_pid = run_phos_restore(args, child_pid, ckpt_dir, restore_out, restore_err, time_out)
            elif args.method == "cuda":
                restored_pid = run_cuda_criu_checkpoint_restore(
                    args,
                    child_proc,
                    child_pid,
                    ckpt_dir,
                    ckpt_out,
                    ckpt_err,
                    restore_out,
                    restore_err,
                    time_out,
                )
            elif args.method == "cuda-gpu":
                run_cuda_gpu_checkpoint(
                    args,
                    child_proc,
                    child_pid,
                    ckpt_out,
                    ckpt_err,
                    restore_out,
                    restore_err,
                    time_out,
                )
        finally:
            terminate_pid_tree(restored_pid)
            terminate_process_tree(child_proc, "training")
            terminate_process_tree(daemon_proc, "daemon")
            cleanup_control_files()

    return log_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run a ResNet checkpoint reproduction.")
    parser.add_argument("--method", choices=("phos", "cuda", "cuda-gpu"), default="phos")
    parser.add_argument("--freq", type=int, default=10, help="checkpoint request frequency in training iterations")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workload", default=None, help="log directory name; defaults to <method>-trans-resnet")
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--ckpt-dir", default="/root/ckpt")
    parser.add_argument("--udp-port", type=int, default=10000)
    parser.add_argument("--signal-timeout", type=float, default=600.0)
    parser.add_argument("--daemon-start-delay", type=float, default=1.0)
    parser.add_argument("--gpu-pid-timeout", type=float, default=30.0)
    parser.add_argument("--restored-pid-timeout", type=float, default=30.0)
    parser.add_argument("--child-exit-timeout", type=float, default=30.0)
    parser.add_argument("--restore-timeout", type=float, default=300.0)
    parser.add_argument("--cuda-checkpoint-bin", default="cuda-checkpoint")
    parser.add_argument(
        "--cuda-resume-after-checkpoint",
        action="store_true",
        help="only for --method cuda-gpu: run a second cuda-checkpoint toggle after measurement",
    )
    parser.add_argument("--skip-restore", action="store_true", help="only run checkpoint, not restore")
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
