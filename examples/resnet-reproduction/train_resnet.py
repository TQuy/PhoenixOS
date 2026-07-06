#!/usr/bin/env python3
import argparse
import os
import signal
import socket
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import ResNet152_Weights, resnet152
from tqdm import tqdm


torch.backends.cudnn.enabled = False


def read_parent_pid(path):
    try:
        raw = path.read_text().strip()
    except FileNotFoundError:
        return None
    return int(raw) if raw else None


def request_checkpoint(args, udp_socket):
    parent_pid = read_parent_pid(args.parent_pid_file)
    print(f"Process {os.getpid()}: parent pid is {parent_pid}", flush=True)

    if parent_pid is None:
        if args.no_checkpoint_signal:
            return
        raise RuntimeError(f"missing parent pid file: {args.parent_pid_file}")

    args.child_pid_file.write_text(f"{os.getpid()}\n")
    print(f"Process {os.getpid()}: requesting checkpoint", flush=True)
    os.kill(parent_pid, signal.SIGUSR2)

    if args.wait_after_signal:
        data, addr = udp_socket.recvfrom(1024)
        print(f"Received UDP data from {addr}: {data.decode().strip()}", flush=True)


def build_model(num_classes):
    weights = ResNet152_Weights.DEFAULT
    model = resnet152(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model, weights.transforms()


def run_train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this reproduction")

    device = torch.device("cuda:0")
    model, preprocess = build_model(args.num_classes)
    model = model.to(device)

    print("Preparing CIFAR10 dataset...", flush=True)
    train_dataset = datasets.CIFAR10(
        root=str(args.data_dir),
        train=True,
        download=True,
        transform=preprocess,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    print("Dataset ready.", flush=True)

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(("0.0.0.0", args.udp_port))
    print(f"UDP listening port {args.udp_port}...", flush=True)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    nb_iteration = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}")

        for data, target in progress:
            start_t = time.time()

            data = data.to(device)
            target = target.to(device)

            output = model(data)
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            torch.cuda.default_stream(0).synchronize()

            nb_iteration += 1
            duration_ms = (time.time() - start_t) * 1000.0
            progress.set_postfix({"loss": f"{loss.item():.4f}", "iter_time_ms": f"{duration_ms:.2f}"})
            print(f"iteration {nb_iteration} duration: {duration_ms:.2f} ms", flush=True)

            if args.max_iters and nb_iteration >= args.max_iters:
                print(f"Reached max iterations ({args.max_iters}).", flush=True)
                return

            if args.freq > 0 and nb_iteration % args.freq == 0:
                request_checkpoint(args, udp_socket)


def parse_args():
    parser = argparse.ArgumentParser(description="GCR-style ResNet152 training workload for PhOS C/R.")
    parser.add_argument("-f", "--freq", type=int, default=10, help="request a checkpoint every N iterations; 0 disables it")
    parser.add_argument("--max-iters", type=int, default=0, help="stop after this many iterations; 0 means unlimited")
    parser.add_argument("--epochs", type=int, default=100000, help="maximum number of epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--data-dir", type=lambda p: os.path.abspath(p), default="./data")
    parser.add_argument("--udp-port", type=int, default=int(os.environ.get("RESNET_REPRO_UDP_PORT", "10000")))
    parser.add_argument("--parent-pid-file", type=lambda p: os.path.abspath(p), default="/dev/shm/cr/parent_pid")
    parser.add_argument("--child-pid-file", type=lambda p: os.path.abspath(p), default="/dev/shm/cr/child_pid")
    parser.add_argument("--no-checkpoint-signal", action="store_true", help="skip parent signaling, useful for warmup")
    parser.add_argument("--no-wait-after-signal", dest="wait_after_signal", action="store_false")
    parser.set_defaults(wait_after_signal=True)
    args = parser.parse_args()
    from pathlib import Path

    args.data_dir = Path(args.data_dir)
    args.parent_pid_file = Path(args.parent_pid_file)
    args.child_pid_file = Path(args.child_pid_file)
    return args


if __name__ == "__main__":
    run_train(parse_args())

