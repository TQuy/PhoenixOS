# Copyright 2024 The PhoenixOS Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT = (
    "PhoenixOS is measuring GPU checkpoint and restore latency for a "
    "Llama2-7B workload. Keep this process alive while it repeatedly runs "
    "small CUDA forward passes."
)


def resolve_example_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Long-running Llama2-7B workload for PhOS checkpoint/restore."
    )
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH", "model"))
    parser.add_argument("--tokenizer-path", default=os.getenv("TOKENIZER_PATH", "tokenizer"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16" if torch.cuda.is_available() else "float32",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of loop steps. Use 0 to run until the process is dumped or killed.",
    )
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser.parse_args()


def load_model(model_path, torch_dtype):
    kwargs = {"torch_dtype": torch_dtype}
    try:
        return AutoModelForCausalLM.from_pretrained(
            str(model_path),
            low_cpu_mem_usage=True,
            **kwargs,
        )
    except ImportError as exc:
        print(f"[WARN] loading without low_cpu_mem_usage: {exc}", flush=True)
        return AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)


def build_batch(tokenizer, args, device):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        [args.prompt] * args.batch_size,
        return_tensors="pt",
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        max_length=args.seq_len,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def run_step(model, batch, device):
    with torch.no_grad():
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return float(loss.detach().cpu())


def train():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.seq_len < 2:
        raise ValueError("--seq-len must be >= 2")
    if args.steps < 0:
        raise ValueError("--steps must be >= 0")

    pid = os.getpid()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is not available")

    model_path = resolve_example_path(args.model_path)
    tokenizer_path = resolve_example_path(args.tokenizer_path)
    if not tokenizer_path.exists() and model_path.exists():
        tokenizer_path = model_path

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    torch.backends.cudnn.enabled = False
    print(f"process id: {pid}", flush=True)
    print(f"loading tokenizer from {tokenizer_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)

    print(f"loading model from {model_path}", flush=True)
    model = load_model(model_path, torch_dtype=torch_dtype).to(device)
    model.eval()

    batch = build_batch(tokenizer, args, device)

    print("warming up CUDA workload...", flush=True)
    warmup_loss = run_step(model, batch, device)
    print(f"[READY] pid={pid} warmup_loss={warmup_loss:.4f}", flush=True)
    print(
        f"[READY] dump with: pos_cli --dump --dir /root/ckpt --pid {pid}",
        flush=True,
    )

    step = 0
    while args.steps == 0 or step < args.steps:
        step += 1
        start = time.time()
        loss = run_step(model, batch, device)
        elapsed = time.time() - start
        if args.log_every > 0 and step % args.log_every == 0:
            print(
                f"[STEP] step={step} loss={loss:.4f} duration={elapsed:.4f}s pid={pid}",
                flush=True,
            )
        if args.sleep > 0:
            time.sleep(args.sleep)


if __name__ == '__main__':
    train()
