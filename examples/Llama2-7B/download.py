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

import os
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_example_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
if not hf_token:
    raise RuntimeError(
        'HF_TOKEN is not set. Export a Hugging Face read token that has access '
        'to the gated Llama 2 repository before running this script.'
    )

model_id = os.getenv('LLAMA_MODEL_ID', 'meta-llama/Llama-2-7b-chat-hf')
model_path = resolve_example_path(os.getenv('MODEL_PATH', 'model'))
tokenizer_path = resolve_example_path(os.getenv('TOKENIZER_PATH', 'tokenizer'))

# Download model parameters and tokenizer files into a local snapshot.
snapshot_download(
    repo_id=model_id,
    local_dir=str(model_path),
    token=hf_token,
    local_dir_use_symlinks=False,
)

# Save a standalone tokenizer directory for scripts that expect ./tokenizer.
tokenizer_path.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
tokenizer.save_pretrained(str(tokenizer_path))

print(f"downloaded {model_id} to {model_path}")
print(f"saved tokenizer to {tokenizer_path}")

exit(0)
