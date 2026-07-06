#!/usr/bin/env bash
set -euo pipefail

if [ ! -d /root/scripts/build_scripts ]; then
    echo "Expected PhoenixOS to be mounted at /root inside the container." >&2
    exit 1
fi

apt-get update
apt-get install -y git wget python3-pip

cd /root/scripts/build_scripts
bash download_assets.sh

bash build.sh -c -3
bash build.sh -3 -i
source /etc/profile
./pos_build -3 -i

python3 -m pip install --no-cache-dir psutil pynvml tqdm

if ! python3 - <<'PY'
from torchvision.models import ResNet152_Weights, resnet152
PY
then
    python3 -m pip install --no-cache-dir torchvision==0.14 --no-deps
fi

source /etc/profile

echo "Build complete. Next:"
echo "  cd /root/examples/resnet-reproduction"
echo "  python3 runner.py --preflight-only"
echo "  bash run_once.sh"
