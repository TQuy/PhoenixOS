#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PHOS_CONTAINER_NAME=${PHOS_CONTAINER_NAME:-phos_resnet_repro}
IMAGE=${PHOS_IMAGE:-phoenixos/pytorch:11.3-ubuntu20.04}
SUDO=${SUDO:-sudo}

if $SUDO docker ps -aq -f "name=^${PHOS_CONTAINER_NAME}$" | grep -q .; then
    echo "Container ${PHOS_CONTAINER_NAME} already exists."
    echo "Enter it with: ${SUDO} docker exec -it ${PHOS_CONTAINER_NAME} /bin/bash"
    exit 1
fi

$SUDO docker run -dit --gpus all --privileged --ipc=host --network=host \
    -v "$REPO_ROOT":/root \
    --name "$PHOS_CONTAINER_NAME" \
    "$IMAGE"

echo "Started ${PHOS_CONTAINER_NAME} from ${IMAGE}."
echo "Enter it with: ${SUDO} docker exec -it ${PHOS_CONTAINER_NAME} /bin/bash"
echo "Then run: bash /root/examples/resnet-reproduction/build_inside_container.sh"

