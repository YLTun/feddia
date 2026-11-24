#!/bin/bash
# =============================================================================
# Build the Isaac Sim + Isaac Lab Docker image for RL Navigation
# =============================================================================
set -e

# Load configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "${SCRIPT_DIR}/.env"

# Preserve the invoking user's UID/GID when running via sudo.
HOST_UID="${SUDO_UID:-$(id -u)}"
HOST_GID="${SUDO_GID:-$(id -g)}"

echo "============================================="
echo " Building Isaac RL Navigation Docker Image"
echo "============================================="
echo " Isaac Sim Version : ${ISAAC_SIM_VERSION}"
echo " Isaac Lab Version : ${ISAAC_LAB_VERSION}"
echo " Image Name        : ${IMAGE_NAME}"
echo " User ID           : ${HOST_UID}"
echo " Group ID          : ${HOST_GID}"
echo "============================================="
echo ""

docker build \
    --build-arg ISAAC_SIM_VERSION=${ISAAC_SIM_VERSION} \
    --build-arg ISAAC_LAB_VERSION=${ISAAC_LAB_VERSION} \
    --build-arg USER_ID=${HOST_UID} \
    --build-arg GROUP_ID=${HOST_GID} \
    -t ${IMAGE_NAME} \
    "${SCRIPT_DIR}"

echo ""
echo "============================================="
echo " Build complete: ${IMAGE_NAME}"
echo "============================================="
