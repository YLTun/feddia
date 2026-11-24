#!/bin/bash
# =============================================================================
# Run the Isaac Sim + Isaac Lab Docker container for RL Navigation
#
# Usage:
#   ./run.sh                          # Interactive mode on all GPUs (default)
#   ./run.sh headless                 # Headless mode on all GPUs
#   ./run.sh --gpu 0                  # Interactive mode on GPU 0 only
#   ./run.sh headless --gpu 0         # Headless mode on GPU 0 only
#   ./run.sh --gpu all                # Explicitly expose all GPUs
# =============================================================================
set -e

# Load configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(realpath "${SCRIPT_DIR}/..")"
source "${SCRIPT_DIR}/.env"

MODE="interactive"
GPU_DEVICE="all"

while [ $# -gt 0 ]; do
    case "$1" in
        interactive|headless)
            MODE="$1"
            ;;
        --gpu)
            if [ $# -lt 2 ]; then
                echo "Error: --gpu requires a value (for example: 0 or all)" >&2
                exit 1
            fi
            GPU_DEVICE="$2"
            shift
            ;;
        --gpu=*)
            GPU_DEVICE="${1#*=}"
            ;;
        -h|--help)
            sed -n '1,11p' "$0"
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$1'" >&2
            echo "Usage: ./run.sh [interactive|headless] [--gpu <index|all>]" >&2
            exit 1
            ;;
    esac
    shift
done

if [ "${GPU_DEVICE}" = "all" ]; then
    GPU_FLAG="all"
    NVIDIA_VISIBLE_DEVICES="all"
else
    GPU_FLAG="\"device=${GPU_DEVICE}\""
    NVIDIA_VISIBLE_DEVICES="${GPU_DEVICE}"
fi

echo "============================================="
echo " Isaac RL Navigation Container"
echo "============================================="
echo " Image     : ${IMAGE_NAME}"
echo " Mode      : ${MODE}"
echo " GPUs      : ${GPU_DEVICE}"
echo " Project   : ${PROJECT_DIR}"
echo "============================================="

# Common Docker flags
DOCKER_FLAGS=(
    -it --rm
    --gpus ${GPU_FLAG}
    --network host
    --name "${CONTAINER_NAME}"
    # NVIDIA driver capabilities
    -e NVIDIA_DRIVER_CAPABILITIES=all
    -e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}
    # Isaac Sim EULA acceptance
    -e ACCEPT_EULA=Y
    # Bind mount: project workspace
    -v "${PROJECT_DIR}":/home/rl_user/workspace
    -w /home/rl_user/workspace
    # Named volumes: persist Isaac Sim caches across container restarts
    -v isaac-sim-cache-kit:/isaac-sim/kit/cache
    -v isaac-sim-cache-ov:/root/.cache/ov
    -v isaac-sim-cache-pip:/root/.cache/pip
    -v isaac-sim-cache-compute:/root/.nv/ComputeCache
    -v isaac-sim-logs:/root/.nvidia-omniverse/logs
    -v isaac-sim-data:/root/.local/share/ov/data
)

# Mode-specific flags
if [ "${MODE}" = "interactive" ]; then
    # Allow X11 forwarding for GUI
    xhost +local:root 2>/dev/null || true
    DOCKER_FLAGS+=(
        -e DISPLAY=${DISPLAY}
        -v /tmp/.X11-unix:/tmp/.X11-unix
        # The image ENTRYPOINT is /isaac-sim/runheadless.sh, which immediately
        # launches a WebRTC streaming Isaac Sim and holds GPU memory for the
        # life of the container -- wasteful when we only want a shell to run
        # training in. Override it for a plain shell; start streaming on demand
        # with /isaac-sim/isaac-sim.streaming.sh if ever needed.
        # (Unrelated to the black-window issue -- see the /dev/dri block below.)
        --entrypoint bash
    )

    # nvidia-container-toolkit >= 1.20.0 no longer injects /dev/dri into the
    # container. Compute and offscreen rendering still work through
    # /dev/nvidia*, but a Vulkan/GLX client needs the DRI nodes to present to
    # the X server -- without them Isaac Sim opens a window that renders
    # nothing (black). Docker recreates the device nodes without the host's
    # ACLs, so the container user also needs the owning groups (video /
    # render) to open them.
    if [ -d /dev/dri ]; then
        for _dri in /dev/dri/card* /dev/dri/renderD*; do
            [ -e "${_dri}" ] && DOCKER_FLAGS+=( --device "${_dri}:${_dri}" )
        done
        for _grp in video render; do
            _gid="$(getent group "${_grp}" | cut -d: -f3)"
            [ -n "${_gid}" ] && DOCKER_FLAGS+=( --group-add "${_gid}" )
        done
        unset _dri _grp _gid
    fi

    echo " Display   : ${DISPLAY}"
elif [ "${MODE}" = "headless" ]; then
    DOCKER_FLAGS+=(
        -e DISPLAY=""
    )
    echo " Display   : headless (no GUI)"
fi

echo "============================================="
echo ""

docker run "${DOCKER_FLAGS[@]}" ${IMAGE_NAME}
