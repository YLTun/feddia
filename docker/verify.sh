#!/bin/bash
# =============================================================================
# Verify the Isaac Sim + Isaac Lab Docker environment
# Run this INSIDE the container: bash docker/verify.sh
# Or from the host:  docker exec rl-navigation bash /home/rl_user/workspace/docker/verify.sh
# =============================================================================

PASS=0
FAIL=0
PYTHON="/opt/IsaacLab/_isaac_sim/python.sh"

check() {
    local label="$1"
    local result="$2"
    if [ $? -eq 0 ] && [ -n "$result" ]; then
        echo "  ✅ ${label}: ${result}"
        ((PASS++))
    else
        echo "  ❌ ${label}: FAILED"
        ((FAIL++))
    fi
}

echo "============================================="
echo " Isaac RL Navigation — Environment Check"
echo "============================================="
echo ""

# 1. GPU
echo "🔧 GPU"
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)
check "GPU" "$GPU"
echo ""

# 2. PyTorch + CUDA
echo "🔧 PyTorch"
TORCH=$($PYTHON -c "import torch; print(f'v{torch.__version__}, CUDA={torch.cuda.is_available()}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null)
check "PyTorch" "$TORCH"
echo ""

# 3. Isaac Lab packages
echo "🔧 Isaac Lab"
ILAB_RL=$($PYTHON -c "import pkg_resources; print(pkg_resources.get_distribution('isaaclab-rl').version)" 2>/dev/null)
check "isaaclab-rl" "$ILAB_RL"
ILAB_TASKS=$($PYTHON -c "import pkg_resources; print(pkg_resources.get_distribution('isaaclab-tasks').version)" 2>/dev/null)
check "isaaclab-tasks" "$ILAB_TASKS"
ILAB_ASSETS=$($PYTHON -c "import pkg_resources; print(pkg_resources.get_distribution('isaaclab-assets').version)" 2>/dev/null)
check "isaaclab-assets" "$ILAB_ASSETS"
echo ""

# 4. RL Libraries
echo "🔧 RL Libraries"
SB3=$($PYTHON -c "import stable_baselines3; print(stable_baselines3.__version__)" 2>/dev/null)
check "stable-baselines3" "$SB3"
GYM=$($PYTHON -c "import gymnasium; print(gymnasium.__version__)" 2>/dev/null)
check "gymnasium" "$GYM"
TB=$($PYTHON -c "import tensorboard; print(tensorboard.__version__)" 2>/dev/null)
check "tensorboard" "$TB"
echo ""

# 5. Workspace
echo "🔧 Workspace"
if [ -d "/home/rl_user/workspace" ]; then
    check "workspace mount" "/home/rl_user/workspace exists"
else
    echo "  ❌ workspace mount: /home/rl_user/workspace not found"
    ((FAIL++))
fi
echo ""

# Summary
echo "============================================="
echo " Results: ${PASS} passed, ${FAIL} failed"
echo "============================================="

[ $FAIL -eq 0 ] && exit 0 || exit 1
