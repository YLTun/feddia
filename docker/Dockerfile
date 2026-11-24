# =============================================================================
# Isaac Sim + Isaac Lab Docker Environment for RL Navigation
# Base: NVIDIA Isaac Sim 5.1.0 (includes CUDA, PhysX, Omniverse Kit)
# =============================================================================
ARG ISAAC_SIM_VERSION=5.1.0
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION}

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# =============================================================================
# Install system dependencies
# The Isaac Sim base image may use a non-root user, so switch to root first
# =============================================================================
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Install Isaac Lab
# Pinned to a commit on main after v2.3.0 (Isaac Lab 0.54.3, 2026-02-21). The
# v2.3.0 release predates the MultiMeshRayCaster fix the LiDAR depends on: it
# must hit many separate obstacle meshes. Tracking main instead would make
# every rebuild a different version. Set via ISAAC_LAB_VERSION in .env.
# =============================================================================
ARG ISAAC_LAB_VERSION=80094be3245aa5c8376a7464d29cb4412ea518f5
ENV ISAACLAB_PATH=/opt/IsaacLab

# Fetch by ref rather than `git clone --branch`, which accepts only branches
# and tags; this takes a commit SHA as well.
RUN git init -q ${ISAACLAB_PATH} \
    && git -C ${ISAACLAB_PATH} remote add origin https://github.com/isaac-sim/IsaacLab.git \
    && git -C ${ISAACLAB_PATH} fetch --depth 1 origin ${ISAAC_LAB_VERSION} \
    && git -C ${ISAACLAB_PATH} checkout -q FETCH_HEAD

# Install Isaac Lab and its dependencies using the provided install script
# Set TERM to avoid 'ansi+tabs: unknown terminal type' error during build
ENV TERM=xterm
WORKDIR ${ISAACLAB_PATH}

# Isaac Lab expects a symlink _isaac_sim -> Isaac Sim install directory
# In the official Isaac Sim Docker image, Isaac Sim is at /isaac-sim
RUN ln -sf /isaac-sim ${ISAACLAB_PATH}/_isaac_sim

RUN ./isaaclab.sh --install

# The isaaclab.sh --install does NOT pip-install the core 'isaaclab' package.
# We use PYTHONPATH + install only the missing pure-Python runtime deps.
ENV PYTHONPATH="${ISAACLAB_PATH}/source/isaaclab:${PYTHONPATH}"
RUN ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install --no-cache-dir \
    "flatdict" "prettytable==3.3.0" "einops" "flaky"

# =============================================================================
# Install RL dependencies into Isaac Sim's Python environment
# Isaac Sim bundles its own Python, so we install into that
# =============================================================================
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
    stable-baselines3 \
    tensorboard \
    jupyter \
    jupyterlab \
    && ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
    sb3-contrib

# =============================================================================
# Setup non-root user matching host UID/GID
# =============================================================================
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=rl_user

RUN if getent passwd ${USER_ID} > /dev/null; then \
    existing_user=$(getent passwd ${USER_ID} | cut -d: -f1); \
    if [ "${existing_user}" != "root" ]; then userdel -f ${existing_user}; fi; \
    fi && \
    if getent group ${GROUP_ID} > /dev/null; then \
    existing_group=$(getent group ${GROUP_ID} | cut -d: -f1); \
    if [ "${existing_group}" != "root" ]; then groupdel ${existing_group}; fi; \
    fi && \
    groupadd -g ${GROUP_ID} ${USERNAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    # Add user to isaac-sim group so they can access /isaac-sim (owned by isaac-sim:isaac-sim, perms 0750)
    usermod -aG isaac-sim ${USERNAME}

# Give user access to Isaac Lab directory
RUN chown -R ${USER_ID}:${GROUP_ID} ${ISAACLAB_PATH}

# Fix Isaac Sim cache/data permissions so rendering works (materials, configs)
# Without this, the viewport appears black because material cache can't be built
RUN mkdir -p /isaac-sim/kit/cache /isaac-sim/kit/data \
    /isaac-sim/kit/logs \
    /home/${USERNAME}/Documents/Kit/shared \
    /home/${USERNAME}/.nvidia-omniverse/logs \
    /home/${USERNAME}/.local/share/ov/data \
    /home/${USERNAME}/.cache/ov \
    /home/${USERNAME}/.nv/ComputeCache \
    && chmod -R 777 /isaac-sim/kit/cache /isaac-sim/kit/data /isaac-sim/kit/logs \
    && chown -R ${USER_ID}:${GROUP_ID} /home/${USERNAME}

# =============================================================================
# Environment setup
# =============================================================================
USER ${USERNAME}
WORKDIR /home/${USERNAME}/workspace

# Add Isaac Lab to PATH and PYTHONPATH for convenience
RUN echo "# Isaac Lab environment" >> /home/${USERNAME}/.bashrc && \
    echo "export ISAACLAB_PATH=${ISAACLAB_PATH}" >> /home/${USERNAME}/.bashrc && \
    echo 'alias isaaclab="${ISAACLAB_PATH}/isaaclab.sh"' >> /home/${USERNAME}/.bashrc && \
    echo "# Convenience: use isaaclab.sh's Python" >> /home/${USERNAME}/.bashrc && \
    echo 'alias ilab-python="${ISAACLAB_PATH}/isaaclab.sh -p"' >> /home/${USERNAME}/.bashrc

# Accept NVIDIA EULA
ENV ACCEPT_EULA=Y

CMD ["bash"]
