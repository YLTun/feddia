# Copyright (c) 2024-2026. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation environment package — Gymnasium registration."""

import gymnasium as gym

from .navigation_env import NavigationEnv, NavigationEnvCfg
from .scene_env import (
    SceneNavigationEnv,
    SceneNavigationEnvCfg,
    FactoryNavigationEnvCfg,
    LivingNavigationEnvCfg,
)

# Register with Gymnasium
gym.register(
    id="Isaac-Navigation-v0",
    entry_point="envs.navigation_env:NavigationEnv",
    kwargs={"cfg": NavigationEnvCfg()},
    disable_env_checker=True,
)

# Hand-authored scenes are a separate env class rather than another
# obstacle_mode, so each needs its own id. Scripts that build the env through
# gym.make pick the id from the requested mode.
gym.register(
    id="Isaac-Navigation-Factory-v0",
    entry_point="envs.scene_env:SceneNavigationEnv",
    kwargs={"cfg": FactoryNavigationEnvCfg()},
    disable_env_checker=True,
)

# Same env class, different layout module. Both scenes are 5 x 5 m and inherit
# reward/observation/termination from NavigationEnv, so they are comparable
# with each other and with the toy obstacle modes.
gym.register(
    id="Isaac-Navigation-Living-v0",
    entry_point="envs.scene_env:SceneNavigationEnv",
    kwargs={"cfg": LivingNavigationEnvCfg()},
    disable_env_checker=True,
)

#: Hand-authored scenes, handled by a dedicated env class rather than by the
#: base environment's obstacle sampling. Maps ``--obstacle_mode`` value ->
#: (gym id, cfg class). Anything not listed here is an ordinary obstacle mode.
SCENE_MODES = {
    "factory": ("Isaac-Navigation-Factory-v0", FactoryNavigationEnvCfg),
    "living": ("Isaac-Navigation-Living-v0", LivingNavigationEnvCfg),
}
