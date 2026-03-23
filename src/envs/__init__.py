from gymnasium.envs.registration import register, registry

from .pusht import PushTEnv

__all__ = ["PushTEnv"]

if "PushT-v0" not in registry:
    register(
        id="PushT-v0",
        entry_point="src.envs.pusht:PushTEnv",
    )