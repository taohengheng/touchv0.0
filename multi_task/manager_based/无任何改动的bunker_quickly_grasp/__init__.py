# import gymnasium as gym

# from . import agents, robot_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

# gym.register(
#     id="bunker_v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": robot_env_cfg.RobotEnvCfg,
#         # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DoorPPORunnerCfg",
#         # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#     },
#     disable_env_checker=True,
# )