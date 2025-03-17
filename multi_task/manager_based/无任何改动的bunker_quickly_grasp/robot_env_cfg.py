import torch
import math

#from multi_task.manager_based.bunker_quickly_grasp.scene_env_cfg import GraspEnvCfg
from .scene_env_cfg import GraspEnvCfg

from .robot_cfg import SUMMIT_CFG   #机器人配置文件

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass


@configclass
class RobotEnvCfg(GraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.robot = SUMMIT_CFG.replace(prim_path="{ENV_REGEX_NS}/Bunker_Robot")
        
        







@configclass
class RobotEnvCfg_PLAY(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # make a smaller scene for play
        self.scene.num_envs = 1
        