import torch
import math

from dataclasses import MISSING

from isaaclab.sensors import ContactSensor
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg, OffsetCfg


from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation, RigidObject




# def distance_out(env: ManagerBasedRLEnv) -> torch.Tensor:
#     j1_pos = env.scene["robot"].data.joint_pos[:, 0]
#     return env.scene["ee_frame"].data.target_pos_w.squeeze(1)[:,0]  > 0.1 +  env.scene["object"].data.root_pos_w[:,0]


def j1_angle_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    j1_pos = env.scene["robot"].data.joint_pos[:, 0]
    return j1_pos < 0.174



def reset_j1_history(env: ManagerBasedRLEnv,env_ids: torch.Tensor) -> torch.Tensor:
    if hasattr(env, 'j1_v_flag'):
        del env.j1_v_flag