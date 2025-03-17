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

from scipy.spatial.transform import Rotation as R





def quaternion_to_rotation_matrix(quaternions):
    quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    wz = w * z
    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - wz)
    m10 = 2 * (xy + wz)
    m11 = 1 - 2 * (xx + zz)
    rotation_matrices = torch.stack([m00, m01, m10, m11], dim=1).reshape(-1, 2, 2)
    return rotation_matrices




#自己速度，标量，正负表示前进或者后退
def robot_linear(env: ManagerBasedRLEnv) -> torch.Tensor:

    root_lin_vel_w = mdp.root_lin_vel_w(env, asset_cfg=SceneEntityCfg(name='robot'))[:,:2]
    root_quat_w = mdp.root_quat_w(env, asset_cfg=SceneEntityCfg(name='robot'))
    rotation_matrices = quaternion_to_rotation_matrix(root_quat_w)
    robot_directions = rotation_matrices[:, :, 0]
    velocity_scalar = torch.norm(root_lin_vel_w, dim=1, keepdim=True)
    projection = torch.sum(root_lin_vel_w * robot_directions, dim=1, keepdim=True)
    velocity_scalar *= torch.sign(projection)
    return velocity_scalar
    
    
#自己的角速度, 符号需要确定一下  
def robot_angular(env: ManagerBasedRLEnv) -> torch.Tensor:
    return mdp.root_ang_vel_w(env, asset_cfg=SceneEntityCfg(name='robot'))[:, 2:3]
    
    
# 物体相对于相机坐标系的位置
def obj_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    return mdp.root_pos_w(env, asset_cfg=SceneEntityCfg(name='object')) - env.scene["camera_frame"].data.target_pos_w.squeeze(1)
    