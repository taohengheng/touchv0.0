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


def rotate_vector_by_quaternion(vectors: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
    assert vectors.shape[-1] == 3, "Vectors should have shape (n, 3)"
    assert quaternions.shape[-1] == 4, "Quaternions should have shape (n, 4)"
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    t = 2 * torch.cross(quaternions[:, 1:], vectors, dim=1)
    rotated_vectors = vectors + w[:, None] * t + torch.cross(quaternions[:, 1:], t, dim=1)
    return rotated_vectors





# ========================================= bunker ====================================================


# 计算机械臂与物体2D朝向偏差，从而调整底盘朝向
def orientation_obj_hand2d(env: ManagerBasedRLEnv):
    cr5_que = env.scene["link1_frame"].data.target_quat_w.squeeze(1)  # 臂的四元数
    cr5_pos3d = env.scene["link1_frame"].data.target_pos_w.squeeze(1)[:, :2]     # 三维坐标，用于计算二维向量
    obj_pos = env.scene["object"].data.root_pos_w[:, :2]  # 物体世界坐标系位置

    num_env = cr5_que.shape[0]
    init_ori = torch.tensor([[1., 0., 0.]]*num_env).to(cr5_que.device)
    cr5_ori_3d = rotate_vector_by_quaternion(init_ori, cr5_que)
    cr5_ori_2d = cr5_ori_3d[:, :2] / cr5_ori_3d[:, :2].norm(dim=1, keepdim=True) # 实际的朝向，二维
    
    obj_cr5_vector = obj_pos - cr5_pos3d
    obj_cr5_vector_norm = torch.norm(obj_cr5_vector, dim=1, keepdim=True)
    obj_cr5_ori = obj_cr5_vector / obj_cr5_vector_norm  # 应该的朝向,二维
    
    dot_product = torch.sum(obj_cr5_ori * cr5_ori_2d, dim=1)
    angles = torch.acos(dot_product)
    reward = angles / torch.pi  # 转成 0-1 的权重 权重越大，偏的越远
    
    if torch.isnan(reward).any():
        reward = torch.nan_to_num(reward, nan=0.0)
    
    return reward


# 底盘与物体距离
def distance_bunker_handle(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    handle_pos = env.scene["object"].data.root_pos_w[:, :2]
    bunker_pos = env.scene["robot"].data.root_pos_w[:, :2]
    distance = torch.norm(handle_pos - bunker_pos, dim=1, p=2)
    # reward = 1 / (1.0 + distance**2)
    # reward = torch.pow(reward, 2)
    reward = 1 / (1.0 + distance)
    return reward


# 速度的值越大越好
def velocity_bunker_overall(env: ManagerBasedRLEnv):
    root_l_v = torch.sqrt(torch.sum(env.scene["robot"].data.root_lin_vel_w[:, :2]**2, dim=1, keepdim=True))
    root_an_v = torch.sqrt(torch.sum(env.scene["robot"].data.root_ang_vel_w[:, :2]**2, dim=1, keepdim=True))
    v = root_an_v + root_l_v
    return v.reshape(-1) * 1




# ========================================= cr5 ====================================================


# ee与物体距离
def distance_ee_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    handle_pos = env.scene["object"].data.root_pos_w[:, :2]  # shape: [num_envs,3]
    ee_pos = env.scene["ee_frame"].data.target_pos_w.squeeze(1)[:, :2]  # shape: [num_envs,3]
    distance = torch.norm(handle_pos - ee_pos, dim=1, p=2)
    reward = 1 / (1.0 + distance)
    return reward


def distance_success(env: ManagerBasedRLEnv, dis_th) -> torch.Tensor:
    handle_pos = env.scene["object"].data.root_pos_w  # shape: [num_envs,3]
    ee_pos = env.scene["ee_frame"].data.target_pos_w.squeeze(1)  # shape: [num_envs,3]
    distance = torch.norm(handle_pos - ee_pos, dim=1, p=2)
    reward = distance
    reward = torch.where(distance < dis_th, 10, 0)
    # reward[handle_pos[:, 2]>ee_pos[:, 2]+0.01] = 0
    return reward



# ee与物体高度
def height_ee_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    handle_pos_height = env.scene["object"].data.root_pos_w[:, 2]
    ee_pos_height = env.scene["ee_frame"].data.target_pos_w.squeeze(1)[:, 2]  
    distance = torch.abs(handle_pos_height - ee_pos_height)
    reward = 1 / (1.0 + distance)
    return reward




# j1旋转只能逆时针
# 满足抓取条件的旋转可以加更多分
def joint1_rot(env: ManagerBasedRLEnv, dis_th) -> torch.Tensor:
    
    # 高度差
    object_pos = env.scene["object"].data.root_pos_w  
    ee_pos = env.scene["ee_frame"].data.target_pos_w.squeeze(1)  
    height_diff = torch.abs(object_pos[:, 2] - ee_pos[:, 2])
    # # 半径差
    # link1_pos3d = env.scene["link1_frame"].data.target_pos_w.squeeze(1)
    # obj_radius = torch.norm(object_pos[:, :2] - link1_pos3d[:, :2], dim=1, keepdim=True)
    # hand_radius = torch.norm(ee_pos[:, :2] - link1_pos3d[:, :2], dim=1, keepdim=True)
    # radius_diff = torch.abs(obj_radius - hand_radius).reshape(-1)
    
    score = torch.ones_like(height_diff)
    score[(height_diff < 0.04)] += 1
    # score[(radius_diff < 0.05)] += 1
    
    # print(radius_diff)
    # print(height_diff)
    
    # # 和物体的角度差
    # obj_ori = (object_pos[:, :2] - link1_pos3d[:, :2])/torch.norm(object_pos[:, :2] - link1_pos3d[:, :2], dim=1, keepdim=True)
    # ee_ori = (ee_pos[:, :2] - link1_pos3d[:, :2])/torch.norm(ee_pos[:, :2] - link1_pos3d[:, :2], dim=1, keepdim=True)
    # dot_product = torch.sum(obj_ori * ee_ori, dim=1)
    # angles = torch.acos(dot_product)    # 弧度
    # weight = angles / torch.pi  # 差的越多，weight越大

    j1_p = env.scene["robot"].data.joint_pos[:, 0]
    reward = torch.zeros_like(j1_p)
    reward[j1_p < 0.785] = 1
    
    distance = torch.norm(handle_pos - bunker_pos, dim=1, p=2)
    
    # print(reward)
    return score * reward



# 摆动的时机确定
def help_joint1_rot(env: ManagerBasedRLEnv, max_dis_th, min_dis_th) -> torch.Tensor:
    j1_p = env.scene["robot"].data.joint_pos[:, 0]
    j1_v = env.scene["robot"].data.joint_vel[:, 0]
    
    reward = torch.zeros_like(j1_p)
    
    handle_pos = env.scene["object"].data.root_pos_w
    ee_pos = env.scene["ee_frame"].data.target_pos_w.squeeze(1)
    height_distance = torch.abs(handle_pos[:, 2] - ee_pos[:, 2])
    # reward[(height_distance < 0.04)] += 1

    bunker_pos = env.scene["robot"].data.root_pos_w
    distance = torch.norm(handle_pos[:, :2] - bunker_pos[:, :2], dim=1, p=2)
    dis_judge1 = (distance < max_dis_th).float()
    dis_judge2 = (distance > min_dis_th).float()   # 判断是否在这个区间
    
    # print(dis_judge2)
    
    j1_pos_judge = (j1_p > 1.1).float()
    
    if hasattr(env, 'j1_v_flag'):  # 判断是否第一次达到这个速度
        env.j1_v_flag[(env.j1_v_flag != 0) & (j1_v <= -0.1)] = 2
        env.j1_v_flag[(env.j1_v_flag == 0) & (j1_v <= -0.1)] = 1
    else:
        env.j1_v_flag = torch.zeros_like(dis_judge1)
        reward[(j1_v > 0)] = -3
        return reward
    
    reward[(dis_judge1 == 0) & (j1_v < 0)] -= 0.1  # 提早开始转的
    reward[(dis_judge1 == 1) & (dis_judge2 == 1) & (env.j1_v_flag == 1) & (j1_pos_judge == 1)] += 1
    
    reward[(reward > 0) & (height_distance < 0.04)] *= 1.5
    # print(reward)
    
    reward[(j1_v > 0)] = -3
    return reward
















# 在dis内手应满足半径差
def hand_radius(env: ManagerBasedRLEnv, dis_th) -> torch.Tensor:
    
    # 在底盘与物体的距离阈值内，手的半径要与物体距离相同
    ee_pos3d = env.scene["ee_frame"].data.target_pos_w.squeeze(1)
    link1_pos3d = env.scene["link1_frame"].data.target_pos_w.squeeze(1)
    obj_pos = env.scene["object"].data.root_pos_w
    obj_radius = torch.norm(obj_pos[:, :2] - link1_pos3d[:, :2], dim=1, keepdim=True)
    hand_radius = torch.norm(ee_pos3d[:, :2] - link1_pos3d[:, :2], dim=1, keepdim=True)
    difference = torch.abs(obj_radius - hand_radius).reshape(-1)    # 半径差

    # 车物距离
    bunker_pos = env.scene["robot"].data.root_pos_w[:, :2]
    distance = torch.norm(obj_pos[:, :2] - bunker_pos, dim=1, p=2)
    
    # import pdb;pdb.set_trace()
    reward = 1/(1+difference)
    reward[(distance>dis_th)] /= 10
    
    

    reward[obj_link1_dis > dis_th] = 0

    return reward.reshape(-1)







# 臂已经转到位置，但是没碰到物体
def touch_pani(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_pos3d = env.scene["ee_frame"].data.target_pos_w.squeeze(1)
    link1_pos3d = env.scene["link1_frame"].data.target_pos_w.squeeze(1)
    obj_pos3d = env.scene["object"].data.root_pos_w
    
    ee_ori = ee_pos3d[:, :2] - link1_pos3d[:, :2]
    ee_ori = ee_ori / torch.norm(ee_ori, dim=1, keepdim=True)
    
    obj_ori = obj_pos3d[:, :2] - link1_pos3d[:, :2]
    obj_ori = obj_ori / torch.norm(obj_ori, dim=1, keepdim=True)
    
    dot_product = torch.sum(ee_ori * obj_ori, dim=1)
    angles = torch.acos(dot_product)    # 弧度

    distance = torch.norm(ee_pos3d - obj_pos3d, dim=1, p=2)

    reward = torch.zeros_like(angles)
    reward[(angles < 0.05) & (distance > 0.04)] = -1
    
    return reward




# 手要保持水平
def hand_ori(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_pos = env.scene["ee_frame"].data.target_pos_w.squeeze(1)
    ee_que = env.scene["ee_frame"].data.target_quat_w.squeeze(1)
    num_env = ee_pos.shape[0]
    init_ori = torch.tensor([[1.0, 0.0, 0.0]]*num_env).to(ee_pos.device)
    hand_ori = rotate_vector_by_quaternion(init_ori, ee_que)
    need_ori = torch.tensor([[0.0, 0.0, -1]]*num_env).to(ee_pos.device)
    dot_product = torch.sum(hand_ori * need_ori, dim=-1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    weight = torch.acos(dot_product) / torch.pi    # 越小越好  0-1
    if torch.isnan(weight).any():
        weight = torch.nan_to_num(weight, nan=0.0)
    return 1/(1+weight)







# def ori_ee_obj(env: ManagerBasedRLEnv):
#     ee_pos = env.scene["ee_frame"].data.target_pos_w.squeeze(1)
#     ee_que = env.scene["ee_frame"].data.target_quat_w.squeeze(1)
#     obj_pos = env.scene["object"].data.root_pos_w

#     num_env = ee_pos.shape[0]
#     init_ori = torch.tensor([[0., 0., 1.]]*num_env).to(ee_pos.device)
#     hand_ori = rotate_vector_by_quaternion(init_ori, ee_que)
    
#     need_ori = obj_pos - ee_pos
#     need_norm = torch.norm(need_ori, dim=1, keepdim=True)
#     need_ori = need_ori / need_norm
    
#     dot_product = torch.sum(hand_ori * need_ori, dim=-1)
#     dot_product = torch.clamp(dot_product, -1.0, 1.0)
#     reward = torch.acos(dot_product) / torch.pi
#     if torch.isnan(reward).any():
#         reward = torch.nan_to_num(reward, nan=0.0)
#     return reward






# ======================================== catch ======================================

def judge_dis_v(env: ManagerBasedRLEnv):
    obj_pos = env.scene["object"].data.root_pos_w  # 物体的坐标
    ee_pos = env.scene["ee_frame"].data.target_pos_w.squeeze(1)
    obj_v = env.scene["object"].data.root_lin_vel_w
    robot_v = env.scene["robot"].data.root_lin_vel_w
    num_env = obj_v.shape[0]
    distances = torch.sqrt(torch.sum((obj_pos - ee_pos) ** 2, dim=1))
    dis_flag = (distances < 0.07).int().unsqueeze(1) 
    velocity_diff = torch.norm(obj_v - robot_v, dim=1)  
    v_flag = (velocity_diff < 0.01).float().view(num_env, 1)  # 小于 0.05 的地方为 1，其它为 0
    dis_v_mask = dis_flag * v_flag
    return dis_v_mask


def catch_judge(env: ManagerBasedRLEnv):  # 判断是否抓住
    dis_v_maks = judge_dis_v(env)
    num_env = dis_v_maks.shape[0]
    hand_pos = env.scene["robot"].data.joint_pos[:, 12:]
    close_judge = (hand_pos[:, 0] > 1).float().unsqueeze(1)  # 收紧为1，否则为0
    score = (dis_v_maks * close_judge).reshape(-1)     # 抓住为1，否则为0

    reward = torch.zeros_like(dis_v_maks)
    if hasattr(env, 'touch_time'):
        reward[(score==1)] += 5  # 此次碰撞的加分
        reward[(score==1)] += env.touch_time[(score==1)] * 5
        env.touch_time[(score==1)] += 1
        env.touch_time[(score==0)] = 0
    else:
        env.touch_time = torch.zeros_like(dis_v_maks)
        reward = score
    reward = reward.reshape(-1)
    return reward

def catch_reward_compute(env: ManagerBasedRLEnv) -> torch.Tensor:
    reward_wight = catch_judge(env)
    return reward_wight





def catch_pose_judge(env: ManagerBasedRLEnv, dis: float) -> torch.Tensor:
    handle_pos = env.scene["object"].data.root_pos_w  # shape: [num_envs,3]
    ee_pos = env.scene["ee_frame"].data.target_pos_w.squeeze(1)  # shape: [num_envs,3]
    distance = torch.norm(handle_pos - ee_pos, dim=1, p=2)
    dis_judge = (distance < dis).float().unsqueeze(1)
    
    hand_pos = env.scene["robot"].data.joint_pos[:, 12:]
    close_judge = (hand_pos[:, 0] > 1).float().unsqueeze(1)  # 收紧为1，否则为0

    reward_weight = torch.zeros_like(dis_judge)
    reward_weight[(dis_judge == 1) & (close_judge == 1)] = 10
    reward_weight[(dis_judge == 1) & (close_judge == 0)] = 0    
    reward_weight[(dis_judge == 0) & (close_judge == 1)] = 0
    reward_weight[(dis_judge == 0) & (close_judge == 0)] = 0

    return reward_weight.reshape(-1)



def help_catch_pose(env: ManagerBasedRLEnv, dis: float) -> torch.Tensor:
    handle_pos = env.scene["object"].data.root_pos_w  # shape: [num_envs,3]
    ee_pos = env.scene["ee_frame"].data.target_pos_w.squeeze(1)  # shape: [num_envs,3]
    distance = torch.norm(handle_pos - ee_pos, dim=1, p=2)
    dis_judge = (distance < dis).float().unsqueeze(1)
    
    hand_pos = env.scene["robot"].data.joint_pos[:, 12:]
    close_judge = (hand_pos[:, 0] > 1).float().unsqueeze(1)  # 收紧为1，否则为0

    reward_weight = torch.zeros_like(dis_judge)
    reward_weight[(dis_judge == 1) & (close_judge == 1)] = 1
    reward_weight[(dis_judge == 1) & (close_judge == 0)] = -0.1    
    reward_weight[(dis_judge == 0) & (close_judge == 1)] = 0
    reward_weight[(dis_judge == 0) & (close_judge == 0)] = 0.1

    return reward_weight.reshape(-1)