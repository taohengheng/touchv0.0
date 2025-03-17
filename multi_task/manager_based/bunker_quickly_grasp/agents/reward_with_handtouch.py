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


# def calculate_angle_deviation(vehicle_velocity, vehicle_position, object_position):
#     velocity_magnitude = torch.norm(vehicle_velocity, dim=1, keepdim=True)  # 计算速度向量的模
#     velocity_unit = vehicle_velocity / velocity_magnitude  # 速度单位向量
#     displacement = object_position - vehicle_position 
#     displacement_magnitude = torch.norm(displacement, dim=1, keepdim=True)  # 计算位置差向量的模
#     displacement_unit = displacement / displacement_magnitude  # 车到物体的单位方向向量
#     dot_product = torch.sum(velocity_unit * displacement_unit, dim=1, keepdim=True)
#     angle_deviation = torch.acos(dot_product)
#     angles_degrees = (angle_deviation / torch.pi) * 180  # 转换为度数
#     cross_product_z = displacement_unit[:, 0] * velocity_unit[:, 1] - displacement_unit[:, 1] * velocity_unit[:, 0]
#     sign = torch.sign(cross_product_z).reshape(-1, 1)
#     angles_degrees = angles_degrees * sign  # [-180, 180] 范围的角度
#     return angles_degrees


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




# ee与物体距离
def distance_ee_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    handle_pos = env.scene["object"].data.root_pos_w  # shape: [num_envs,3]
    ee_pos = env.scene["ee_frame"].data.target_pos_w.squeeze(1)  # shape: [num_envs,3]
    distance = torch.norm(handle_pos - ee_pos, dim=1, p=2)
    # reward = 1 / (1.0 + distance**2)
    # reward = torch.pow(reward, 2)
    reward = 1 / (1.0 + distance)
    return reward


def distance_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    handle_pos = env.scene["object"].data.root_pos_w  # shape: [num_envs,3]
    ee_pos = env.scene["ee_frame"].data.target_pos_w.squeeze(1)  # shape: [num_envs,3]
    distance = torch.norm(handle_pos - ee_pos, dim=1, p=2)
    reward = distance
    reward = torch.where(distance < 0.04, 10, 0)
    reward[handle_pos[:,2]>ee_pos[:,2]+0.01] = 0
    return reward



# ee与物体高度
def height_ee_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    handle_pos_height = env.scene["object"].data.root_pos_w[:, 2]  # shape: [num_envs,3]
    ee_pos_height = env.scene["ee_frame"].data.target_pos_w.squeeze(1)[:, 2]  # shape: [num_envs,3]
    distance = torch.abs(handle_pos_height - ee_pos_height)
    reward = 1 / (1.0 + distance)
    return reward


# # 希望手能朝向物体，方便抓取
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


# 希望手能朝向物体，方便抓取
def ori_ee(env: ManagerBasedRLEnv):
    ee_que = env.scene["ee_frame"].data.target_quat_w.squeeze(1)
    num_env = ee_que.shape[0]
    init_ori = torch.tensor([[0., 0., 1.]]*num_env).to(ee_que.device)
    hand_ori = rotate_vector_by_quaternion(init_ori, ee_que)
    hand_ori_xz = hand_ori[:, [0, 2]]
    hand_ori_xz_norm = hand_ori_xz / hand_ori_xz.norm(dim=1, keepdim=True)

    goal_ori = torch.tensor([[1.0, 0.0]]*num_env).to(ee_que.device)

    dot_product = torch.sum(hand_ori_xz_norm * goal_ori, dim=-1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    reward = torch.acos(dot_product) / torch.pi
    if torch.isnan(reward).any():
        reward = torch.nan_to_num(reward, nan=0.0)
    return reward



# 碰到为1 ，否则为0
def touch_obj_hand(env: ManagerBasedRLEnv, armlink_sensor_cfg: SceneEntityCfg,):
    armlink_contact_sensor: ContactSensor = env.scene.sensors[armlink_sensor_cfg.name]
    armlink_obj = (torch.sum(armlink_contact_sensor.data.force_matrix_w .squeeze(1).squeeze(1), dim=1, keepdim=True) != 0).float()
    # armlink_obj[(armlink_obj==0)] = -1
    reward = armlink_obj 
    return reward.reshape(-1)





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


def catch_judge(env: ManagerBasedRLEnv, hand_sensor_cfg_dict: dict):  # 判断是否抓住
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

def catch_reward_compute(env: ManagerBasedRLEnv, score: float, hand_sensor_cfg_dict: dict) -> torch.Tensor:
    reward_wight = catch_judge(env, hand_sensor_cfg_dict)
    return reward_wight





def catch_pose_judge(env: ManagerBasedRLEnv, dis: float) -> torch.Tensor:
    handle_pos = env.scene["object"].data.root_pos_w  # shape: [num_envs,3]
    ee_pos = env.scene["ee_frame"].data.target_pos_w.squeeze(1)  # shape: [num_envs,3]
    distance = torch.norm(handle_pos - ee_pos, dim=1, p=2)
    dis_judge = (distance < dis).float().unsqueeze(1)
    
    hand_pos = env.scene["robot"].data.joint_pos[:, 12:]
    close_judge = (hand_pos[:, 0] > 1).float().unsqueeze(1)  # 收紧为1，否则为0
    print(ee_pos)
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
    #print(close_judge)
    reward_weight = torch.zeros_like(dis_judge)
    reward_weight[(dis_judge == 1) & (close_judge == 1)] = 1
    reward_weight[(dis_judge == 1) & (close_judge == 0)] = -0.1    
    reward_weight[(dis_judge == 0) & (close_judge == 1)] = 0
    reward_weight[(dis_judge == 0) & (close_judge == 0)] = 0.1

    return reward_weight.reshape(-1)