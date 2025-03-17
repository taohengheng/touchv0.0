import torch
import math

from dataclasses import MISSING

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



# 基础场景配置
@configclass
class GraspEnvSceneCfg(InteractiveSceneCfg):
    
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    
    # 先声明为MISSING，后继承对象
    robot: ArticulationCfg = MISSING
    # ee_frame: FrameTransformerCfg = MISSING
    # contact_forces: ContactSensorCfg = MISSING
    
    
    #objects cfg
    '''
    '''
    
    
    
    
    

#action配置
@configclass
class ActionsCfg:
    pass

    #所有运动，包括臂和手的位置、底盘的速度，都是基于当前帧的偏移
    
    
    # # 轮子4个自由度
    bunker_action_left = mdp.JointVelocityActionCfg(
        asset_name="robot", 
        joint_names=["left_wheel_.*"], 
        scale=1.0
    )
    bunker_action_right = mdp.JointVelocityActionCfg(
        asset_name="robot", 
        joint_names=["right_wheel_.*"], 
        scale=1.0
    )
    
    
    # 机械臂6个自由度
    cr5 = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["joint.*"], 
        scale=1.0
    )


    
    # 手16个自由度
    leaphand = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["a_.*"], 
        scale=1.0
    )
  
    
    
    
    
    
    
    
    
    
    
    

#观察配置
@configclass
class ObservationsCfg:
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""


        # 可以直接返回每个关节弧度
        cr5_pos_1 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot',joint_names=["joint1"])})  
        cr5_pos_2 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot',joint_names=["joint2"])}) 
        cr5_pos_3 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot',joint_names=["joint3"])}) 
        cr5_pos_4 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot',joint_names=["joint4"])}) 
        cr5_pos_5 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot',joint_names=["joint5"])}) 
        cr5_pos_6 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot',joint_names=["joint6"])}) 
        
        
        
        # cr5_v_1 = ObsTerm(func=mdp.joint_vel, params={'asset_cfg': SceneEntityCfg(name='robot',joint_names=["joint1"])})  
        # cr5_v_2 = ObsTerm(func=mdp.joint_vel, params={'asset_cfg': SceneEntityCfg(name='robot',joint_names=["joint2"])}) 
        # cr5_v_3 = ObsTerm(func=mdp.joint_vel, params={'asset_cfg': SceneEntityCfg(name='robot',joint_names=["joint3"])}) 
        # cr5_v_4 = ObsTerm(func=mdp.joint_vel, params={'asset_cfg': SceneEntityCfg(name='robot',joint_names=["joint4"])}) 
        # cr5_v_5 = ObsTerm(func=mdp.joint_vel, params={'asset_cfg': SceneEntityCfg(name='robot',joint_names=["joint5"])}) 
        # cr5_v_6 = ObsTerm(func=mdp.joint_vel, params={'asset_cfg': SceneEntityCfg(name='robot',joint_names=["joint6"])}) 
        
        
        
        
        # 弧度
        hand_pos_0  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_0"])})    
        hand_pos_1  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_1"])})  
        hand_pos_2  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_2"])}) 
        hand_pos_3  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_3"])}) 
        hand_pos_4  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_4"])}) 
        hand_pos_5  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_5"])}) 
        hand_pos_6  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_6"])}) 
        hand_pos_7  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_7"])}) 
        hand_pos_8  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_8"])}) 
        hand_pos_9  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_9"])}) 
        hand_pos_10 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_10"])}) 
        hand_pos_11 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_11"])}) 
        hand_pos_12 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_12"])}) 
        hand_pos_13 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_13"])}) 
        hand_pos_14 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_14"])}) 
        hand_pos_15 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_15"])}) 
        
        
        
        
            
           
        # leftbase_vel_1 = ObsTerm(func=mdp.joint_vel, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["left_wheel_base_1_joint"])})
        # leftbase_vel_5 = ObsTerm(func=mdp.joint_vel, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["left_wheel_base_5_joint"])})
        # rightbase_vel_1= ObsTerm(func=mdp.joint_vel, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["right_wheel_base_1_joint"])})
        # rightbase_vel_5= ObsTerm(func=mdp.joint_vel, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["right_wheel_base_5_joint"])})

        
        # 线速度，角速度
        root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w)
        root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w)






        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False



    # observation groups
    policy: PolicyCfg = PolicyCfg()
    











#事件配置
@configclass
class EventCfg:
    '''
    EventCfg 类是用来配置和管理与仿真事件相关的参数。
    仿真事件是指在仿真过程中发生的各种状态变化或动作，例如碰撞检测、物体状态更新、传感器触发等。
    EventCfg 允许你定义这些事件的行为、频率和触发条件等。
    '''
    pass

    # # Reset the robot joints with offsets around the default position and velocity by the given ranges.
    # reset_cr5_joint_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,    
    #     mode="reset",   #reset时会触发该事件
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["joint.*"]),
    #         "position_range": (-0.000, 0.0000),
    #         "velocity_range": (-0.000, 0.0000),
    #     },
    # )
    
    reset_bunker_velocity = EventTerm(
        func=mdp.reset_joints_by_offset,    
        mode="reset",   #reset时会触发该事件
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["left_.*", "right_.*"]),
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        },
    )
    
    
    scene_reset = EventTerm(
        func=mdp.reset_scene_to_default,    
        mode="reset",   #reset时会触发该事件
        # params={
        #     "asset_cfg": SceneEntityCfg("robot", joint_names=["left_.*", "right_.*"]),
        #     "position_range": (0.0, 0.0),
        #     "velocity_range": (0.0, 0.0),
        # },
    )






@configclass
class RewardsCfg:
    pass








@configclass
class TerminationsCfg:
    '''
        用于管理强化学习环境中的 终止条件
        满足终止条件时，若定义了终止事件，则会执行
    '''
    pass
    time_out = DoneTerm(func=mdp.time_out, time_out=True)









@configclass
class GraspEnvCfg(ManagerBasedRLEnvCfg):
    
    # Scene settings
    scene: GraspEnvSceneCfg = GraspEnvSceneCfg(num_envs=1, env_spacing=4.0)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    
    #  Post initialization
    def __post_init__(self) -> None:
        """Post initialization. copy"""
        # general settings
        self.decimation = 2         #每隔多少帧就进行一次物理计算
        self.episode_length_s = 6  # 每个训练回合（episode）的时长，秒
        
        # viewer settings
        self.viewer.eye = (3, 3, 3)
        
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation