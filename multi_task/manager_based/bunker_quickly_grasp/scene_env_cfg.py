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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

from isaaclab.assets import RigidObject, RigidObjectCfg

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg, OffsetCfg
from .robot_cfg import SUMMIT_CFG

from .agents import reward

from .agents import observation
from .agents import terminate


from isaaclab.markers.config import FRAME_MARKER_CFG
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


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
    
    robot: ArticulationCfg = SUMMIT_CFG.replace(prim_path="{ENV_REGEX_NS}/Bunker_Robot")

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.CuboidCfg(
            size=[0.07, 0.07, 0.07],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.2, 0.25, 0.83), rot=(1.0, 0.0, 0.0, 0.0)),
    ) 
    
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Bunker_Robot/arm_link_1",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Bunker_Robot/arm_link_6",
                    name="ee",
                    offset=OffsetCfg(
                        pos=(0.0, 0.03, 0.06),
                    ),
                ),
        ]
    )

    link1_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Bunker_Robot/base_footprint",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Bunker_Robot/arm_link_1",
                    name="link1",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.00),
                    ),
                ),
        ]
    )




    # # 用于转局部坐标系
    # camera_frame: FrameTransformerCfg = FrameTransformerCfg(
    #     prim_path="{ENV_REGEX_NS}/Bunker_Robot/base_footprint",
    #     debug_vis=False,
    #     # visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(
    #                 prim_path="{ENV_REGEX_NS}/Bunker_Robot/arm_link_6",
    #                 name="camera",
    #                 offset=OffsetCfg(
    #                     pos=(0.0, -0.1, 0.01),
    #                 ),
    #             ),
    #     ]
    # )
    




# action配置
@configclass
class ActionsCfg:
    # bunker_action = mdp.JointVelocityActionCfg(
    #     asset_name="robot", 
    #     joint_names=["left_wheel_base_3_joint","right_wheel_base_3_joint"],  
    #     scale=1.0,
    # )

    bunker_action = mdp.BinaryJointVelocityActionCfg(
        asset_name="robot", 
        joint_names=["left_wheel_base_3_joint", "right_wheel_base_3_joint"], 
        open_command_expr={
            'left_wheel_base_3_joint':  10.0,
            'right_wheel_base_3_joint': 10.0,
        },
        close_command_expr={
            'left_wheel_base_3_joint':  10.0,
            'right_wheel_base_3_joint': 10.0,
        }
    )

    cr5 = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint2", "joint3", "joint4", "joint6"],
        scale=1.0
    )
    
    cr5_joint1 = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint1"],
        open_command_expr={
            "joint1":  1.57,  # 2.17,
        },
        close_command_expr={
            "joint1": 0.0001,       #无限接近这个值
        }
    )   

    # cr5_joint5 = mdp.BinaryJointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=["joint5"], 
    #     open_command_expr={
    #         "joint5":  1.57,
    #     },
    #     close_command_expr={
    #         "joint5": 1.57,
    #     }
    # )   

    # leadphand = mdp.BinaryJointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=["a_1","a_2","a_3","a_5","a_6","a_7","a_9","a_10","a_11","a_12","a_13","a_14","a_15"], 
    #     open_command_expr={
    #         "a_1":  0.0,  # 0 .363,
    #         "a_2":  0.0,  # 0 .339,
    #         "a_3":  0.0,  # 0 .234,
    #         "a_12": 0.0,  # 0 .627,
    #         "a_13": 0.0,  # - 0.366,
    #         "a_14": 0.0,  # 0 .024,
    #         "a_15": 0.0,  # 0 .156,
    #         "a_5":  0.0,  # 0 .2613,
    #         "a_6":  0.0,  # 0 .2613,
    #         "a_7":  0.0,  # 0 .312,
    #         "a_9":  0.0,  # 0 .363,
    #         "a_10": 0.0,  # 0 .2613,
    #         "a_11": 0.0,  # 0 .2613,
    #     },
    #     close_command_expr={
    #         "a_1":  1.21,
    #         "a_2":  1.13,
    #         "a_3":  0.78,
    #         "a_12": 2.09,
    #         "a_13": -1.22,
    #         "a_14": 0.08,
    #         "a_15": 0.52,
    #         "a_5":  0.871,
    #         "a_6":  0.871,
    #         "a_7":  1.04,
    #         "a_9":  1.21,
    #         "a_10": 0.871,
    #         "a_11": 0.871,
    #     }
    # )   
    



@configclass
class ObservationsCfg:
    
    @configclass
    class PolicyCfg(ObsGroup):
        #机械臂 rad
        cr5_pos_1 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["joint1"])})  
        cr5_pos_2 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["joint2"])}) 
        cr5_pos_3 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["joint3"])}) 
        cr5_pos_4 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["joint4"])}) 
        # cr5_pos_5 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["joint5"])}) 
        cr5_pos_6 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["joint6"])}) 
        
        # #手 rad
        # # hand_pos_0  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_0"])})    
        # hand_pos_1  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_1"])})  
        # hand_pos_2  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_2"])}) 
        # hand_pos_3  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_3"])}) 
        # # hand_pos_4  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_4"])}) 
        # hand_pos_5  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_5"])}) 
        # hand_pos_6  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_6"])}) 
        # hand_pos_7  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_7"])}) 
        # # hand_pos_8  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_8"])}) 
        # hand_pos_9  = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_9"])}) 
        # hand_pos_10 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_10"])}) 
        # hand_pos_11 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_11"])}) 
        # hand_pos_12 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_12"])}) 
        # hand_pos_13 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_13"])}) 
        # hand_pos_14 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_14"])}) 
        # hand_pos_15 = ObsTerm(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=["a_15"])}) 
          
        # root_lin_v = ObsTerm(func=observation.robot_linear)
        # root_ang_v = ObsTerm(func=observation.robot_angular)
        # object_pos = ObsTerm(func=observation.obj_pos) 

        root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w)
        root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w)
        object_pos_w = ObsTerm(func=mdp.root_pos_w, params={'asset_cfg': SceneEntityCfg(name='object')}) 

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    



#事件配置
@configclass
class EventCfg:

    reset_cr5_joint_position = EventTerm(
        func=mdp.reset_joints_by_offset,    
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint.*"]),
            "position_range": (-0.000, 0.0000),
            "velocity_range": (-0.000, 0.0000),
        },
    )
    
    reset_leaphand_joint_position = EventTerm(
        func=mdp.reset_joints_by_offset,    
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["a_.*"]),
            "position_range": (-0.000, 0.0000),
            "velocity_range": (-0.000, 0.0000),
        },
    )
    
    reset_bunker_velocity = EventTerm(
        func=mdp.reset_joints_by_offset,    
        mode="reset",   
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["left_.*", "right_.*"]),
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        },
    )
       
    scene_reset = EventTerm(
        func=mdp.reset_root_state_uniform,    
        mode="reset",   
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "pose_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "velocity_range": {},
        },
    )

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "pose_range": {"x": [-0.1, 0.], "y": [-0.1, 0.1], "z": [-0.1, 0.1]},
            "velocity_range": {},
        },
    )

    reset_j1_history = EventTerm(func=terminate.reset_j1_history, mode="reset")





@configclass
class RewardsCfg:
    
    # =================================  底盘运动 ================================

    # 计算机械臂与物体2D朝向偏差，从而调整底盘朝向
    # obj_hand2d_ori = RewTerm(func=reward.orientation_obj_hand2d, weight=-7.0,)
    
    # # root速度值，正
    # joint_vel_overall = RewTerm(func=reward.velocity_bunker_overall, weight=2.0)
    
    # # #底盘与物体的距离，距离越小，reward越大
    # distance_bunker_handle = RewTerm(func=reward.distance_bunker_handle, weight=5.0, params={"threshold": 2})



    # =================================  臂运动 =================================

    # 手物距离判断是否成功触碰
    distance_ee_obj1 = RewTerm(func=reward.distance_success, weight=20.0, params={"dis_th": 0.1}) 
    distance_ee_obj2 = RewTerm(func=reward.distance_success, weight=50.0, params={"dis_th": 0.04})  

    # 手掌与物体高度奖励，高度越近，奖励越大
    height_ee_root = RewTerm(func=reward.height_ee_handle, weight=5.0,)
    
    # 定义摆的过程满足半径要求
    # hand_radius = RewTerm(func=reward.hand_radius, weight=12.0, params={"dis_th": 1.2})


    # 让手能摆过去，定义何时摆
    help_j1_rotate1 = RewTerm(func=reward.help_joint1_rot, weight=2.5,
                              params={"max_dis_th": 1.45, "min_dis_th": 1.2})
    # help_j1_rotate2 = RewTerm(func=reward.help_joint1_rot, weight=1.5,
    #                           params={"max_dis_th": 1.5})
    # help_j1_rotate3 = RewTerm(func=reward.help_joint1_rot, weight=0.9,
    #                           params={"max_dis_th": 1.75})

    
    # 让手水平
    hand_ori = RewTerm(func=reward.hand_ori, weight=3.0,)  




    # =================================  手运动 ==================================
       
    # 帮助手学会抓取动作
    # help_hand__catch_pose = RewTerm(func=reward.help_catch_pose, weight=4.0, params={'dis': 0.1})    # 0.09

    # # 抓住物体得分，抓住物体时间越长分越高
    # hand__catch = RewTerm(func=reward.catch_reward_compute, weight=5.0,)

    # # 判断手是否收紧，在dis内收紧加分
    # hand__catch_pose_judge = RewTerm(func=reward.catch_pose_judge, weight=6, params={'dis': 0.07})    # 0.09


    # =================================  其它 ==================================

    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=0.0001)
    
    joint_vel = RewTerm(
                func=mdp.joint_vel_l2,
                weight=0.0,
                params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=["joint.*"])},
                )








@configclass
class TerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    
    # dis_out = DoneTerm(func=terminate.distance_out, time_out=True)
    # j1_angle_out = DoneTerm(func=terminate.j1_angle_out, time_out=True)



@configclass
class CommandsCfg:
    pass


@configclass
class CurriculumCfg:
    pass

    # help_catch_pose = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "help_hand__catch_pose", "weight": 0.5, "num_steps": 2500}
    # )
    # catch_succe = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "hand__catch", "weight": 20, "num_steps": 2500}
    # )
    # grasp_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "hand__catch_pose_judge", "weight": 10.0, "num_steps": 2500}
    # )
    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 6000}
    # )
    # joint_vel = CurrTerm(
    #         func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    # )
    # height = CurrTerm(
    #         func=mdp.modify_reward_weight, params={"term_name": "height_ee_root", "weight": 5.0, "num_steps": 5500}
    # )



@configclass
class GraspEnvCfg(ManagerBasedRLEnvCfg):
    scene: GraspEnvSceneCfg = GraspEnvSceneCfg(num_envs=1, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2         # 每隔多少帧就进行一次物理计算
        self.episode_length_s = 2.5  # 每个训练回合（episode）的时长，秒
        
        # viewer settings
        self.viewer.eye = (10, 10, 10)
        
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation