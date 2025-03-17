"""Configuration for the Summit-Manipulation robots.

The following configurations are available:

* :obj:`SUMMIT_FRANKA_PANDA_CFG`: Summit base with Franka Emika arm
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg





BUNKER_ROBOT = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/th/WorkPlcae/iqebot_isaac_cfg/iqr_sk_robot_v3.1.usd/iqr_sk_robot_v3.urdf_test.usd",
        activate_contact_sensors=True,
        
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            # kinematic_enabled=False,
            # disable_gravity=False,
            # retain_accelerations=False,
            
            # linear_damping =10000.0,
            # angular_damping=1.0,
            # max_linear_velocity=1.48,
            # max_angular_velocity=58.0,
            
            max_depenetration_velocity=0.01,     # 最大去穿透速度，当物体发生重叠时，物理引擎会试图“推开”它们，使它们不再穿透
        ),
        
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # fix_root_link = False,       # Whether to fix the root link of the articulation.
            # articulation_enabled =True, # Whether to enable or disable articulation.
            enabled_self_collisions=True,  #启用自我碰撞
            solver_position_iteration_count=10,
            solver_velocity_iteration_count=10,
            sleep_threshold=0.001,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.00295),
        rot=(1.0, 0.0, 0.0, 0.0),  # rot=(0.707, 0.707, 0.0, 0.0),
        joint_pos={
            'joint1': 0.0,
            'joint2': -0.78,
            'joint3': -1.57,
            'joint4': 2.35,
            'joint5': 1.57,
            'joint6': 0.0,
            'a_.*':   0.0,
        },        
        joint_vel={
            "left_wheel_.*":  0.0,
            "right_wheel_.*": 0.0
        },
    ),
    

    actuators={
        
        '''
            左右轮的速度绝对值应相等
            30.015对应最大速度1.5m, 20差不多1m/s
            -12.5/12.5 即25的差对应旋转，差不多 1rad/s
        '''
        "bunker": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_.*", "right_wheel_.*"],
            effort_limit=140.0,
            stiffness=0.0,
            damping=100000.0,
        ),
        
        
        
        "cr5": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            velocity_limit = {
                  'joint1': 5.05,
                  'joint2': 5.45,
                  'joint3': 5.55,
                  'joint4': 5.55,
                  'joint5': 5.65,
                  'joint6': 5.76,
                },
            effort_limit=1000000.0,
            stiffness=300000.0,  # 钢度，难以弯曲、变形的程度。
            damping=8000.0,      # 阻尼的作用通常是减少系统的能量，使得系统的振动不会无限期地持续，而是迅速衰减。
        ),
        
        "leaphand": ImplicitActuatorCfg(
            joint_names_expr=["a_.*"],
            velocity_limit=500.0,
            effort_limit=1.0,
            stiffness=10000000.0,
            damping=0.0,
        ),
        
    },
)



SUMMIT_CFG = BUNKER_ROBOT.copy()  # type: ignore










