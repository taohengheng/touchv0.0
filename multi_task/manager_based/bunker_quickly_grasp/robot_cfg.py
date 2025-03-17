import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg




BUNKER_ROBOT = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/th/WorkPlcae/RotateGrasp/iqr_bunker/iqr_bunker.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            # max_linear_velocity=1.48,
            max_angular_velocity=58.0,
            max_depenetration_velocity=0.01,     # 最大去穿透速度，当物体发生重叠时，物理引擎会试图“推开”它们，使它们不再穿透
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # fix_root_link = False,       # Whether to fix the root link of the articulation.
            articulation_enabled=True,  # Whether to enable or disable articulation.
            enabled_self_collisions=True,  # 启用自我碰撞
            solver_position_iteration_count=10,
            solver_velocity_iteration_count=10,
            sleep_threshold=0.001,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),  # rot=(0.707, 0.707, 0.0, 0.0),
        joint_pos={
            'joint1': 1.57,  # 2.17,
            'joint2': -0.785,
            'joint3': -1.04,
            'joint4': 0.26,
            'joint5': 0.0,
            'joint6': 3.14,
            'a_.*': 0.0,
        },
        joint_vel={
            "left_wheel_base_3_joint":  0.0,
            "right_wheel_base_3_joint": 0.0,
        },
    ),
    
    actuators={
        "bunker": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_base_3_joint", "right_wheel_base_3_joint"],
            effort_limit={
                "left_wheel_base_3_joint": 856.0, 
                "right_wheel_base_3_joint":800.0    
            },
            stiffness=0.0,
            damping=100000.0,
        ),

        "cr5": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            velocity_limit=100.0,
            effort_limit={
                "joint1": 1800.0,  # 1175.0,  # 940.0,  # 235.0,
                "joint2": 1800.0,  # 1175.0,  # 940.0,  # 235.0,
                "joint3": 1800.0,  # 1175.0,  # 940.0,  # 235.0,
                "joint4": 1200.0,  # 820.0,  # 164.0,  # 41.0,
                "joint5": 1200.0,  # 820.0,  # 164.0,  # 41.0,
                "joint6": 1200.0,  # 820.0,  # 164.0,  # 41.0,
            },
            stiffness={
                "joint1": 1512.0,  # 2016.0,  # 1512.0,  # 1008.43,
                "joint2": 2600.0,  # 2016.0,  # 1512.0,  # 1008.43,
                "joint3": 2800.0,  # 2016.0,  # 1512.0,  # 1008.43,
                "joint4": 2100.0,  # 1140.0,  # 570.0,  # 380.0,
                "joint5": 1000000000.0,
                "joint6": 1500.0,  # 1140.0,  # 570.0,  # 380.0,
            },
            damping={
                "joint1": 600.0,
                "joint2": 600.0,
                "joint3": 603.0,
                "joint4": 152.0,
                "joint5": 1000000000.0,
                "joint6": 252.0,
            },
        ),
        
        "leaphand": ImplicitActuatorCfg(
            joint_names_expr=["a_.*"],
            effort_limit=50.0,
            velocity_limit=200.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

SUMMIT_CFG = BUNKER_ROBOT.copy()