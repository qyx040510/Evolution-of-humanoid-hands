
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.converters import UrdfConverterCfg

# Configuration based on URDF file /home/qyx/Desktop/Isaaclab_other/agent_for_isaaclab/urdf/current_agent.urdf
CURRENT_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="/home/qyx/Desktop/Isaaclab_other/agent_for_isaaclab/urdf/current_agent.urdf",  
        activate_contact_sensors=False,  # 禁用接触传感器模拟
        fix_base=True, #True
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # 禁用重力
            retain_accelerations=True,  # 保留加速度
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,  # 启用自碰撞
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type="force",
                                                   gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=30.1,damping=0.1)),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),  # 将驱动器类型设置为“强制”
        # fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),  # 配置具有刚度和阻尼的肌腱
    ),
    init_state=ArticulationCfg.InitialStateCfg(  # 在这里修改初始化姿态并没有用
        pos=(0.0, 0.0, 0.5),  # 初始位置
        rot=(0.0, 0.0, -0.7071, 0.7071),  # 四元数表示的初始方向
        joint_pos={".*": 0.0},  # 初始状态关节位置全部设为0
    ),
    actuators={

        "finger": ImplicitActuatorCfg(
            joint_names_expr=['link_0_0_to_link_1_0', 'link_1_0_to_link_1_1', 'link_1_1_to_link_1_2', 'link_0_0_to_link_2_0', 'link_2_0_to_link_2_1', 'link_0_0_to_link_3_0', 'link_3_0_to_link_3_1', 'link_3_1_to_link_3_2', 'link_3_2_to_link_3_3', 'link_0_0_to_link_4_0', 'link_4_0_to_link_4_1', 'link_4_1_to_link_4_2', 'link_4_2_to_link_4_3', 'link_0_0_to_link_5_0', 'link_5_0_to_link_5_1', 'link_5_1_to_link_5_2', 'link_5_2_to_link_5_3', 'link_5_3_to_link_5_3_child_67d97dfa', 'link_4_3_to_link_4_3_child_e4e32045', 'link_3_3_to_link_3_3_child_deb4d61f'],
            effort_limit_sim={'link_0_0_to_link_1_0': 15.0, 'link_1_0_to_link_1_1': 10.0, 'link_1_1_to_link_1_2': 5.0, 'link_0_0_to_link_2_0': 15.0, 'link_2_0_to_link_2_1': 10.0, 'link_0_0_to_link_3_0': 15.0, 'link_3_0_to_link_3_1': 10.0, 'link_3_1_to_link_3_2': 5.0, 'link_3_2_to_link_3_3': 5.0, 'link_0_0_to_link_4_0': 15.0, 'link_4_0_to_link_4_1': 10.0, 'link_4_1_to_link_4_2': 5.0, 'link_4_2_to_link_4_3': 5.0, 'link_0_0_to_link_5_0': 15.0, 'link_5_0_to_link_5_1': 10.0, 'link_5_1_to_link_5_2': 5.0, 'link_5_2_to_link_5_3': 5.0, 'link_5_3_to_link_5_3_child_67d97dfa': 5.0, 'link_4_3_to_link_4_3_child_e4e32045': 5.0, 'link_3_3_to_link_3_3_child_deb4d61f': 5.0},
            stiffness={'link_0_0_to_link_1_0': 1.0, 'link_1_0_to_link_1_1': 1.0, 'link_1_1_to_link_1_2': 1.0, 'link_0_0_to_link_2_0': 1.0, 'link_2_0_to_link_2_1': 1.0, 'link_0_0_to_link_3_0': 1.0, 'link_3_0_to_link_3_1': 1.0, 'link_3_1_to_link_3_2': 1.0, 'link_3_2_to_link_3_3': 1.0, 'link_0_0_to_link_4_0': 1.0, 'link_4_0_to_link_4_1': 1.0, 'link_4_1_to_link_4_2': 1.0, 'link_4_2_to_link_4_3': 1.0, 'link_0_0_to_link_5_0': 1.0, 'link_5_0_to_link_5_1': 1.0, 'link_5_1_to_link_5_2': 1.0, 'link_5_2_to_link_5_3': 1.0, 'link_5_3_to_link_5_3_child_67d97dfa': 1.0, 'link_4_3_to_link_4_3_child_e4e32045': 1.0, 'link_3_3_to_link_3_3_child_deb4d61f': 1.0},
            damping={'link_0_0_to_link_1_0': 0.1, 'link_1_0_to_link_1_1': 0.1, 'link_1_1_to_link_1_2': 0.1, 'link_0_0_to_link_2_0': 0.1, 'link_2_0_to_link_2_1': 0.1, 'link_0_0_to_link_3_0': 0.1, 'link_3_0_to_link_3_1': 0.1, 'link_3_1_to_link_3_2': 0.1, 'link_3_2_to_link_3_3': 0.1, 'link_0_0_to_link_4_0': 0.1, 'link_4_0_to_link_4_1': 0.1, 'link_4_1_to_link_4_2': 0.1, 'link_4_2_to_link_4_3': 0.1, 'link_0_0_to_link_5_0': 0.1, 'link_5_0_to_link_5_1': 0.1, 'link_5_1_to_link_5_2': 0.1, 'link_5_2_to_link_5_3': 0.1, 'link_5_3_to_link_5_3_child_67d97dfa': 0.1, 'link_4_3_to_link_4_3_child_e4e32045': 0.1, 'link_3_3_to_link_3_3_child_deb4d61f': 0.1},
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
