
from isaaclab_tasks.evolution_tasks.current_left_hand.current_left_hand_cfg import CURRENT_HAND_CFG as LEFT_HAND_CFG#hand cfg需要修改

from isaaclab_tasks.evolution_tasks.current_right_hand.current_right_hand_cfg import CURRENT_HAND_CFG  as RIGHT_HAND_CFG#hand cfg需要修改

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg


@configclass
class EventCfg:
    # Configuration for randomization.
    # -- robot
    # 定义材料相关属性
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("right_hand"),
            "static_friction_range": (0.7, 1.3), # 静摩擦范围
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0), # 回弹
            "num_buckets": 250,
        },
    )
    # 定义关节阻尼与刚度
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    # 定义关节范围
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )
    # tendon
    robot_tendon_properties = EventTerm(
        func=mdp.randomize_fixed_tendon_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_hand", fixed_tendon_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # -- object
    # 对象的材料属性
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )

    # 操作对象的质量分布
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- scene
    # 重力分布
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

from collections import defaultdict
@configclass
class EvolutionManipulationEnvCfg(DirectRLEnvCfg):
    # Configuration for the environment

    # urdf_path = "/home/qyx/Desktop/Isaaclab_other/agent_for_isaaclab/urdf/current_agent.urdf"
    # observation_number = 157

    # Actuated joints and fingertip links
    actuated_joint_names = ['link_0_0_to_link_2_0', 'link_2_0_to_link_2_1', 'link_2_1_to_link_2_2', 'link_2_2_to_link_2_3', 'link_0_0_to_link_3_0', 'link_3_0_to_link_3_1', 'link_3_1_to_link_3_2', 'link_3_2_to_link_3_3', 'link_0_0_to_link_4_0', 'link_4_0_to_link_4_1', 'link_4_1_to_link_4_2', 'link_4_2_to_link_4_3', 'link_0_0_to_link_5_0', 'link_5_0_to_link_5_1', 'link_5_1_to_link_5_2', 'link_5_2_to_link_5_3', 'link_3_3_to_link_3_3_child_3ec3ba39', 'link_4_3_to_link_4_3_child_e43bb520', 'link_5_3_to_link_5_3_child_73737248']
    finger_body_names = ['link_0_0', 'link_2_0', 'link_2_1', 'link_2_2', 'link_2_3', 'link_3_0', 'link_3_1', 'link_3_2', 'link_3_3', 'link_4_0', 'link_4_1', 'link_4_2', 'link_4_3', 'link_5_0', 'link_5_1', 'link_5_2', 'link_5_3', 'link_3_3_child_3ec3ba39', 'link_4_3_child_e43bb520', 'link_5_3_child_73737248']

    #分离出指尖
    finger_links = defaultdict(list)
    # Step 1: 分组
    for name in finger_body_names:
        parts = name.split('_')
        if len(parts) != 3:
            continue
        finger_id = parts[1]
        joint_id = int(parts[2])
        # 跳过手掌 link_0_0
        if finger_id == '0':
            continue
        finger_links[finger_id].append((joint_id, name))

    # Step 2: 找最大 joint_id 的 link
    fingertip_body_names = []
    
    for finger_id, joints in finger_links.items():
        max_joint = max(joints, key=lambda x: x[0])
        fingertip_body_names.append(max_joint[1])  # only name

    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = len(actuated_joint_names)
    observation_space = len(actuated_joint_names)*3+len(fingertip_body_names)*13+24  # (full)
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )


    
    # Robot configuration (using parsed joint names)
    #grasp_hand 位置还得改 左手
    robot_cfg: ArticulationCfg = LEFT_HAND_CFG.replace(prim_path="/World/envs/env_.*/LeftRobot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.35),
            rot=(-0.707107, 0.707107, 0.0, 0),
            joint_pos={".*": 0.0},
        )
    )
    
    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            
            usd_path=f"/home/qyx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/evolution_tasks/task_manipulation/DexCube/dex_cube_new.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.05, 0.01, 0.378), rot=(1.0, 0.0, 0.0, 0.0)), #pos=(0.0, -0.39, 0.6)
        
    )
   

    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"/home/qyx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/evolution_tasks/task_manipulation/DexCube/dex_cube_new.usd",
                scale=(0.5, 0.5, 0.5),
            )
        },
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=0.75, replicate_physics=True)
    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -5.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 1000
    fall_penalty = 0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.1
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0
