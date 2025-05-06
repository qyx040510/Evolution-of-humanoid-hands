from class_agent import UrdfAgent
from code_to_urdf import generate_urdf_from_dict
from mirror_agent import create_mirror_hand

from isaaclab_tool import parse_urdf_and_generate_articulation_cfg

"""定义标准手部节点"""
base_geometry = {
    "geometry_type": "capsule",  # where is the center
    "geometry_radius": 0.01,
    "geometry_length": 0.035,
}

# 定义多个链接和关节的参数
links = [
    {
        "name_code": "link_1_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.005,  # too wide
        "geometry_length": 0.04,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0],
        "joint_origin_rpy":[0, 1.57+0.3, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 1, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_1_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.022,
        "joint_parent": "link_1_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.04+0.005], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_1_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.003,
        "geometry_length": 0.015,
        "joint_parent": "link_1_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.022+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_2_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.005,
        "geometry_length": 0.045,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0.01],
        "joint_origin_rpy":[0, 1.57+0.1, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 1, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_2_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.028,
        "joint_parent": "link_2_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.045+0.005], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_2_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.02,
        "joint_parent": "link_2_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.028+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_2_3",
        "geometry_type": "capsule",
        "geometry_radius": 0.0025,
        "geometry_length": 0.015,
        "joint_parent": "link_2_2",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.02+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_3_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.048,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0.02],
        "joint_origin_rpy":[0, 1.57, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 1, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_3_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.029,
        "joint_parent": "link_3_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.048+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_3_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.0035,
        "geometry_length": 0.022,
        "joint_parent": "link_3_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.029+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_3_3",
        "geometry_type": "capsule",
        "geometry_radius": 0.0025,
        "geometry_length": 0.015,
        "joint_parent": "link_3_2",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.022+0.0035], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_4_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.003,
        "geometry_length": 0.04,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0.03],
        "joint_origin_rpy":[0, 1.57-0.1, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 1, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_4_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.029,
        "joint_parent": "link_4_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.04+0.003], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_4_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.0035,
        "geometry_length": 0.021,
        "joint_parent": "link_4_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.029+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_4_3",
        "geometry_type": "capsule",
        "geometry_radius": 0.0025,
        "geometry_length": 0.015,
        "joint_parent": "link_4_2",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.021+0.0035], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_5_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.003,
        "geometry_length": 0.039,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0.04],
        "joint_origin_rpy":[0, 1.57-0.2, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 1, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_5_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.003,
        "geometry_length": 0.02,
        "joint_parent": "link_5_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.039+0.003], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_5_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.003,
        "geometry_length": 0.015,
        "joint_parent": "link_5_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.02+0.003], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_5_3",
        "geometry_type": "capsule",
        "geometry_radius": 0.002,
        "geometry_length": 0.012,
        "joint_parent": "link_5_2",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.015+0.003], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
]


# 定义多个链接和关节的参数
mirror_links = [
    {
        "name_code": "link_1_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.005,  # too wide
        "geometry_length": 0.04,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0],
        "joint_origin_rpy":[0, 1.57-0.3, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 1, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_1_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.022,
        "joint_parent": "link_1_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.04+0.005], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_1_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.003,
        "geometry_length": 0.015,
        "joint_parent": "link_1_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.022+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_2_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.005,
        "geometry_length": 0.045,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0.01],
        "joint_origin_rpy":[0, 1.57-0.1, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 1, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_2_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.028,
        "joint_parent": "link_2_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.045+0.005], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_2_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.02,
        "joint_parent": "link_2_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.028+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_2_3",
        "geometry_type": "capsule",
        "geometry_radius": 0.0025,
        "geometry_length": 0.015,
        "joint_parent": "link_2_2",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.02+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_3_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.048,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0.02],
        "joint_origin_rpy":[0, 1.57, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 1, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_3_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.029,
        "joint_parent": "link_3_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.048+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_3_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.0035,
        "geometry_length": 0.022,
        "joint_parent": "link_3_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.029+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_3_3",
        "geometry_type": "capsule",
        "geometry_radius": 0.0025,
        "geometry_length": 0.015,
        "joint_parent": "link_3_2",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.022+0.0035], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_4_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.003,
        "geometry_length": 0.04,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0.03],
        "joint_origin_rpy":[0, 1.57+0.1, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 1, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_4_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.029,
        "joint_parent": "link_4_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.04+0.003], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_4_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.0035,
        "geometry_length": 0.021,
        "joint_parent": "link_4_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.029+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_4_3",
        "geometry_type": "capsule",
        "geometry_radius": 0.0025,
        "geometry_length": 0.015,
        "joint_parent": "link_4_2",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.021+0.0035], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_5_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.003,
        "geometry_length": 0.039,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0.04],
        "joint_origin_rpy":[0, 1.57+0.2, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 1, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_5_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.003,
        "geometry_length": 0.02,
        "joint_parent": "link_5_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.039+0.003], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_5_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.003,
        "geometry_length": 0.015,
        "joint_parent": "link_5_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.02+0.003], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_5_3",
        "geometry_type": "capsule",
        "geometry_radius": 0.002,
        "geometry_length": 0.012,
        "joint_parent": "link_5_2",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.015+0.003], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
]


# 创建 Agent 实例
initial_agent_hand = UrdfAgent(
    agent_code="initial_agent_hand",
    base_link_name="link_0_0",
    base_link_geometry=base_geometry,
    links=links
).to_dict()

mirror_agent_hand = UrdfAgent(
    agent_code="mirror_agent_hand",
    base_link_name="link_0_0",
    base_link_geometry=base_geometry,
    links=links
).to_dict()

import pybullet as p

# 加载模型





if __name__ == "__main__":
    # 查看生成的 agent 数据
    # print(f"initial_agent_hand:{initial_agent_hand}")
    mirror_hand=create_mirror_hand(initial_agent_hand,"mirror_hand")
    # print(f"mirror_hand:{mirror_hand}")
    # 调用函数
    generate_urdf_from_dict(initial_agent_hand, output_dir="output_meshes", output_urdf="human_hand_414.urdf")
    generate_urdf_from_dict(mirror_hand, output_dir="output_meshes_414", output_urdf="human_mirror_hand_414.urdf")
    parse_urdf_and_generate_articulation_cfg("/home/qyx/Desktop/Isaaclab_other/human_hand_414.urdf","/home/qyx/Desktop/Isaaclab_other/human_hand_414.urdf","/home/qyx/Desktop/Isaaclab_other/human_hand_414.py")
    parse_urdf_and_generate_articulation_cfg("/home/qyx/Desktop/Isaaclab_other/human_mirror_hand_414.urdf","/home/qyx/Desktop/Isaaclab_other/human_mirror_hand_414.urdf","/home/qyx/Desktop/Isaaclab_other/human_mirror_hand_414.py")
    # physicsClient = p.connect(p.GUI)
    # right = p.loadURDF("human_hand_414.urdf", [0, 0, 0])
    # left = p.loadURDF("human_mirror_hand_414.urdf", [0, 1, 0])

    # # 设置视角
    # p.resetDebugVisualizerCamera(
    #     cameraDistance=1.5,
    #     cameraYaw=0,
    #     cameraPitch=-30,
    #     cameraTargetPosition=[0, 0.5, 0]
    # )
    # # 保持窗口
    # while True:
    #     p.stepSimulation()
