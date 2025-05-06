"""主要的进化函数"""
from class_population import *

from variation import *
from evaluation_interface import evaluation

# 定义基本信息，不考虑更底层的性别因素
max_generation = 5000 # 最大变异代数
max_population = 1000 # 环境最大承载量，超过此量就要末位淘汰
max_variation = 10 # 每个个体下一代的最大变异数量，也是每个个体最多能出生的后代数，因为事实上每个个体都在变异
variation_probabilities = {  # 各项变异的发生概率
    "change_link_length": 1,
    "change_link_radius": 0,
    "remove_link": 0,
    "add_link": 0,
    "change_joint_origin_translation": 0,
    "change_joint_origin_rpy": 0, }
experiment_save_path = "exp_20250418_1" # 保存谱系图的路径

evaluation_taks={"Isaac-EvolutionHand-StoneGrind-v0"}
# evaluation_taks={"Isaac-Hand-Cube-Evolution-v0"}
check_point = "human"

# 给到isaaclab的数据存放地点，需要绝对路径 要改
# isaaclab_urdf_path = "/home/qyx/Desktop/Isaaclab_other/agent_for_isaaclab/urdf/current_agent.urdf"
# isaaclab_urdf_mesh_path = "/home/qyx/Desktop/Isaaclab_other/agent_for_isaaclab/mesh"
# isaaclab_urdf_code_path = ("/home/qyx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/evolution_tasks/current_evolution_agent/current_hand_cfg.py"
#                            )  # 这个要放到isaaclab文件夹中
# isaaclab_env_code_path =("/home/qyx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/evolution_tasks/"
#                            )

# isaaclab_test_result_path = ("/home/qyx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/evolution_tasks/logs/evaluation_hand"
#                              )#/home/qyx/Desktop/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/evolution_tasks/logs/evaluation_hand

isaaclab_urdf_path = "/home/qyx/Desktop/Isaaclab_other/agent_for_isaaclab/urdf/current_agent.urdf"
isaaclab_urdf_mesh_path = "/home/qyx/Desktop/Isaaclab_other/agent_for_isaaclab/mesh"
isaaclab_urdf_code_path = ("/home/qyx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/evolution_tasks/current_right_hand/current_right_hand_cfg.py"
                           )  # 这个要放到isaaclab文件夹中
isaaclab_env_code_path =("/home/qyx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/evolution_tasks/task_stone/evolution_stone_grind_env_cfg.py"
                           )

isaaclab_test_result_path = ("/home/qyx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/evolution_tasks/logs/evolution_hand_stone_grind"
                             )#/home/qyx/Desktop/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/evolution_tasks/logs/evaluation_hand
# /home/qyx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/evolution_tasks/logs/evolution_hand_stone_grind/

#mirror
isaaclab_mirror_urdf_path = "/home/qyx/Desktop/Isaaclab_other/agent_for_isaaclab_mirror/urdf/current_agent.urdf"
isaaclab_mirror_urdf_mesh_path = "/home/qyx/Desktop/Isaaclab_other/agent_for_isaaclab_mirror/mesh"
isaaclab_mirror_urdf_code_path = ("/home/qyx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/evolution_tasks/current_left_hand/current_left_hand_cfg.py"
                           )  # 这个要放到isaaclab文件夹中
# isaaclab_mirror_env_code_path =("/home/qyx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/evolution_tasks/task_stone/evolution_stone_grind_env_cfg.py"
#                            )

# isaaclab_mirror_test_result_path = ("/home/qyx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/evolution_tasks/logs/evolution_hand_stone_grind"
#                              )#/home/qyx/Desktop/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/evolution_tasks/logs/evaluation_hand



new_lineage = True

if new_lineage:
    # 初始化谱系
    hand_lineage = Lineage()



# 导入初始对象
if check_point == "human":
    from human_hand_agent import initial_agent_hand
elif check_point == "gorilla":
    from gorilla_hand_agent import initial_agent_hand

hand_lineage.add_individual(-1,0,initial_agent_hand,0,initial_agent_hand["evolution_id"])
# print(hand_lineage.lineage)
# 执行变异
# Iterate through generations
for current_generation in range(0, max_generation+1): #max_generation+1
    print(f"Generation {current_generation}: Starting mutation and evaluation.")

    # 获取这一代中的全部存活个体的代码，组成一个list
    surviving_individuals = hand_lineage.get_surviving_individuals_in_generation(current_generation)
    # 遍历这一代中的全部存活个体
    for current_individual in surviving_individuals:
        # 读取urdf信息
        # print(current_generation,current_individual)
        current_urdf = hand_lineage.lineage[(current_generation,current_individual)]['urdf_info']
        # print("current urdf:",current_urdf)
        for trail in range(max_variation):
            # 生成变异
            link_code, task_code, strength = choose_target(current_urdf, variation_probabilities)
            print("link_code, task_code, strength:",link_code, task_code, strength)
            # 执行变异
            success_tag, new_urdf = variation(current_urdf, link_code, task_code, strength,
                                            standard_variation=0.1, standard_length=0.05)
            print("success_tag, new_urdf:", success_tag)
            # 如果变异符合物理约束
            if success_tag:
                # 进行仿真评分
                current_score = evaluation(new_urdf, evaluation_taks,
                                           isaaclab_urdf_path, isaaclab_urdf_mesh_path, isaaclab_urdf_code_path,
                                           isaaclab_mirror_urdf_path, isaaclab_mirror_urdf_mesh_path, isaaclab_mirror_urdf_code_path,
                                           isaaclab_env_code_path, isaaclab_test_result_path)
                # 将个体放入谱系图
                hand_lineage.add_individual(current_generation,current_individual,new_urdf,current_score,uuid.uuid4().hex)
                print(f"trail;{trail}")
    # 每个个体都发生完变异后评价整个种群的状态，进行淘汰
    hand_lineage.evaluate_and_eliminate_individuals_in_generation(current_generation+1, max_population)
    # 储存当前谱系
    hand_lineage.save_to_file(experiment_save_path + '.json')

# 编写另一种加载训练一半的数据的方式