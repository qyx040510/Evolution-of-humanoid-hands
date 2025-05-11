"""进化程序与仿真引擎的接口
如何通过代码控制isaaclab"""

from code_to_urdf import generate_urdf_from_dict
from isaaclab_tool import parse_urdf_and_generate_articulation_cfg, task_generate_env_cfg, calculate_observation_number
import os
import glob
import subprocess
from read_results import check_finished_folder_exists, get_latest_reward
import time
from mirror_agent import create_mirror_hand
def evaluation(urdf_dic, evaluation_tasks, 
               isaaclab_urdf_path, isaaclab_urdf_mesh_path,isaaclab_urdf_code_path,
               isaaclab_mirror_urdf_path, isaaclab_mirror_urdf_mesh_path, isaaclab_mirror_urdf_code_path,
                isaaclab_env_code_path, isaaclab_test_result_path):
    # 删除原路径下的urdf与mesh文件
    delete_urdf_and_stl_files(isaaclab_urdf_path, isaaclab_urdf_mesh_path)
    # 从urdf_dic生成urdf文件
    generate_urdf_from_dict(urdf_dic, output_dir=isaaclab_urdf_mesh_path, output_urdf=isaaclab_urdf_path)
    # 根据urdf生成isaaclab的对象定义文件
    parse_urdf_and_generate_articulation_cfg(isaaclab_urdf_path, isaaclab_urdf_path, isaaclab_urdf_code_path)

    #生成镜像手的相关文件
    mirror_hand= create_mirror_hand(urdf_dic,"mirror_agent_hand") 
    # 删除原路径下的urdf与mesh文件
    delete_urdf_and_stl_files(isaaclab_mirror_urdf_path, isaaclab_mirror_urdf_mesh_path)
    # 从urdf_dic生成urdf文件
    generate_urdf_from_dict(mirror_hand, output_dir=isaaclab_mirror_urdf_mesh_path, output_urdf=isaaclab_mirror_urdf_path)
    # 根据urdf生成isaaclab的对象定义文件
    parse_urdf_and_generate_articulation_cfg(isaaclab_mirror_urdf_path, isaaclab_mirror_urdf_path, isaaclab_mirror_urdf_code_path)
    # calculate observation number
    observation_number = calculate_observation_number(isaaclab_urdf_path)
    # print(f"11111111:{evaluation_tasks}")
    # 循环任务列表
    for current_task in evaluation_tasks:
        # generate tasks cfg according to current hand config 根据对象情况，修改任务环境
        # print(f"task:{current_task}")
        task_generate_env_cfg(current_task, isaaclab_urdf_path, isaaclab_mirror_urdf_path, observation_number, isaaclab_env_code_path)
        # 用命令行控制执行
        score = run_isaaclab_simulation(current_task, isaaclab_test_result_path)
        # 将isaaclab的结果储存为一个文件
        # 进化程序基本写完了，下面就要研究isaaclab的评分与储存等等了
        # read score through file name
        # 代码的加速等等就不是我们的研究问题了

    return score





def delete_urdf_and_stl_files(directory_urdf, directory_mesh):
    """
    删除指定目录下的所有 .urdf 和 .stl 文件。

    :param directory: (str) 目录路径
    :return: None
    """
    # 查找所有 .urdf 文件
    urdf_files = glob.glob(os.path.join(directory_urdf, '*.urdf'))
    # 查找所有 .stl 文件
    stl_files = glob.glob(os.path.join(directory_mesh, '*.stl'))

    # 合并 .urdf 和 .stl 文件列表
    all_files = urdf_files + stl_files

    # 删除文件
    for file in all_files:
        try:
            os.remove(file)
            #print(f"删除了文件: {file}")
        except Exception as e:
            print(f"删除文件 {file} 时出错: {e}")

# 检查是否有正在运行的 isaaclab 进程
def check_previous_simulation():
    result = subprocess.run("ps aux | grep isaaclab", shell=True, capture_output=True, text=True)
    return result.stdout

import gc
# import torch
import time

def run_isaaclab_simulation(task_name, isaaclab_test_result_path, num_envs=64):
    # Conda 环境的激活命令
    conda_activate_cmd = 'conda activate isaaclab_sim45'  # conda环境名称 source

    # 要执行的命令
    # ./isaaclab.sh -p /home/qyx/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/evolution_tasks/train_interface.py --num_envs=64 --task=Isaac-Hand-Cube-Evolution-v0
    command = [
        '/home/qyx/IsaacLab/isaaclab.sh',
        '-p', '/home/qyx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/evolution_tasks/train_interface.py',
        '--num_envs', str(num_envs),
        '--task', str(task_name),
        '--headless'
    ]
    shell_cmd = f"""
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate isaaclab_sim45
    {' '.join(command)}
    """

    process = subprocess.Popen(
        ["bash", "-c", shell_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    # process = subprocess.Popen(
    # ["bash","-c", f"{conda_activate_cmd} && {' '.join(command)}"],
    # stdout=subprocess.PIPE,
    # stderr=subprocess.PIPE,
    # text=True
    # )

    stdout, stderr = process.communicate()  # 会等待完成
    print(f"stdout:{stdout}")
    print(f"stderr:{stderr}")
    print("Simulation finished.")

    # 延迟一下，确保资源释放
    time.sleep(3)

    # 手动清理 CUDA 上下文（如果你用 PyTorch）
    # torch.cuda.empty_cache()
    # gc.collect()
    # try:
    #     # 激活 Conda 环境并运行命令
    #     result = subprocess.run(f"bash -i -c '{conda_activate_cmd} && {' '.join(command)}'", check=True, stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,stdin=subprocess.DEVNULL, shell=True)
    #     print("Simulation completed successfully.")
    #     # 输出命令执行结果
    #     # print("Simulation output:")
    #     # print(result.stdout)
    #     # print("Simulation errors:")
    #     # print(result.stderr)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error executing command with return code {e.returncode}")
    #     # print(f"Error executing command: {e}")
    #     # print(f"Error output: {e.stderr}")

    # full_cmd = f"bash -i -c '{conda_activate_cmd} && {' '.join(command)}'"

    # command = [
    # "conda", "run", "-n", "isaaclab_sim45",
    # "/home/qyx/IsaacLab/isaaclab.sh",
    # "-p", "/home/qyx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/evolution_tasks/train_interface.py",
    # "--num_envs", str(num_envs),
    # "--task", str(task_name),
    # # "--headless"
    # ]
    
    # # previous_simulation = check_previous_simulation()

    # # if previous_simulation:
    # #     print("Previous simulation is still running. Waiting for completion...")
    # #     time.sleep(5)  # 等待前一个仿真结束
    # print("Starting simulation...\n")
    # process = subprocess.Popen(command, 
    #                            stdout=subprocess.DEVNULL, 
    #                            stderr=subprocess.DEVNULL, 
    #                            shell=True)

    # # 实时读取输出
    # for line in process.stdout:
    #     print(line, end='', flush=True)

    # process.wait()  # 等待命令结束

    # if process.returncode != 0:
        
    #     print(f"\nSimulation exited with code {process.returncode}")

    # 执行循环，读取tag，要降低读取频率到1-5秒
    
    tag = check_finished_folder_exists(isaaclab_test_result_path)
    if tag:
        score = get_latest_reward(isaaclab_test_result_path)
        return score
    time.sleep(10)
    print("simulation running")
    score=float('-inf')
    return score

# 调用函数执行仿真
#run_isaaclab_simulation('Isaac-Hand-Cube-Demo-Direct-v0')

