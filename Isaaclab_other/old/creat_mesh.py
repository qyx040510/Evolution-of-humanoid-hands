import os
from urdfpy import URDF
import open3d as o3d
import numpy as np

# 输入 URDF 文件路径
urdf_file = "simple_robot.urdf"

# 输出 Mesh 文件保存目录
output_dir = "generated_meshes"
os.makedirs(output_dir, exist_ok=True)

# 加载 URDF 文件
robot = URDF.load(urdf_file)

# 遍历每个链接并生成对应的网格
for link in robot.links:
    if link.visuals:
        for i, visual in enumerate(link.visuals):
            geometry = visual.geometry
            origin = visual.origin if visual.origin is not None else np.eye(4)

            # 处理 Box 几何
            if hasattr(geometry, "box") and geometry.box:
                size = geometry.box.size
                mesh = o3d.geometry.TriangleMesh.create_box(*size)
                filename = os.path.join(output_dir, f"{link.name}_box_{i}.stl")

            # 处理 Cylinder 几何
            elif hasattr(geometry, "cylinder") and geometry.cylinder:
                radius = geometry.cylinder.radius
                length = geometry.cylinder.length
                mesh = o3d.geometry.TriangleMesh.create_cylinder(radius, length)
                filename = os.path.join(output_dir, f"{link.name}_cylinder_{i}.stl")

            # 处理 Sphere 几何
            elif hasattr(geometry, "sphere") and geometry.sphere:
                radius = geometry.sphere.radius
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius)
                filename = os.path.join(output_dir, f"{link.name}_sphere_{i}.stl")

            else:
                print(f"Unknown geometry type in {link.name}, skipping...")
                continue

            # 计算法向量
            mesh.compute_vertex_normals()

            # 应用位姿变换
            translation = origin[:3, 3]  # 提取平移
            rotation = origin[:3, :3]    # 提取旋转
            #mesh.translate(translation)  # 平移
            #mesh.rotate(rotation, center=False)  # 旋转

            # 保存 Mesh 为 STL 文件
            print(f"Saving mesh for {link.name} as {filename}")
            o3d.io.write_triangle_mesh(filename, mesh)
