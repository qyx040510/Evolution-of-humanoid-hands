import open3d as o3d
from urdfpy import URDF
import numpy as np

# 加载 URDF 文件
robot = URDF.load("simple_robot_with_mesh.urdf")

# 保存几何体信息
geometries = []

# 遍历每个链接
for link in robot.links:
    if link.visuals:
        for visual in link.visuals:
            origin = visual.origin
            pose_matrix = origin if origin is not None else np.eye(4)

            # 处理 box 几何
            if hasattr(visual.geometry, "box"):
                size = visual.geometry.box.size
                box = o3d.geometry.TriangleMesh.create_box(*size)
                box.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色
                box.transform(pose_matrix)
                geometries.append(box)

            # 处理 cylinder 几何
            elif hasattr(visual.geometry, "cylinder"):
                radius = visual.geometry.cylinder.radius
                height = visual.geometry.cylinder.length
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius, height)
                cylinder.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色
                cylinder.transform(pose_matrix)
                geometries.append(cylinder)

# 渲染所有几何体
o3d.visualization.draw_geometries(geometries)
