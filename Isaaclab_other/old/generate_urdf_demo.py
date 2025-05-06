import os
from urdfpy import URDF, Link, Joint, Visual, Geometry, Inertial, Collision, JointLimit
from urdfpy import Box, Cylinder
import open3d as o3d
import numpy as np
from urdfpy import Mesh

# 创建惯性
def create_inertial():
    inertia_matrix = [
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01]
    ]
    return Inertial(
        mass=0.5,
        inertia=inertia_matrix
    )


# 创建碰撞对象
def create_collision():
    return Collision(
        name="collision1",
        geometry=Geometry(box=Box([0.1, 0.1, 0.5])),
        origin=np.eye(4)
    )


# 创建齐次变换矩阵
def create_origin_translation(tx, ty, tz):
    matrix = np.eye(4)
    matrix[0, 3] = tx
    matrix[1, 3] = ty
    matrix[2, 3] = tz
    return matrix


# 输出 Mesh 文件保存目录
output_dir = "generated_meshes"
os.makedirs(output_dir, exist_ok=True)


# 创建机器人模型
link1_visual = Visual(
    geometry=Geometry(box=Box([0.1, 0.1, 0.5])),
    origin=create_origin_translation(0, 0, 0.25)
)
link2_visual = Visual(
    geometry=Geometry(cylinder=Cylinder(radius=0.05, length=0.4)),
    origin=create_origin_translation(0, 0, 0.2)
)

link1 = Link(
    name="link1",
    visuals=[link1_visual],
    inertial=create_inertial(),
    collisions=[create_collision()]
)
link2 = Link(
    name="link2",
    visuals=[link2_visual],
    inertial=create_inertial(),
    collisions=[create_collision()]
)

joint1 = Joint(
    name="joint1",
    parent="link1",
    child="link2",
    joint_type="revolute",
    axis=[0, 0, 1],
    origin=create_origin_translation(0, 0, 0.5),
    limit=JointLimit(lower=-1.57, upper=1.57, effort=100, velocity=1.0)
)

robot = URDF(name="simple_robot", links=[link1, link2], joints=[joint1])

# 为每个 Link 生成 Mesh 并保存
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

            else:
                print(f"Unknown geometry type in {link.name}, skipping...")
                continue

            # 计算法向量
            mesh.compute_vertex_normals()

            # 应用位姿变换
            translation = origin[:3, 3]
            rotation = origin[:3, :3]
            #mesh.translate(translation)
            #mesh.rotate(rotation, center=False)

            # 保存 Mesh 文件
            o3d.io.write_triangle_mesh(filename, mesh)

            # 更新 Visual 几何为 Mesh 文件路径
            visual.geometry = Geometry(mesh=Mesh(filename))

# 保存更新后的 URDF 文件
output_urdf = "simple_robot_with_mesh.urdf"
robot.save(output_urdf)
print(f"Updated URDF with meshes saved to: {output_urdf}")
