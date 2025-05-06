import numpy as np
from urdfpy import URDF, Link, Joint, Visual, Geometry, Inertial, Collision, JointLimit
from urdfpy import Box, Cylinder  # 导入 Box 和 Cylinder 类型
from urdfpy.utils import configure_origin  # 导入 configure_origin

# 生成齐次变换矩阵（平移部分）
def create_origin_translation(tx, ty, tz):
    # 生成单位矩阵，并添加平移部分
    matrix = np.eye(4)
    matrix[0, 3] = tx
    matrix[1, 3] = ty
    matrix[2, 3] = tz
    return matrix

# 创建惯性（使用 urdfpy.Inertial）
def create_inertial():
    # 定义惯性矩阵，使用 3x3 矩阵来表示惯性
    inertia_matrix = [
        [0.01, 0.0, 0.0],  # ixx, ixy, ixz
        [0.0, 0.01, 0.0],  # iyx, iyy, iyz
        [0.0, 0.0, 0.01]   # izx, izy, izz
    ]
    # 创建惯性对象
    return Inertial(
        mass=0.5,  # 示例质量值，可以根据需要修改
        inertia=inertia_matrix  # 传递正确格式的惯性矩阵
    )

# 创建碰撞体（可以为空的碰撞对象）
def create_collision():
    # 为空的碰撞对象，使用 Box 或 Cylinder 来定义碰撞几何体
    return Collision(
        name="collision1",  # 给碰撞体指定一个名称
        geometry=Geometry(box=Box([0.1, 0.1, 0.5])),  # 使用 Box 类型定义长方体碰撞体
        origin=create_origin_translation(0, 0, 0.25)  # 设置平移部分
    )

# 第一段机械臂 Link1
link1 = Link(
    name="link1",
    visuals=[
        Visual(
            geometry=Geometry(
                box=Box([0.1, 0.1, 0.5])  # 使用 Box 类型定义长方体
            ),
            origin=create_origin_translation(0, 0, 0.25)  # 平移 [0, 0, 0.25]
        )
    ],
    inertial=create_inertial(),  # 使用 Inertial 对象
    collisions=[create_collision()]  # 添加碰撞（此处使用碰撞对象）
)

# 第二段机械臂 Link2
link2 = Link(
    name="link2",
    visuals=[
        Visual(
            geometry=Geometry(
                cylinder=Cylinder(radius=0.05, length=0.4)  # 使用 Cylinder 类型定义圆柱体
            ),
            origin=create_origin_translation(0, 0, 0.2)  # 平移 [0, 0, 0.2]
        )
    ],
    inertial=create_inertial(),  # 使用 Inertial 对象
    collisions=[create_collision()]  # 添加碰撞（此处使用碰撞对象）
)

# 定义关节限制（使用 JointLimit）
joint_limit = JointLimit(
    lower=-1.57,  # 关节的最小角度（弧度）
    upper=1.57,   # 关节的最大角度（弧度）
    effort=100,   # 关节的最大力矩（可选）
    velocity=1.0  # 关节的最大角速度（可选）
)

# 定义关节
joint1 = Joint(
    name="joint1",
    parent="link1",
    child="link2",
    joint_type="revolute",  # 旋转关节
    axis=[0, 0, 1],   # 关节轴方向
    origin=configure_origin(create_origin_translation(0, 0, 0.5)),  # 使用 configure_origin 来创建齐次变换矩阵
    limit=joint_limit  # 使用 JointLimit 对象来设置关节限制
)

# 创建 URDF 机器人模型
robot = URDF(
    name="simple_robot",
    links=[link1, link2],
    joints=[joint1]
)

# 保存为 URDF 文件已生成
robot.save("simple_robot.urdf")
print("URDF 文件已生成: simple_robot.urdf")
