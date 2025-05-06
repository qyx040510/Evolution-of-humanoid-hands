import os
from urdfpy import URDF, Link, Joint, Visual, Geometry, Inertial, Collision, JointLimit
from urdfpy import Box, Cylinder, Mesh
import open3d as o3d
import numpy as np


# Helper function to create a transformation matrix
def create_origin_translation(tx, ty, tz):
    matrix = np.eye(4)
    matrix[0, 3] = tx
    matrix[1, 3] = ty
    matrix[2, 3] = tz
    return matrix


# Helper function to create inertial properties
def create_inertial():
    inertia_matrix = [
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01]
    ]
    return Inertial(
        mass=0.1,
        inertia=inertia_matrix
    )


# Helper function to create a collision object
def create_collision(geometry):
    if isinstance(geometry, Box):
        return Collision(
            name="collision_box",
            geometry=Geometry(box=geometry),
            origin=np.eye(4)
        )
    elif isinstance(geometry, Cylinder):
        return Collision(
            name="collision_cylinder",
            geometry=Geometry(cylinder=geometry),
            origin=np.eye(4)
        )
    else:
        return None  # Add additional conditions if necessary

# Create the palm
palm_visual = Visual(
    geometry=Geometry(box=Box([0.1, 0.02, 0.1])), # x, y, z
    origin=create_origin_translation(-0.02, 0, 0.0)
)
palm_collision = create_collision(Box([0.1, 0.2, 0.02]))  # Palm collision
palm = Link(
    name="palm",
    visuals=[palm_visual],
    inertial=create_inertial(),
    collisions=[palm_collision]  # Define collision geometry for palm
)

# Parameters for fingers
finger_length = [0.04, 0.03, 0.025]  # Length of each phalanx
finger_radius = 0.01
joint_limits = (-1.57, 1.57)

# Create fingers (index, middle, ring, little, thumb)
fingers = []
joints = []

# Finger origins relative to palm
finger_origins = [
    [-0.05, 0.0, 0.07],  # Thumb
    [-0.03, 0.0, 0.07],  # Index
    [0.0, 0.0, 0.07],    # Middle
    [0.03, 0.0, 0.07],   # Ring
    [0.05, 0.0, 0.07],   # Little
]

# Generate each finger
for i, origin in enumerate(finger_origins):
    parent_link = "palm"
    for j in range(len(finger_length)):
        link_name = f"finger{i+1}_link{j+1}"
        joint_name = f"finger{i+1}_joint{j+1}"

        # Create finger link
        finger_link = Link(
            name=link_name,
            visuals=[
                Visual(
                    geometry=Geometry(cylinder=Cylinder(radius=finger_radius, length=finger_length[j])),
                    origin=create_origin_translation(0, 0, finger_length[j] / 2)
                )
            ],
            inertial=create_inertial(),
            collisions=[create_collision(Cylinder(radius=finger_radius, length=finger_length[j]))]  # Define collision for each link
        )
        fingers.append(finger_link)

        # Create joint
        finger_joint = Joint(
            name=joint_name,
            parent=parent_link,
            child=link_name,
            joint_type="revolute",
            axis=[1, 0, 0],
            origin=create_origin_translation(*origin),
            limit=JointLimit(lower=joint_limits[0], upper=joint_limits[1], effort=10, velocity=1.0)
        )
        joints.append(finger_joint)

        # Update parent for next segment
        parent_link = link_name
        origin = [0, 0, finger_length[j]]

# Create URDF
robot = URDF(name="human_hand", links=[palm] + fingers, joints=joints)


# 输出 Mesh 文件保存目录
output_dir = "generated_meshes"
os.makedirs(output_dir, exist_ok=True)

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

            vertices = np.asarray(mesh.vertices)  # Get current vertices
            # Apply translation
            vertices += translation
            # Apply rotation
            vertices = np.dot(vertices, rotation.T)  # Matrix multiplication for rotation

            # Update mesh vertices
            mesh.vertices = o3d.utility.Vector3dVector(vertices)

            # 保存 Mesh 文件
            o3d.io.write_triangle_mesh(filename, mesh)

            # 更新 Visual 几何为 Mesh 文件路径
            visual.geometry = Geometry(mesh=Mesh(filename))

# 保存更新后的 URDF 文件
output_urdf = "hand.urdf"
robot.save(output_urdf)
print(f"Updated URDF with meshes saved to: {output_urdf}")

