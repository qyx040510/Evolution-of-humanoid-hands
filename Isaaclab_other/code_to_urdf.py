def generate_urdf_from_dict(agent_dict, output_dir="generated_meshes", output_urdf="robot.urdf"):
    import os
    import numpy as np
    from urdfpy import URDF, Link, Joint, Visual, Geometry, Inertial, Collision, JointLimit
    from urdfpy import Box, Cylinder, Mesh
    # import urdf
    import open3d as o3d
    from scipy.spatial.transform import Rotation as R

    def create_capsule_mesh(radius, length, resolution=20):
        """
        Create a capsule mesh by combining a cylinder and two hemispheres.
        """
        # Cylinder part
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius, length)
        cylinder.compute_vertex_normals()

        # Upper hemisphere
        hemisphere_top = o3d.geometry.TriangleMesh.create_sphere(radius, resolution)
        hemisphere_top.compute_vertex_normals()
        # Directly adjust vertices of the hemisphere
        hemisphere_top_vertices = np.asarray(hemisphere_top.vertices)
        hemisphere_top_vertices[:, 2] += length / 2  # Adjust Z-coordinate
        hemisphere_top.vertices = o3d.utility.Vector3dVector(hemisphere_top_vertices)

        # Lower hemisphere
        hemisphere_bottom = o3d.geometry.TriangleMesh.create_sphere(radius, resolution)
        hemisphere_bottom.compute_vertex_normals()
        # Directly adjust vertices of the hemisphere
        hemisphere_bottom_vertices = np.asarray(hemisphere_bottom.vertices)
        hemisphere_bottom_vertices[:, 2] -= length / 2  # Adjust Z-coordinate
        hemisphere_bottom.vertices = o3d.utility.Vector3dVector(hemisphere_bottom_vertices)

        # Combine meshes
        capsule = cylinder + hemisphere_top + hemisphere_bottom
        return capsule

    def create_origin_translation_rotation(tx, ty, tz, roll, pitch, yaw):
        """
        Create a 4x4 transformation matrix from translation and rotation (rpy).
        """
        translation = np.eye(4)
        translation[0, 3] = tx
        translation[1, 3] = ty
        translation[2, 3] = tz
        rotation = np.eye(4)
        rotation[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        return np.dot(translation, rotation)

    def create_inertial():
        """
        Define a default inertial matrix.
        """
        inertia_matrix = [
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.01]
        ]
        return Inertial(
            mass=0.1,
            inertia=inertia_matrix
        )

    def create_collision(geometry, name="collision_geometry"):
        """
        Create a collision object for the given geometry.
        """
        if isinstance(geometry, Geometry):
            return Collision(
                name=name,
                geometry=geometry,
                origin=np.eye(4)
            )
        return None

    # Process links
    links = []
    for link_data in agent_dict["base_link"] + agent_dict["links"]:
        name_code = link_data["name_code"]
        geometry_type = link_data.get("geometry_type")

        # Force convert `geometry_radius` and `geometry_length` to float
        geometry_radius = float(link_data.get("geometry_radius", 0.1))
        geometry_length = float(link_data.get("geometry_length", 0.1))

        if geometry_type == "capsule":
            # Generate capsule mesh and save as STL file, then use it for visual and collision geometries
            capsule_mesh = create_capsule_mesh(geometry_radius, geometry_length)
            stl_path = os.path.join(output_dir, f"{name_code}_capsule.stl")
            o3d.io.write_triangle_mesh(stl_path, capsule_mesh)
            visual_geometry = Geometry(mesh=Mesh(filename=stl_path))
            collision_geometry = Geometry(mesh=Mesh(filename=stl_path))
        elif geometry_type == "cylinder":
            visual_geometry = Geometry(cylinder=Cylinder(radius=geometry_radius, length=geometry_length))
            collision_geometry = Geometry(cylinder=Cylinder(radius=geometry_radius, length=geometry_length))
        elif geometry_type == "box":
            visual_geometry = Geometry(box=Box([geometry_radius] * 3))
            collision_geometry = Geometry(box=Box([geometry_radius] * 3))
        else:
            raise ValueError(f"Unsupported geometry type: {geometry_type}")

        link_visual = Visual(geometry=visual_geometry, origin=create_origin_translation_rotation(0, 0, geometry_length / 2, 0, 0, 0))
        link_collision = create_collision(
            collision_geometry,
            name=f"collision_{link_data['name_code']}"
        )
        link_inertial = create_inertial()

        link = Link(
            name=name_code,
            visuals=[link_visual],
            collisions=[link_collision],
            inertial=link_inertial
        )
        links.append(link)

    # Process joints
    joints = []
    for link_data in agent_dict["links"]:
        joint_name = link_data["joint_name"]
        parent_link = link_data.get("joint_parent", "base_link")
        child_link = link_data["name_code"]
        joint_type = link_data.get("joint_type", "revolute")
        joint_axis = link_data.get("joint_axis", [1, 0, 0])
        joint_limit = link_data.get("joint_limit", {"lower": -1.57, "upper": 1.57, "effort": 10.0, "velocity": 1.0})
        origin_translation = link_data.get("joint_origin_translation", [0, 0, 0])
        origin_rpy = link_data.get("joint_origin_rpy", [0, 0, 0])

        origin = create_origin_translation_rotation(*origin_translation, *origin_rpy)

        joint = Joint(
            name=joint_name,
            parent=parent_link,
            child=child_link,
            joint_type=joint_type,
            axis=joint_axis,
            origin=origin,
            limit=JointLimit(**joint_limit)
        )
        joints.append(joint)

    # Create the URDF robot
    robot = URDF(name=agent_dict["agent_code"], links=links, joints=joints)
    os.makedirs(output_dir, exist_ok=True)

    # Export meshes and attach to geometry
    for link in robot.links:
        if link.visuals:
            for i, visual in enumerate(link.visuals):
                geometry = visual.geometry
                origin = visual.origin if visual.origin is not None else np.eye(4)

                # 创建几何体并保存到 STL 文件
                if hasattr(geometry, "box") and geometry.box:
                    size = geometry.box.size
                    mesh = o3d.geometry.TriangleMesh.create_box(*size)
                    filename = os.path.join(output_dir, f"{link.name}_box.stl")
                elif hasattr(geometry, "cylinder") and geometry.cylinder:
                    radius = geometry.cylinder.radius
                    length = geometry.cylinder.length
                    #radius = geometry_radius  # Extract radius from the link data
                    #length = geometry_length
                    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius, length)
                    filename = os.path.join(output_dir, f"{link.name}_cylinder.stl")
                elif geometry_type == "capsule":
                    # Process Capsule geometry
                    radius = geometry_radius  # Extract radius from the link data
                    length = geometry_length  # Extract length from the link data
                    mesh = create_capsule_mesh(radius, length)
                    filename = os.path.join(output_dir, f"{link.name}_capsule.stl")
                    # Save the capsule mesh to an STL file
                    #mesh.compute_vertex_normals()
                    #o3d.io.write_triangle_mesh(filename, mesh)
                    # Update visual geometry to reference the generated STL file
                    #visual.geometry = Geometry(mesh=Mesh(filename))
                else:
                    print(f"Unknown geometry type in {link.name}, skipping...")
                    continue

                # 保存 Mesh 到 STL 文件
                mesh.compute_vertex_normals()
                translation = origin[:3, 3]
                rotation = origin[:3, :3]
                vertices = np.asarray(mesh.vertices)
                vertices += translation
                vertices = np.dot(vertices, rotation.T)
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                o3d.io.write_triangle_mesh(filename, mesh)

                # 更新 Visual 几何绑定到生成的 STL 文件路径
                visual.geometry = Geometry(mesh=Mesh(filename))
                #link.collisions = Geometry(mesh=Mesh(filename))

    # Save the URDF
    robot.save(output_urdf)
    print(f"URDF saved to {output_urdf}")

# from urdfpy import URDF, Link, Joint, Visual, Collision, Inertial, Geometry, Box, Cylinder, Mesh, JointLimit
# import open3d as o3d
# import numpy as np
# import os
# from scipy.spatial.transform import Rotation as R

# def generate_urdf_from_dict(agent_dict, output_dir="urdf_meshes", output_urdf="robot.urdf"):
#     """
#     从URDF字典生成URDF文件
#     参数:
#         agent_dict: 包含机器人结构的字典
#         output_dir: 网格文件输出目录
#         output_urdf: 生成的URDF文件路径
#     """
#     # 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)

#     # 辅助函数定义 ----------------------------------------------------------------
#     def create_capsule_mesh(radius, length):
#         """创建胶囊体网格"""
#         cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius, height=length)
#         sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
        
#         # 调整半球位置
#         top = sphere.translate((0, 0, length/2))
#         bottom = sphere.translate((0, 0, -length/2))
        
#         # 合并网格
#         capsule = cylinder + top + bottom
#         return capsule

#     def create_transform_matrix(translation, rotation):
#         """创建4x4变换矩阵 (x,y,z平移 + RPY旋转)"""
#         tf = np.eye(4)
#         tf[:3, 3] = translation
#         tf[:3, :3] = R.from_euler('xyz', rotation).as_matrix()
#         return tf

#     def create_inertial(mass=0.1):
#         """创建默认惯性矩阵"""
#         return Inertial(
#             mass=mass,
#             inertia=np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]),
#             origin=np.eye(4)
#         )

#     # 处理Links ----------------------------------------------------------------
#     links = []
#     for link_data in agent_dict["base_link"] + agent_dict.get("links", []):
#         # 提取参数
#         link_name = link_data["name_code"]
#         geo_type = link_data["geometry_type"]
#         radius = float(link_data.get("geometry_radius", 0.1))
#         length = float(link_data.get("geometry_length", 0.1))

#         # 创建几何体
#         if geo_type == "capsule":
#             mesh = create_capsule_mesh(radius, length)
#             stl_path = os.path.join(output_dir, f"{link_name}_capsule.stl")
#             o3d.io.write_triangle_mesh(stl_path, mesh)
#             geometry = Geometry(mesh=Mesh(filename=stl_path))
#         elif geo_type == "cylinder":
#             geometry = Geometry(cylinder=Cylinder(radius=radius, length=length))
#         elif geo_type == "box":
#             geometry = Geometry(box=Box(size=[radius*2]*3))  # 假设radius表示半长
#         else:
#             raise ValueError(f"不支持的几何类型: {geo_type}")

#         # 创建视觉和碰撞元素
#         visual = Visual(
#             geometry=geometry,
#             origin=create_transform_matrix([0,0,length/2], [0,0,0])  # 默认Z轴居中
#         )
#         collision = Collision(geometry=geometry, origin=visual.origin)
        
#         # 构建Link
#         link = Link(
#             name=link_name,
#             visuals=[visual],
#             collisions=[collision],
#             inertial=create_inertial()
#         )
#         links.append(link)

#     # 处理Joints ---------------------------------------------------------------
#     joints = []
#     for joint_data in agent_dict.get("joints", []):
#         joint = Joint(
#             name=joint_data["joint_name"],
#             joint_type=joint_data["joint_type"],
#             parent=joint_data["joint_parent"],
#             child=joint_data["name_code"],
#             axis=joint_data.get("joint_axis", [1,0,0]),
#             origin=create_transform_matrix(
#                 joint_data.get("joint_origin_translation", [0,0,0]),
#                 joint_data.get("joint_origin_rpy", [0,0,0])
#             ),
#             limit=JointLimit(
#                 lower=joint_data.get("joint_limit", {}).get("lower", -3.14),
#                 upper=joint_data.get("joint_limit", {}).get("upper", 3.14),
#                 effort=joint_data.get("joint_limit", {}).get("effort", 10),
#                 velocity=joint_data.get("joint_limit", {}).get("velocity", 1)
#             )
#         )
#         joints.append(joint)

#     # 构建并保存URDF -----------------------------------------------------------
#     robot = URDF(
#         name=agent_dict["agent_code"],
#         links=links,
#         joints=joints,
#         materials=[]
#     )
#     robot.save(output_urdf)
#     print(f"成功生成URDF文件: {os.path.abspath(output_urdf)}")
