import open3d as o3d
import numpy as np


def plot_camera(poses, exp_path, color=[0, 1, 1]):
    camera_meshes = o3d.geometry.TriangleMesh()

    for pose in poses:
        pose = np.array(pose)
        arrow_start = np.array([[0],[0],[0],[1]]).astype(np.float32)
        arrow_start = np.matmul(pose, arrow_start)

        camera_mesh=o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.75, cone_radius=1.25, cylinder_height=5.0, cone_height=4.0)
        # blender camera coordinates
        R_bcam = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float32) 
        # camera poses
        camera_mesh.rotate(R_bcam)
        camera_mesh.rotate(pose[:3, :3])
        camera_mesh.translate((arrow_start[0], arrow_start[1], arrow_start[2]), relative=False)

        camera_mesh.scale(0.02, center=camera_mesh.get_center())
        camera_mesh.paint_uniform_color(color)
        camera_meshes+=camera_mesh
    
    o3d.io.write_triangle_mesh(exp_path, camera_meshes)
