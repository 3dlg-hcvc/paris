import json
import open3d as o3d
import numpy as np

def get_rotation_axis_angle(k, theta):
    '''
    Rodrigues' rotation formula
    args:
    * k: direction unit vector of the axis to rotate about
    * theta: the (radian) angle to rotate with
    return:
    * 3x3 rotation matrix
    '''
    if np.linalg.norm(k) == 0.:
        return np.eye(3)
    k = k / np.linalg.norm(k)
    kx, ky, kz = k[0], k[1], k[2]
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.zeros((3, 3))
    R[0, 0] = cos + (kx**2) * (1 - cos)
    R[0, 1] = kx * ky * (1 - cos) - kz * sin
    R[0, 2] = kx * kz * (1 - cos) + ky * sin
    R[1, 0] = kx * ky * (1 - cos) + kz * sin
    R[1, 1] = cos + (ky**2) * (1 - cos)
    R[1, 2] = ky * kz * (1 - cos) - kx * sin
    R[2, 0] = kx * kz * (1 - cos) - ky * sin
    R[2, 1] = ky * kz * (1 - cos) + kx * sin
    R[2, 2] = cos + (kz**2) * (1 - cos)
    return R

def save_axis_mesh(k, center, filepath):
    '''support rotate only for now'''
    axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.02, cone_radius=0.04, cylinder_height=1.0, cone_height=0.08)
    arrow = np.array([0., 0., 1.], dtype=np.float32)
    n = np.cross(arrow, k) 
    rad = np.arccos(np.dot(arrow, k))
    R_arrow = get_rotation_axis_angle(n, rad)
    axis.rotate(R_arrow, center=(0, 0, 0))
    axis.translate(center[:3])
    o3d.io.write_triangle_mesh(filepath, axis)

def parse_json(meta):
    # translation = np.array(meta['translation'], dtype=np.float32)
    # center = np.array(meta['axis_o'], dtype=np.float32) - translation
    center = np.array(meta['axis_o'], dtype=np.float32)
    k = np.array(meta['axis_d'], dtype=np.float32)
    k = k / np.linalg.norm(k)

    return k, center

if __name__ == '__main__':
    exp_dir = '/localhome/jla861/Documents/SFU/Research/Articulated-NeRF/AN3/test_exp/sapien/stapler/103111/d2nerf/gttest-woratio-womax-wodistort@20221213-095152/save/it20000-test'
    json_path = f'{exp_dir}/transform.json'
    with open(json_path, 'r') as f:
        meta = json.load(f)
        f.close()
    k, center = parse_json(meta)
    save_axis_mesh(k, center, os.path.join(exp_dir, 'axis.ply'))