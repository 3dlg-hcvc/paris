
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
import torch
import os.path as osp
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



def combine_pred_mesh(paths, exp_path):
    recon_mesh = o3d.geometry.TriangleMesh()
    for path in paths:
        mesh = o3d.io.read_triangle_mesh(path)
        recon_mesh += mesh
    o3d.io.write_triangle_mesh(exp_path, recon_mesh)


def compute_chamfer(recon_pts,gt_pts):
	with torch.no_grad():
		recon_pts = recon_pts.cuda()
		gt_pts = gt_pts.cuda()
		dist,_ = chamfer_distance(recon_pts,gt_pts,batch_reduction=None)
		dist = dist.item()
	return dist

def compute_recon_error(recon_path, gt_path, n_samples=10000, vis=False):
    verts, faces = load_ply(recon_path)
    recon_mesh = Meshes(verts=[verts], faces=[faces])
    verts, faces = load_ply(gt_path)
    gt_mesh = Meshes(verts=[verts], faces=[faces])

    gt_pts = sample_points_from_meshes(gt_mesh, num_samples=n_samples)
    recon_pts = sample_points_from_meshes(recon_mesh, num_samples=n_samples)


    if vis:
        pts = gt_pts.clone().detach().squeeze().numpy()
        gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        o3d.io.write_point_cloud("gt_points.ply", gt_pcd)
        pts = recon_pts.clone().detach().squeeze().numpy()
        recon_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        o3d.io.write_point_cloud("recon_points.ply", recon_pcd)

    return (compute_chamfer(recon_pts, gt_pts) + compute_chamfer(gt_pts, recon_pts)) * 0.5
    

# def eval_d2neus_combine():
#     case = 'USB/100109'
#     model = 'd2neus'
#     trial = 'mlr-gttest-womax-wobias-no_grid@20221218-200019'
#     exp_name = f'exp/sapien/{case}/{model}/{trial}'
#     iteration = '20000'
#     part_paths = [
#         f'{exp_name}/save/it{iteration}_dynamic_256_thre0.obj',
#         f'{exp_name}/save/it{iteration}_static_256_thre0.obj'
#     ]

#     # obj_gt_path = f'data/sapien/{case}/textured_objs/canonical/canonical_rotate.obj'
#     ply_gt_path = f'data/sapien/{case}/textured_objs/canonical/canonical_rotate.ply'

#     # m = o3d.io.read_triangle_mesh(obj_gt_path)
#     # o3d.io.write_triangle_mesh(ply_gt_path, m)


#     ply_combine_path = f'{exp_name}/save/it{iteration}_combine_256_thre0.ply'
#     combine_pred_mesh(part_paths, ply_combine_path)

#     chamfer_dist = compute_recon_error(ply_combine_path, ply_gt_path, vis=False)

#     with open(osp.join(f'{exp_name}/save/' f'it{iteration}_256_thre0_chamfer.txt'), 'w') as f:
#         f.write(f'{chamfer_dist}\n')
#         f.close()

# def eval_d2neus_part():
#     def obj2ply(name):
#         m = o3d.io.read_triangle_mesh(f'{name}.obj')
#         o3d.io.write_triangle_mesh(f'{name}.ply', m)
#     def combine_mesh(names):
#         recon_mesh = o3d.geometry.TriangleMesh()
#         mesh_s = o3d.io.read_triangle_mesh(names['static'] + '.obj')
#         recon_mesh += mesh_s
#         mesh_d = o3d.io.read_triangle_mesh(names['dynamic'] + '.obj')
#         recon_mesh += mesh_d
#         o3d.io.write_triangle_mesh(names['combine'] + '.ply', recon_mesh)

#     case = 'fridge/12066'
#     model = 'd2neus'
#     trial = 'mlr-gttest-womax-wobias-no_grid@20221218-221149'
#     exp_name = f'exp/sapien/{case}/{model}/{trial}'
#     iteration = '20000'
#     part_names = {
#         # 'combine': f'{exp_name}/save/it{iteration}_combine_256_thre0',
#         'static': f'{exp_name}/save/it{iteration}_static_256_thre0',
#         'dynamic': f'{exp_name}/save/it{iteration}_dynamic_256_thre0'
#     }

#     static_gt_path = f'data/sapien/{case}/textured_objs/canonical/canonical_static_rotate.ply'
#     dynamic_gt_path = f'data/sapien/{case}/textured_objs/canonical/canonical_dynamic_rotate.ply'
#     # combine_gt_path = f'data/sapien/{case}/textured_objs/canonical/canonical_rotate.ply'

#     # convert to ply format
#     # combine_mesh(part_names)
#     obj2ply(part_names['static'])
#     obj2ply(part_names['dynamic'])

#     # compute distance
#     chamfer_dist_dynamic = compute_recon_error(part_names['dynamic']+'.ply', dynamic_gt_path, n_samples=10000, vis=False)

#     chamfer_dist_static = compute_recon_error(part_names['static']+'.ply', static_gt_path, n_samples=10000, vis=False)
#     # chamfer_dist_combine = compute_recon_error(part_names['combine']+'.ply', combine_gt_path, n_samples=10000, vis=False)



#     with open(osp.join(f'{exp_name}/save/' f'it{iteration}_256_thre0_chamfer.txt'), 'w') as f:
#         f.write(f'static: {chamfer_dist_static}\n')
#         f.write(f'dynamic: {chamfer_dist_dynamic}\n')
#         # f.write(f'combine: {chamfer_dist_combine}\n')
#         f.close()


# def eval_d2nerf_part(clean=False):

#     def obj2ply(name):
#         m = o3d.io.read_triangle_mesh(f'{name}.obj')
#         o3d.io.write_triangle_mesh(f'{name}.ply', m)
#     def combine_mesh(names):
#         recon_mesh = o3d.geometry.TriangleMesh()
#         mesh_s = o3d.io.read_triangle_mesh(names['static'] + '.obj')
#         recon_mesh += mesh_s
#         mesh_d = o3d.io.read_triangle_mesh(names['dynamic'] + '.obj')
#         recon_mesh += mesh_d
#         o3d.io.write_triangle_mesh(names['combine'] + '.ply', recon_mesh)

#     case = 'fridge/12066'
#     model = 'd2nerf'
#     trial = 'long-gttest-wmax-wodistort-no_grid@20221218-172523'
#     exp_name = f'exp/sapien/{case}/{model}/{trial}'
#     iteration = '20000'
#     threshold = 3.0
#     res = 256
#     part_names = {
#         # 'combine': f'{exp_name}/save/it{iteration}_combine_256_thre0',
#         'static': f'{exp_name}/save/it{iteration}_static_{res}_thre{threshold}',
#         'dynamic': f'{exp_name}/save/it{iteration}_dynamic_{res}_thre{threshold}'
#     }

#     static_gt_path = f'data/sapien/{case}/textured_objs/canonical/canonical_static_rotate.ply'
#     dynamic_gt_path = f'data/sapien/{case}/textured_objs/canonical/canonical_dynamic_rotate.ply'
#     # combine_gt_path = f'data/sapien/{case}/textured_objs/canonical/canonical_rotate.ply'

#     # convert to ply format
#     # combine_mesh(part_names)
#     obj2ply(part_names['static'])
#     obj2ply(part_names['dynamic'])

#     # compute distance
#     chamfer_dist_dynamic = compute_recon_error(part_names['dynamic']+'.ply', dynamic_gt_path, n_samples=10000, vis=False)

#     chamfer_dist_static = compute_recon_error(part_names['static']+'.ply', static_gt_path, n_samples=10000, vis=False)
#     # chamfer_dist_combine = compute_recon_error(part_names['combine']+'.ply', combine_gt_path, n_samples=10000, vis=False)



#     with open(osp.join(f'{exp_name}/save/' f'it{iteration}_{res}_thre{threshold}_chamfer.txt'), 'w') as f:
#         f.write(f'static: {chamfer_dist_static}\n')
#         f.write(f'dynamic: {chamfer_dist_dynamic}\n')
#         # f.write(f'combine: {chamfer_dist_combine}\n')
#         f.close()


def eval_CD(pred_s_ply, pred_d_ply, pred_w_ply, gt_s_ply, gt_d_ply, gt_w_ply):
    # combine the part meshes as a whole
    combine_pred_mesh([pred_s_ply, pred_d_ply], pred_w_ply)

    # resave the mesh just in case the original one is broken
    # mesh = o3d.io.read_triangle_mesh(gt_w_ply)
    # gt_w_resave_ply = gt_w_ply[:-4]+'_resave.ply'
    # o3d.io.write_triangle_mesh(gt_w_resave_ply, mesh)

    # compute synmetric distance
    chamfer_dist_d = compute_recon_error(pred_d_ply, gt_d_ply, n_samples=10000, vis=False)
    chamfer_dist_s = compute_recon_error(pred_s_ply, gt_s_ply, n_samples=10000, vis=False)
    chamfer_dist_w = compute_recon_error(pred_w_ply, gt_w_ply, n_samples=10000, vis=False)

    return chamfer_dist_s, chamfer_dist_d, chamfer_dist_w

# if __name__ == '__main__':
#     # eval_d2neus_part()
#     eval_d2nerf_part()