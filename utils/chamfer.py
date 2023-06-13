
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
import torch
import open3d as o3d


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


def eval_CD(pred_s_ply, pred_d_ply, pred_w_ply, gt_s_ply, gt_d_ply, gt_w_ply):
    # combine the part meshes as a whole
    combine_pred_mesh([pred_s_ply, pred_d_ply], pred_w_ply)

    # compute synmetric distance
    chamfer_dist_d = compute_recon_error(pred_d_ply, gt_d_ply, n_samples=10000, vis=False)
    chamfer_dist_s = compute_recon_error(pred_s_ply, gt_s_ply, n_samples=10000, vis=False)
    chamfer_dist_w = compute_recon_error(pred_w_ply, gt_w_ply, n_samples=10000, vis=False)

    return chamfer_dist_s, chamfer_dist_d, chamfer_dist_w

