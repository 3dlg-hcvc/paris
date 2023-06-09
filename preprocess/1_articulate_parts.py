import pymeshlab
import numpy as np
import json
import os
import sys
import open3d as o3d
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)


def normalize(v):
    return v / np.sqrt(np.sum(v**2))

def get_rotation_axis_angle(k, theta):
    '''
    Rodrigues' rotation formula
    args:
    * k: direction vector of the axis to rotate about
    * theta: the (radian) angle to rotate with
    return:
    * 3x3 rotation matrix
    '''
    k = normalize(k)
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

def merge_meshsets(mss: list):
    for ms in mss:
        ms.generate_by_merging_visible_meshes(mergevisible=True,
                                              deletelayer=False,
                                              mergevertices=True,
                                              alsounreferenced=True)
    return mss

def z_up_frame_meshsets(mss: list):
    for ms in mss:
        ms.compute_matrix_from_rotation(rotaxis='X axis',
                                        rotcenter='origin',
                                        angle=90,
                                        snapflag=False,
                                        freeze=True,
                                        alllayers=True)
    return mss

def save_meshsets_ply(mss: list, fnames: list):
    for ms, fname in zip(mss, fnames):
        ms.save_current_mesh(fname,
                             save_vertex_color=False,
                             save_face_color=False,
                             save_wedge_texcoord=False)

def get_arti_info(entry, motion):
    res = {
        'axis': {
            'o': entry['jointData']['axis']['origin'],
            'd': entry['jointData']['axis']['direction']
        }
    }

    # hinge joint
    if entry['joint'] == 'hinge':
        assert motion['type'] == 'rotate'
        R_limit_l, R_limit_r = motion['rotate'][0], motion['rotate'][1]
        res.update({
            'rotate': {
                'l': R_limit_l,  # start state
                'r': R_limit_r  # end state
            },
        })
    # slider joint
    elif entry['joint'] == 'slider':
        assert motion['type'] == 'translate'
        T_limit_l, T_limit_r = motion['translate'][0], motion['translate'][1]
        res.update({
                'translate': {
                'l': T_limit_l,
                'r': T_limit_r
            }
        })
    # other joint
    else:
        raise NotImplemented(
            '{} joint is not implemented'.format(entry['joint']))

    return res

def load_articulation(src_root, joint_id):
    with open(os.path.join(src_root, 'mobility_v2_self.json'), 'r') as f:
        meta = json.load(f)
        f.close()

    for entry in meta:
        if entry['id'] == joint_id:
            arti_info = get_arti_info(entry, motions['motion']) 

    return arti_info, meta

def save_axis_mesh(arti, exp_dir):
    center = np.array(arti['axis']['o'], dtype=np.float32)
    k = np.array(arti['axis']['d'], dtype=np.float32)
    axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.02, cone_radius=0.04, cylinder_height=1.0, cone_height=0.08)
    arrow = np.array([0., 0., 1.], dtype=np.float32)
    if (k == arrow).sum() < 3:
        n = np.cross(arrow, k) 
        rad = np.arccos(np.dot(arrow, k))
        R_arrow = get_rotation_axis_angle(n, rad)
        axis.rotate(R_arrow, center=(0, 0, 0))
    axis.translate(center[:3])
    o3d.io.write_triangle_mesh(os.path.join(exp_dir, 'axis_rotate.ply'), axis)

    axis_oppo = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.02, cone_radius=0.04, cylinder_height=1.0, cone_height=0.08)
    if (k == arrow).sum() < 3:
        n_oppo = np.cross(arrow, -k) 
        rad_oppo = np.arccos(np.dot(arrow, -k))
        R_arrow_oppo = get_rotation_axis_angle(n_oppo, rad_oppo)
        axis_oppo.rotate(R_arrow_oppo, center=(0, 0, 0))
    else:
        R_arrow_oppo = get_rotation_axis_angle(np.array([0., 1., 0.], dtype=np.float32), np.pi)
        axis_oppo.rotate(R_arrow_oppo, center=(0, 0, 0))
    axis_oppo.translate(center[:3])
    o3d.io.write_triangle_mesh(os.path.join(exp_dir, 'axis_rotate_oppo.ply'), axis_oppo)

def generate_state(motions, src_root, exp_dir, state):
    joint_id = motions['joint_id']
    motion_type = motions['motion']['type']
    
    ms = pymeshlab.MeshSet()
    ms_static = pymeshlab.MeshSet()
    ms_dynamic = pymeshlab.MeshSet()

    # 1. Load parts needs transformation to the mesh set
    for entry in meta:
        # add all moving parts into the meshset
        if entry['id'] == joint_id or entry['parent'] == joint_id:
            for mesh_fname in entry['visuals']:
                ms.load_new_mesh(os.path.join(src_root, mesh_fname))
                ms_dynamic.load_new_mesh(os.path.join(src_root, mesh_fname))


    # 2. Apply transformation
    if 'rotate' == motion_type:
        if state == 'start':
            degree = arti_info['rotate']['l']
        elif state == 'end':
            degree = arti_info['rotate']['r']
        elif state == 'canonical':
            degree = 0.5 * (arti_info['rotate']['r'] + arti_info['rotate']['l'])
        else:
            raise NotImplementedError
        # Filter: Transform: Rotate
        ms.compute_matrix_from_rotation(rotaxis='custom axis',
                                        rotcenter='custom point',
                                        angle=degree,
                                        customaxis=arti_info['axis']['d'],
                                        customcenter=arti_info['axis']['o'],
                                        snapflag=False,
                                        freeze=True,
                                        alllayers=True)
        ms_dynamic.compute_matrix_from_rotation(rotaxis='custom axis',
                                        rotcenter='custom point',
                                        angle=degree,
                                        customaxis=arti_info['axis']['d'],
                                        customcenter=arti_info['axis']['o'],
                                        snapflag=False,
                                        freeze=True,
                                        alllayers=True)
    elif 'translate' == motion_type:
        if state == 'start':
            dist = arti_info['translate']['l']
        elif state == 'end':
            dist = arti_info['translate']['r']
        elif state == 'canonical':
            dist = 0.5 * (arti_info['translate']['r'] + arti_info['translate']['l'])
        else:
            raise NotImplementedError

        # Filter: Transform: Translate, Center, set Origin
        ms.compute_matrix_from_translation_rotation_scale(
                                        translationx=arti_info['axis']['d'][0]*dist,
                                        translationy=arti_info['axis']['d'][1]*dist,
                                        translationz=arti_info['axis']['d'][2]*dist,
                                        alllayers=True)
        ms_dynamic.compute_matrix_from_translation_rotation_scale(
                                        translationx=arti_info['axis']['d'][0]*dist,
                                        translationy=arti_info['axis']['d'][1]*dist,
                                        translationz=arti_info['axis']['d'][2]*dist,
                                        alllayers=True)
    else:
        raise NotImplementedError

    # 3. load static parts to the mesh set
    for entry in meta:
        if entry['id'] != joint_id and entry['parent'] != joint_id:
            for mesh_fname in entry['visuals']:
                ms.load_new_mesh(os.path.join(src_root, mesh_fname))
                ms_static.load_new_mesh(os.path.join(src_root, mesh_fname))


    # 4. Merge Filter: Flatten Visible Layers
    ms, ms_static, ms_dynamic = merge_meshsets([ms, ms_static, ms_dynamic])


    # 5. Save original obj: y is up
    ms.save_current_mesh(os.path.join(exp_dir, f'{state}.obj'))

    # 6. Transform: Rotate, so that the object is at z-up frame
    mss = z_up_frame_meshsets([ms, ms_static, ms_dynamic])

    # 7. Save rotated meshes: z is up (align with the blender rendering)
    fnames = [
        os.path.join(exp_dir, f'{state}_rotate.ply'),
        os.path.join(exp_dir, f'{state}_static_rotate.ply'),
        os.path.join(exp_dir, f'{state}_dynamic_rotate.ply')
    ]
    save_meshsets_ply(mss, fnames)

    return arti_info

def record_motion_json(motions, arti_info, dst_root):
    # coordinates changes from y-up to z-up
    axis_o = np.array(arti_info['axis']['o'])
    axis_d = np.array(arti_info['axis']['d'])
    R_coord = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    axis_o = np.matmul(R_coord, axis_o).tolist()
    axis_d = np.matmul(R_coord, axis_d).tolist()
    arti_info['axis']['o'] = axis_o
    arti_info['axis']['d'] = axis_d
    arti_info['type'] = motions['motion']['type']

    with open(os.path.join(dst_root, f'trans.json'), 'w') as f:
        conf = {
            'input': motions,
            'trans_info': arti_info
        }
        json.dump(conf, f)
        f.close()

    return arti_info


if __name__ == '__main__':
    '''
    This script is to generate object mesh for each state.
    The articulation is referred to PartNet-Mobility <mobility_v2_self.json> which is created from step 0
    '''
    # specify the object category
    category = 'storage'
    # specify the model id to be loaded
    model_id = '45135'     
    # specify the export identifier
    model_id_exp = '45135'
    # specify the motion to generate new states
    motions = {
        'joint_id': 0, # joint id to be transformed (need to look up mobility_v2_self.json)
        'motion': {
            # type of motion expected: "rotate or translate"
            'type': 'translate',   
            # range of the motion from start to end states
            'rotate': [0., 0.], 
            'translate': [0.3, 0.6],
        },
    }
    # states to be generated
    states = ['start', 'end']
    # paths
    src_root = os.path.join(ROOT_DIR, '../AN3/data/PartNet-Mobility', model_id)
    dst_root =  os.path.join(ROOT_DIR, f'data/sapien/{category}', model_id_exp, 'textured_objs')

    # load articulations (y-up frame)
    arti_info, meta = load_articulation(src_root, motions['joint_id'])
    # save meshes for each states
    for state in states:
        exp_dir = os.path.join(dst_root, state)
        os.makedirs(exp_dir, exist_ok=True)
        generate_state(motions, src_root, exp_dir, state)
        print(f'{state} done')

    # backup transformation json, convert articulation to z-up frame
    arti = record_motion_json(motions, arti_info, dst_root)

    # save mesh for motion axis
    save_axis_mesh(arti, dst_root)
