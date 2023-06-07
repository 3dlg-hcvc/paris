import pymeshlab
import numpy as np
import json
import os
import sys
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
        # to make sure this is not a R slider type
        assert 'rotates' not in entry['jointData']['limit'].keys()
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

def generate_state(motions, src_root, exp_dir, state):
    joint_id = motions['joint_id']
    motion_type = motions['motion']['type']
    # 1. load parts needs transformation to the mesh set
    ms = pymeshlab.MeshSet()
    for entry in meta:
        # add all moving parts into the meshset
        if entry['id'] == joint_id or entry['parent'] == joint_id:
            for mesh_fname in entry['visuals']:
                ms.load_new_mesh(os.path.join(src_root, mesh_fname))

    # 2. apply transformation
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
    else:
        raise NotImplementedError

    # 3. load static parts to the mesh set
    for entry in meta:
        if entry['id'] != joint_id and entry['parent'] != joint_id:
            for mesh_fname in entry['visuals']:
                ms.load_new_mesh(os.path.join(src_root, mesh_fname))

    # 4. Merge Filter: Flatten Visible Layers
    ms.generate_by_merging_visible_meshes(mergevisible=True,
                                          deletelayer=False,
                                          mergevertices=True,
                                          alsounreferenced=True)

    # save original obj: y is up
    ms.save_current_mesh(os.path.join(exp_dir, f'{state}.obj'))

    # Filter: Transform: Rotate, so that the object is at z-up frame
    ms.compute_matrix_from_rotation(rotaxis='X axis',
                                    rotcenter='origin',
                                    angle=90,
                                    snapflag=False,
                                    freeze=True,
                                    alllayers=True)

    # save rotated obj: z is up (align with the blender rendering)
    ms.save_current_mesh(os.path.join(exp_dir, f'{state}_rotate.obj'))
    ms.save_current_mesh(os.path.join(exp_dir, f'{state}_rotate.ply'), save_face_color=False)

    return arti_info

def record_motion_json(motions, arti_info, dst_root):
    with open(os.path.join(dst_root, f'trans.json'), 'w') as f:
        conf = {
            'input': motions,
            'trans_info': arti_info
        }
        json.dump(conf, f)
        f.close()

if __name__ == '__main__':
    '''
    This script is to generate object mesh for each state.
    The articulation is referred to PartNet-Mobility <mobility_v2_self.json> which is created from step 0
    '''
    # specify the object category
    category = 'safe'
    # specify the model id to be loaded
    model_id = '100189'     
    # specify the export identifier
    model_id_exp = '100189'
    # specify the motion to generate new states
    motions = {
        'joint_id': 0, # joint id to be transformed
        'motion': {
            # type of motion expected
            'type': 'rotate',   
            # range of the motion from start to end states
            'rotate': [0., -40.0], 
            'translate': [0., 0.],
        },
    }
    # states to be generated
    states = ['start', 'end', 'canonical']


    src_root = os.path.join(ROOT_DIR, 'data/PartNet-Mobility', model_id)
    dst_root =  os.path.join(ROOT_DIR, f'data/sapien/{category}', model_id_exp, 'textured_objs')

    arti_info, meta = load_articulation(src_root, motions['joint_id'])

    for state in states:
        exp_dir = os.path.join(dst_root, state)
        os.makedirs(exp_dir, exist_ok=True)
        # arti_info = generate_state(model_id_inp, motions, exp_dir, state)
        generate_state(motions, src_root, exp_dir, state)
        print(f'{state} done')

    # 6. Backup transformation json
    record_motion_json(motions, arti_info, dst_root)

