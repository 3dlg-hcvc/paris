import os
import bpy
import json
import argparse
import numpy as np

def add_lighting():
    # add a new light
    bpy.ops.object.light_add(type="POINT", location=(3.4154, 4.6753, 6.5981))
    bpy.ops.object.light_add(type="POINT", location=(0.80797, -7.77557, 4.78247))
    bpy.ops.object.light_add(type="POINT", location=(-4.96121, 1.9155, 9.01307))

def remove_cube():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()

def camera_setting(exp_dir, n_frames=100, stage='train', track_to='start'):
    '''Set the render setting for the camera and the scene'''
    # bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'

    remove_cube()

    add_lighting()

    cam_obj = bpy.data.objects['Camera']
    constraint = cam_obj.constraints.new('TRACK_TO')
    constraint.target = bpy.data.objects[track_to]
    bpy.context.scene.frame_end = n_frames - 1
    bpy.context.scene.frame_start = 0
    bpy.context.scene.render.resolution_x = 800
    bpy.context.scene.render.resolution_y = 800
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.filepath = f'{exp_dir}/{stage}/'
    bpy.context.scene.render.film_transparent = True


    radius = 3.2 # distance from the origin
    if stage == 'test':  # turntable
        phi = [np.pi / 3. for i in range(n_frames)]  # pitch: larger constant. higher view
        theta = [i * (2 * np.pi) / n_frames for i in range(n_frames)]
    else: # randomly sampled on the upper hemisphere
        phi = np.random.random_sample(n_frames) * 0.5 * np.pi
        theta = np.random.random_sample(n_frames) * 2. * np.pi

    for f in range(n_frames):
        x = radius * np.cos(theta[f]) * np.sin(phi[f])
        y = radius * np.sin(theta[f]) * np.sin(phi[f])
        z = radius * np.cos(phi[f])
        cam_obj.location = np.array([x, y, z])
        cam_obj.keyframe_insert(data_path="location", frame=f)
        print(cam_obj.location)
        print(cam_obj.matrix_world)

def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

def get_K(camd):
    '''Calculate intrinsic matrix K from the Blender camera data.'''
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = [[s_u, skew, u_0],
        [0., s_v, v_0],
        [0., 0., 1.]]
    return K

def get_camera_dict():
    scene = bpy.context.scene
    frame_start = scene.frame_start
    frame_end = scene.frame_end
    
    cam_dict = {}
    cameras = []

    for obj in scene.objects:
        if obj.type != 'CAMERA':
            continue
        cameras.append((obj, obj.data))
    
    for obj, obj_data in cameras:
        K = get_K(obj_data)
        cam_dict['K'] = K

    frame_range = range(frame_start, frame_end + 1)
    for f in frame_range:
        scene.frame_set(f)
        frame_name = '{:0>4d}'.format(f)
        for obj, obj_data in cameras:
            mat = obj.matrix_world
            mat_list =[
                [mat[0][0], mat[0][1], mat[0][2], mat[0][3]],
                [mat[1][0], mat[1][1], mat[1][2], mat[1][3]],
                [mat[2][0], mat[2][1], mat[2][2], mat[2][3]],
                [mat[3][0], mat[3][1], mat[3][2], mat[3][3]],
            ]
            cam_dict[frame_name] = mat_list
    
    return cam_dict


def camera_export(stage='train', dst_dir=''):
    file_path = f'{dst_dir}/camera_{stage}.json'
    with open(file_path, 'w') as fh:
        cam = get_camera_dict()
        json.dump(cam, fh)
        fh.close()

if __name__ == "__main__":
    '''
    Script to render images and export camera parameters using Blender (tested on version 4.1.0)
    
    Structure of the output folder:
    <dst_dir>
        ├── <stage>
        │   ├── 0000.png
        │   ├── 0001.png
        │   ├── ...
        │── camera_<stage>.json
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True, help='folder containing .obj files')
    parser.add_argument('--dst_dir', type=str, required=True, help='folder to save rendered images and camera parameters')
    parser.add_argument('--n_frames', type=int, default=100, help='number of frames to render')
    parser.add_argument('--stage', type=str, default='train', choices=['train', 'val', 'test'], help='render images for which set')
    args = parser.parse_args()

    assert os.path.exists(args.src_dir), f'{args.src_dir} does not exist'
    os.makedirs(args.dst_dir, exist_ok=True)

    # STEP 0: Load .obj files into the blender scene
    bpy.ops.wm.obj_import(filepath=args.src_dir, import_vertex_groups=True)

    # STEP 1: Config the camera setting
    fname = os.path.basename(args.src_dir)
    track_to = fname[:-4]  # the object name that camera should track to
    camera_setting(exp_dir=args.dst_dir, n_frames=args.n_frames, stage=args.stage, track_to=track_to)

    # STEP 2: Export the camera parameters
    camera_export(args.stage, args.dst_dir)

    # STEP 3: Render Images
    bpy.ops.render.render(animation=True)
    