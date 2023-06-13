import os
import re
import shutil
import numpy as np
import cv2
import imageio
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import json
import torch
import open3d as o3d
import gc

from utils.rotation import R_from_axis_angle
import torch.nn.functional as F
import copy


class SaverMixin():
    @property
    def save_dir(self):
        return self.config.save_dir
    
    def convert_data(self, data):
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, list):
            return [self.convert_data(d) for d in data]
        elif isinstance(data, dict):
            return {k: self.convert_data(v) for k, v in data.items()}
        else:
            raise TypeError('Data must be in type numpy.ndarray, torch.Tensor, list or dict, getting', type(data))
    
    def get_save_path(self, filename):
        save_path = os.path.join(self.save_dir, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path
    
    DEFAULT_RGB_KWARGS = {'data_format': 'CHW', 'data_range': (0, 1)}
    DEFAULT_UV_KWARGS = {'data_format': 'CHW', 'data_range': (0, 1), 'cmap': 'checkerboard'}
    DEFAULT_GRAYSCALE_KWARGS = {'data_range': None, 'cmap': 'jet'}

    def get_rgb_image_(self, img, data_format, data_range, draw_axis=False, axis_info=None):
        img = self.convert_data(img)
        assert data_format in ['CHW', 'HWC']
        if data_format == 'CHW':
            img = img.transpose(1, 2, 0)
        img = img.clip(min=data_range[0], max=data_range[1])
        img = ((img - data_range[0]) / (data_range[1] - data_range[0]) * 255.).astype(np.uint8)
        imgs = [img[...,start:start+3] for start in range(0, img.shape[-1], 3)]
        imgs = [img_ if img_.shape[-1] == 3 else np.concatenate([img_, np.zeros((img_.shape[0], img_.shape[1], 3 - img_.shape[2]), dtype=img_.dtype)], axis=-1) for img_ in imgs]
        img = np.concatenate(imgs, axis=1)   
        if draw_axis: # project axis to the image
            assert axis_info is not None
            p_gt = axis_info['GT'].round().astype(np.int16)
            p_pred = axis_info['pred'].round().astype(np.int16)
            img = cv2.arrowedLine(img, p_gt[0], p_gt[1], color=(0, 255, 0), thickness=2)
            img = cv2.arrowedLine(img, p_pred[0], p_pred[1], color=(0, 0, 255), thickness=2)


        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    def save_rgb_image(self, filename, img, data_format=DEFAULT_RGB_KWARGS['data_format'], data_range=DEFAULT_RGB_KWARGS['data_range']):
        img = self.get_rgb_image_(img, data_format, data_range)
        cv2.imwrite(self.get_save_path(filename), img)
    
    def get_uv_image_(self, img, data_format, data_range, cmap):
        img = self.convert_data(img)
        assert data_format in ['CHW', 'HWC']
        if data_format == 'CHW':
            img = img.transpose(1, 2, 0)
        img = img.clip(min=data_range[0], max=data_range[1])
        img = (img - data_range[0]) / (data_range[1] - data_range[0])
        assert cmap in ['checkerboard', 'color']
        if cmap == 'checkerboard':
            n_grid = 64
            mask = (img * n_grid).astype(int)
            mask = (mask[...,0] + mask[...,1]) % 2 == 0
            img = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
            img[mask] = np.array([255, 0, 255], dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif cmap == 'color':
            img_ = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img_[..., 0] = (img[..., 0] * 255).astype(np.uint8)
            img_[..., 1] = (img[..., 1] * 255).astype(np.uint8)
            img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
            img = img_
        return img
    
    def save_uv_image(self, filename, img, data_format=DEFAULT_UV_KWARGS['data_format'], data_range=DEFAULT_UV_KWARGS['data_range'], cmap=DEFAULT_UV_KWARGS['cmap']):
        img = self.get_uv_image_(img, data_format, data_range, cmap)
        cv2.imwrite(self.get_save_path(filename), img)

    def get_grayscale_image_(self, img, data_range, cmap):
        img = self.convert_data(img)
        img = np.nan_to_num(img)
        if data_range is None:
            # print(f'\n---------max: {img.max()}, min: {img.min()}---------\n')
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = img.clip(data_range[0], data_range[1])
            img = (img - data_range[0]) / (data_range[1] - data_range[0])
        assert cmap in [None, 'jet', 'magma']
        if cmap == None:
            img = (img * 255.).astype(np.uint8)
            img = np.repeat(img[..., None], 3, axis=2)
        elif cmap == 'jet':
            img = (img * 255.).astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        elif cmap == 'magma':
            img = 1. - img
            base = cm.get_cmap('magma')
            num_bins = 256
            colormap = LinearSegmentedColormap.from_list(
                f"{base.name}{num_bins}",
                base(np.linspace(0, 1, num_bins)),
                num_bins
            )(np.linspace(0, 1, num_bins))[:,:3]
            a = np.floor(img * 255.)
            b = (a + 1).clip(max=255.)
            f = img * 255. - a
            a = a.astype(np.uint16).clip(0, 255)
            b = b.astype(np.uint16).clip(0, 255)
            img = colormap[a] + (colormap[b] - colormap[a]) * f[...,None]
            img = (img * 255.).astype(np.uint8)
        return img

    def save_grayscale_image(self, filename, img, data_range=DEFAULT_GRAYSCALE_KWARGS['data_range'], cmap=DEFAULT_GRAYSCALE_KWARGS['cmap']):
        img = self.get_grayscale_image_(img, data_range, cmap)
        cv2.imwrite(self.get_save_path(filename), img)

    def get_image_grid_(self, imgs):
        if isinstance(imgs[0], list):
            return np.concatenate([self.get_image_grid_(row) for row in imgs], axis=0)
        # get data range for "depth:" image across the three images
        depth_max = 0.
        for col in imgs:
            if col['type'] == 'grayscale' and not col['kwargs']:
                img = col['img']
                img = self.convert_data(img)
                img = np.nan_to_num(img)
                max_val = img.max()
                if max_val > depth_max:
                    depth_max = max_val
        depth_range = {'data_range': (0, depth_max)}
        # draw images
        cols = []
        for col in imgs:
            if col['type'] == 'rgb':
                rgb_kwargs = self.DEFAULT_RGB_KWARGS.copy()
                rgb_kwargs.update(col['kwargs'])
                cols.append(self.get_rgb_image_(col['img'], **rgb_kwargs))
            elif col['type'] == 'uv':
                uv_kwargs = self.DEFAULT_UV_KWARGS.copy()
                uv_kwargs.update(col['kwargs'])
                cols.append(self.get_uv_image_(col['img'], **uv_kwargs))
            elif col['type'] == 'grayscale' and not col['kwargs']:
                grayscale_kwargs = self.DEFAULT_GRAYSCALE_KWARGS.copy()
                grayscale_kwargs.update(col['kwargs'])
                grayscale_kwargs.update(depth_range)
                cols.append(self.get_grayscale_image_(col['img'], **grayscale_kwargs))
            elif col['type'] == 'grayscale' and col['kwargs']:
                grayscale_kwargs = self.DEFAULT_GRAYSCALE_KWARGS.copy()
                grayscale_kwargs.update(col['kwargs'])
                cols.append(self.get_grayscale_image_(col['img'], **grayscale_kwargs))
        return np.concatenate(cols, axis=1)
    
    def save_image_grid(self, filename, imgs):
        img = self.get_image_grid_(imgs)
        cv2.imwrite(self.get_save_path(filename), img)
    
    def save_image(self, filename, img):
        img = self.convert_data(img)
        assert img.dtype == np.uint8
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(self.get_save_path(filename), img)
    
    def save_cubemap(self, filename, img, data_range=(0, 1)):
        img = self.convert_data(img)
        assert img.ndim == 4 and img.shape[0] == 6 and img.shape[1] == img.shape[2]

        imgs_full = []
        for start in range(0, img.shape[-1], 3):
            img_ = img[...,start:start+3]
            img_ = np.stack([self.get_rgb_image_(img_[i], 'HWC', data_range) for i in range(img_.shape[0])], axis=0)
            size = img_.shape[1]
            placeholder = np.zeros((size, size, 3), dtype=np.float32)
            img_full = np.concatenate([
                np.concatenate([placeholder, img_[2], placeholder, placeholder], axis=1),
                np.concatenate([img_[1], img_[4], img_[0], img_[5]], axis=1),
                np.concatenate([placeholder, img_[3], placeholder, placeholder], axis=1)
            ], axis=0)
            img_full = cv2.cvtColor(img_full, cv2.COLOR_RGB2BGR)
            imgs_full.append(img_full)
        
        imgs_full = np.concatenate(imgs_full, axis=1)
        cv2.imwrite(self.get_save_path(filename), imgs_full)

    def save_data(self, filename, data):
        data = self.convert_data(data)
        if isinstance(data, dict):
            if not filename.endswith('.npz'):
                filename += '.npz'
            np.savez(self.get_save_path(filename), **data)
        else:
            if not filename.endswith('.npy'):
                filename += '.npy'
            np.save(self.get_save_path(filename), data)
        
    def save_state_dict(self, filename, data):
        torch.save(data, self.get_save_path(filename))
    
    def save_img_sequence(self, filename, img_dir, matcher, save_format='gif', fps=30):
        assert save_format in ['gif', 'mp4']
        if not filename.endswith(save_format):
            filename += f".{save_format}"
        matcher = re.compile(matcher)
        img_dir = os.path.join(self.save_dir, img_dir)
        imgs = []
        for f in os.listdir(img_dir):
            if matcher.search(f):
                imgs.append(f)
        imgs = sorted(imgs, key=lambda f: int(matcher.search(f).groups()[0]))
        imgs = [cv2.imread(os.path.join(img_dir, f)) for f in imgs]
        
        if save_format == 'gif':
            imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
            imageio.mimsave(self.get_save_path(filename), imgs, fps=fps, palettesize=256)
        elif save_format == 'mp4':
            H, W, _ = imgs[0].shape
            writer = cv2.VideoWriter(self.get_save_path(filename), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H), True)
            for img in imgs:
                writer.write(img)
            writer.release()
    
    def save_anim_video(self, filename, img_paths, save_format='mp4', fps=10):
        assert save_format in ['gif', 'mp4']
        if not filename.endswith(save_format):
            filename += f".{save_format}"
        imgs = [cv2.imread(f) for f in img_paths]
        if save_format == 'gif':
            imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
            imageio.mimsave(self.get_save_path(filename), imgs, fps=fps, palettesize=256)
        elif save_format == 'mp4':
            H, W, _ = imgs[0].shape
            writer = cv2.VideoWriter(self.get_save_path(filename), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H), True)
            for img in imgs:
                writer.write(img)
            writer.release()


    def save_mesh_ply(self, filename, v_pos, t_pos_idx, v_rgb=None):
        v_pos, t_pos_idx = self.convert_data(v_pos), self.convert_data(t_pos_idx)
        vertices, faces = o3d.utility.Vector3dVector(v_pos), o3d.utility.Vector3iVector(t_pos_idx)
        mesh = o3d.geometry.TriangleMesh(vertices=vertices, triangles=faces)
        if v_rgb is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(v_rgb)
        o3d.io.write_triangle_mesh(self.get_save_path(filename), mesh)

    # def save_misaligned_mesh(self, exp_path, src_path, R_align, t_align, is_obj_tran=True):
    #     if not is_obj_tran:
    #         R_align = np.linalg.inv(R_align)
    #         t_align = -t_align
        
    #     src_mesh = o3d.io.read_triangle_mesh(src_path)
    #     src_mesh.rotate(R_align, center=np.zeros(3))
    #     src_mesh.translate(t_align)
    #     o3d.io.write_triangle_mesh(exp_path, src_mesh)
    
    def save_trans_part_mesh(self, inp_filename, exp_filenames, motion):
        can_mesh = o3d.io.read_triangle_mesh(self.get_save_path(inp_filename))
        start_mesh = copy.deepcopy(can_mesh)
        end_mesh = copy.deepcopy(can_mesh)

        if motion['type'] == 'rotate':
            axis_d = motion['axis_d'].cpu().squeeze(0)
            rot_ang = motion['rot_angle'].cpu().squeeze(0)
            axis_o = motion['axis_o'].cpu().squeeze(0).numpy()
            R_start = R_from_axis_angle(axis_d, -torch.deg2rad(rot_ang)).numpy()
            R_end = R_from_axis_angle(axis_d, torch.deg2rad(rot_ang)).numpy()
            start_mesh.rotate(R_start, center=axis_o)
            end_mesh.rotate(R_end, center=axis_o)
        elif motion['type'] == 'translate':
            axis_d = motion['axis_d'].cpu().squeeze(0)
            dist = motion['dist'].cpu().squeeze(0)
            d = F.normalize(axis_d, p=2, dim=0)
            start_mesh.translate(-dist * d)
            end_mesh.translate(dist * d)
        else:
            raise ValueError('the motion type is not supported')
        
        o3d.io.write_triangle_mesh(self.get_save_path(exp_filenames[0]), start_mesh)
        o3d.io.write_triangle_mesh(self.get_save_path(exp_filenames[1]), end_mesh)

    


    def save_trans_part_mesh_translate(self, inp_filename, exp_filename, transform_json, to_start=True):
        can_mesh = o3d.io.read_triangle_mesh(self.get_save_path(inp_filename))
        if 'axis_d' in transform_json.keys():
            axis_d = transform_json['axis_d'].cpu().squeeze(0)
            dist = transform_json['dist'].cpu().squeeze(0)
        else:
            axis_d = transform_json['t_axis_d'].cpu().squeeze(0)
            dist = transform_json['t_dist'].cpu().squeeze(0)
        # axis_o = transform_json['axis_o'].cpu().squeeze(0).numpy()
        d = F.normalize(axis_d, p=2, dim=0)
        if to_start:
            can_mesh.translate(-dist * d) # rotate canonical mesh to start mesh
        else:
            can_mesh.translate(dist * d) # rotate canonical mesh to end mesh
        o3d.io.write_triangle_mesh(self.get_save_path(exp_filename), can_mesh)

    def save_file(self, filename, src_path):
        shutil.copyfile(src_path, self.get_save_path(filename))
    
    def save_json(self, filename, payload):
        with open(self.get_save_path(filename), 'w') as f:
            json.dump(payload, f)
            f.close()

    def save_axis(self, filename, motion):
        from utils.axis import save_axis_mesh
        axis_o = self.convert_data(motion['axis_o']).squeeze(0)
        axis_d = self.convert_data(motion['axis_d']).squeeze(0)
        center = np.array(axis_o, dtype=np.float32)
        k = np.array(axis_d, dtype=np.float32)

        k = k / np.linalg.norm(k)
        filepath = self.get_save_path(filename)
        save_axis_mesh(k, center, filepath)
        # save the axis in the opposite direction
        filename_opp = filename[:-4] + '_oppo.ply'
        filepath_oppo = self.get_save_path(filename_opp)
        save_axis_mesh(-k, center, filepath_oppo)
    
    
    def save_volume(self, i_epoch, volumes, radius=1.5, res=128):
        '''
        This function can hang up at some point (depend on the size of the input) after being called multiple times

        [Debuggings]
            * After research on the issues in Plotly/Kaleido repo, it is known that write_image() causes the hunging up.
            * Some attempts: 
                1) some people mentioned with <engine='orca'> can avoid this problem, which is not true in our case
                2) downgrading kaleido to 0.1.0 w/ "pio.kaleido.scope.mathjax = None" and "scope._shutdown_kaleido()" won't work either
            * After reading logs, the specific line gets hung up is "response = self._proc.stdout.readline()" in the file 'miniconda3/envs/an1/lib/python3.9/site-packages/kaleido/base.py'
            * An article <http://thraxil.org/users/anders/posts/2008/03/13/Subprocess-Hanging-PIPE-is-your-enemy/> mentioned PIPE in subprocess.Popen() is the core issue
                - PIPE is a buffer in memory with a fixed size (2^16 characters), which will be full if a large amount of data is trying to fit in
                - It stopped the child process writing to the buffer, so the child processes would then sit there and wait forever.
                - A proposed solution is to pass tempfile.TemporaryFile() to Popen(stdin=, stdout=), which I didn't figure how to do it properly.
                - Some hints: <https://stackoverflow.com/questions/38374063/python-can-we-use-tempfile-with-subprocess-to-get-non-buffering-live-output-in-p>
            * Not sure if it has something to do with 
        [Temporary Work Around]
        - Try to make res small to avoid PIPE being full
        - Clean the memory might help? e.g. del volumes, gc.collect()
        '''
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
        # start_time = time.time()
        pio.templates.default = "plotly_white"
        scope = pio.kaleido.scope
        # important to add if use kaleido!
        scope.mathjax = None 
        

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.0, y=-1.2, z=0.8)
        )

        x = torch.linspace(-radius, radius, steps=res)
        y = torch.linspace(-radius, radius, steps=res)
        z = torch.linspace(-radius, radius, steps=res)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')

        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'volume'}, {'type': 'volume'}]],
                subplot_titles=("static", "dynamic"))
        
        fig.add_trace(go.Volume(
            x=grid_x.flatten(),
            y=grid_y.flatten(),
            z=grid_z.flatten(),
            value=volumes['static'].flatten(),
            isomin=0.2,
            isomax=5.0,
            opacity=0.05, # needs to be small to see through all surfaces
            surface_count=100, # needs to be a large number for good volume rendering
            caps=dict(x_show=False, y_show=False, z_show=False), # no caps
            colorbar=dict(title='density'),
            colorscale='Blues',
            ), row=1, col=1) 
        fig.update_layout(scene_camera=camera)

        fig.add_trace(go.Volume(
            x=grid_x.flatten(),
            y=grid_y.flatten(),
            z=grid_z.flatten(),
            value=volumes['dynamic'].flatten(),
            isomin=0.2,
            isomax=5.0,
            opacity=0.05, # needs to be small to see through all surfaces
            surface_count=100, # needs to be a large number for good volume rendering
            caps=dict(x_show=False, y_show=False, z_show=False), # no caps
            colorbar=dict(title='density'),
            colorscale='Blues',
            ), row=1, col=2) 
        fig.update_layout(scene2_camera = camera)

        # fig.write_html(self.get_save_path(f'volumes/epoch{int(i_epoch)}_fields.html'))
        fig.write_image(self.get_save_path(f'volumes/epoch{int(i_epoch)}_fields.png'), width=960, height=640, engine="kaleido")
        scope._shutdown_kaleido()

        del scope, volumes, fig, x, y, z, grid_x, grid_y, grid_z
        gc.collect()

        # to show the difference of the volumes
        # diff_volume = volumes['static'] -  volumes['dynamic']
        # fig_diff = go.Figure(data=go.Volume(
        #     x=grid_x.flatten(),
        #     y=grid_y.flatten(),
        #     z=grid_z.flatten(),
        #     value=diff_volume.flatten(),
        #     isomin=-2.,
        #     isomax=2.,
        #     opacity=0.1, # needs to be small to see through all surfaces
        #     surface_count=40, # needs to be a large number for good volume rendering
        #     caps=dict(x_show=False, y_show=False, z_show=False), # no caps
        #     colorbar=dict(title='S-D'),
        #     # colorscale='deep',
        # ))
        # fig_diff.update_layout(scene_camera=camera)
        # # fig_diff.write_html(self.get_save_path(f'volumes/epoch{int(i_epoch)}_diff.html'))
        # fig_diff.write_image(self.get_save_path(f'volumes/epoch{int(i_epoch)}_diff.png'), width=1920, height=1080, engine="kaleido")
        # print(f'finish volume visualization in {time.time()-start_time} seconds')

        # debug: logging
        # scope = pio.kaleido.scope
        # print(scope._std_error.getvalue().decode())
        # scope._shutdown_kaleido()
        

    def render_geometry(self, gt_json_path, step, res, thre, R_center, R, config_dataset):
        R = R.cpu()
        R_center = R_center.cpu()
        gt_dir = os.path.dirname(gt_json_path)
        gt_s = os.path.join(gt_dir, 'start', 'start_static_rotate.ply')
        gt_d = os.path.join(gt_dir, 'start', 'start_dynamic_rotate.ply')
        gt_a = os.path.join(gt_dir, 'axis_rotate.ply')

        with open(os.path.join(config_dataset.root_dir, 'start', 'camera_test.json'), 'r') as f:
            meta = json.load(f)
            c2w = np.array(meta[config_dataset.view_idx])
            f.close()
        

        camera_R = c2w[:3, :3]
        camera_t = c2w[:3, 3]
        up = np.matmul(camera_R, np.array([0., 1., 0.]).T).T
        eye = camera_t
        center = [0, 0, 0]  # look_at target



        pred_s = self.get_save_path(f"it{int(step)}_static_{res}_thre{thre}.ply")
        pred_d = self.get_save_path(f"it{int(step)}_dynamic_{res}_thre{thre}.ply")
        pred_a = self.get_save_path(f'it{int(step)}_axis.ply')

        pred_s_mesh = o3d.io.read_triangle_mesh(pred_s)    
        pred_d_mesh = o3d.io.read_triangle_mesh(pred_d)
        # pred_d_mesh.rotate(np.array(R, dtype=np.float64), np.array(R_center.T))  
        pred_a_mesh = o3d.io.read_triangle_mesh(pred_a) 
        pred_s_mesh.compute_triangle_normals()
        pred_s_mesh.compute_vertex_normals()
        pred_d_mesh.compute_triangle_normals()
        pred_d_mesh.compute_vertex_normals()
        pred_a_mesh.compute_triangle_normals()
        pred_a_mesh.compute_vertex_normals()

        gt_s_mesh = o3d.io.read_triangle_mesh(gt_s)    
        gt_d_mesh = o3d.io.read_triangle_mesh(gt_d)  
        gt_a_mesh = o3d.io.read_triangle_mesh(gt_a)
        gt_s_mesh.compute_triangle_normals()
        gt_s_mesh.compute_vertex_normals()
        gt_d_mesh.compute_triangle_normals()
        gt_d_mesh.compute_vertex_normals()
        gt_a_mesh.compute_triangle_normals()
        gt_a_mesh.compute_vertex_normals()

        mtl_axis = o3d.visualization.rendering.MaterialRecord()  # or MaterialRecord(), for later versions of Open3D
        mtl_axis.base_color = [0.8, 0.0, 0.0, 1.0]  # RGBA
        mtl_axis.shader = "defaultLit"

        mtl_static = o3d.visualization.rendering.MaterialRecord()  # or MaterialRecord(), for later versions of Open3D
        mtl_static.base_color = [0.58039216, 0.7372549, 0.89411765, 1.0]  # RGBA
        mtl_static.shader = "defaultLit"

        mtl_dynamic = o3d.visualization.rendering.MaterialRecord()  # or MaterialRecord(), for later versions of Open3D
        mtl_dynamic.base_color = [0.6, 0.8, 0.8, 1.0]  # RGBA
        mtl_dynamic.shader = "defaultLit"

        renderer = o3d.visualization.rendering.OffscreenRenderer(width=800, height=800)
        # Pick a background colour (default is light gray)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])  # RGBA
        renderer.scene.add_geometry("pred_a_mesh", pred_a_mesh, mtl_axis)
        renderer.scene.add_geometry("pred_s_mesh", pred_s_mesh, mtl_static)
        renderer.scene.add_geometry("pred_d_mesh", pred_d_mesh, mtl_dynamic)


        # Since the arrow material is unlit, it is not necessary to change the scene lighting.
        renderer.scene.scene.enable_sun_light(True)
        renderer.scene.set_lighting(renderer.scene.LightingProfile.NO_SHADOWS, (1, 1, 1))

        renderer.scene.camera.look_at(center, eye, up)
        vertical_field_of_view = np.rad2deg(2. * np.arctan(800/2222.2222222222)) 
        renderer.scene.camera.set_projection(vertical_field_of_view, 1.0, 0.1, 10., o3d.visualization.rendering.Camera.FovType.Vertical)
        # Read the image into a variable
        img_o3d = renderer.render_to_image()

        # Optionally write it to a PNG file
        o3d.io.write_image(self.get_save_path("output.png"), img_o3d, 9)

        renderer_gt = o3d.visualization.rendering.OffscreenRenderer(width=800, height=800)
        # Pick a background colour (default is light gray)
        renderer_gt.scene.set_background([1.0, 1.0, 1.0, 1.0])  # RGBA
        renderer_gt.scene.add_geometry("gt_a_mesh", gt_a_mesh, mtl_axis)
        renderer_gt.scene.add_geometry("gt_s_mesh", gt_s_mesh, mtl_static)
        renderer_gt.scene.add_geometry("gt_d_mesh", gt_d_mesh, mtl_dynamic)


        # Since the arrow material is unlit, it is not necessary to change the scene lighting.
        renderer.scene.scene.enable_sun_light(True)
        renderer.scene.set_lighting(renderer.scene.LightingProfile.NO_SHADOWS, (1, 1, 1))

        renderer_gt.scene.camera.look_at(center, eye, up)
        renderer_gt.scene.camera.set_projection(vertical_field_of_view, 1.0, 0.1, 10., o3d.visualization.rendering.Camera.FovType.Vertical)
        # Read the image into a variable
        img_gt = renderer_gt.render_to_image()

        # Optionally write it to a PNG file
        o3d.io.write_image(self.get_save_path("gt.png"), img_gt, 9)

