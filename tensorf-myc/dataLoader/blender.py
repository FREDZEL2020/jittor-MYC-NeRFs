import jittor,cv2
import jittor as jt
from jittor.dataset import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image


from .ray_utils import *


class BlenderDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1, bbox=None, near=None,far=None,white_bg=True):

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.img_wh = (int(800/downsample),int(800/downsample))
        self.define_transforms()

        # Coffee
        # self.scene_bbox = jt.Var([[-5.5, -5.5, -5.5], [5.5, 5.5, 8.5]])
        # Easyship [-2.4324, -0.4333, -2.0333,  2.5000,  1.9667,  5.4333]
        # self.scene_bbox = jt.Var([[-2.5, -2.5, -2.5], [5.5, 2.5, 5.5]])
        # Car [-2.7500, -5.7703, -3.8784, 20.5000,  5.0135,  9.9324]
        # self.scene_bbox = jt.Var([[-3.5, -7.5, -5.5], [20.5, 7.5, 10.5]])
        # Scarf
        # self.scene_bbox = jt.Var([[-30.5, -30.5, -30.5], [30.5, 30.5, 30.5]])
        # Scar
        self.scene_bbox = jt.Var([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        # self.define_proj_mat()

        self.white_bg = white_bg
        # Easyship
        # self.near_far = [3.0,10.0]
        # Coffee
        # self.near_far = [0.5,8.0]
        # Car
        # self.near_far = [5.0,25.0]
        # Scarf
        # self.near_far = [5.0,45.0]
        # Scar
        self.near_far = [5.0,40.0]

        if near is not None and far is not None:
            self.near_far = [near,far]
        if bbox is not None:
            self.scene_bbox = jt.array(bbox).view(self.scene_bbox.shape)
        print(self.near_far,self.scene_bbox)
        
        self.center = jt.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample=downsample

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self):

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh


        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)
        self.directions = self.directions / jt.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = jt.Var([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        self.downsample=1.0

        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#

            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = jt.float32(pose)
            self.poses += [c2w]
            if self.split != 'test':
                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                
                if self.downsample!=1.0:
                    img = img.resize(self.img_wh, Image.LANCZOS)  
                # img = self.transform(img)  # (4, h, w)
                img = (np.array(img) / 255.).astype(np.float32)
                img = jt.Var(img)
                img = img.permute(2,0,1) # (4, h, w)
                img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
                self.all_rgbs += [img]
            else:
                img = jt.zeros((h,w,4))
                img = img.permute(2,0,1) # (4, h, w)
                img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
                self.all_rgbs += [img]


            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [jt.concat([rays_o, rays_d], 1)]  # (h*w, 6)


        self.poses = jt.stack(self.poses)
        if not self.is_stack:
            self.all_rays = jt.concat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = jt.concat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)

#             self.all_depth = jt.concat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = jt.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = jt.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = jt.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)


    def define_transforms(self):
        # jt.permute(2,0,1) /255
        self.transform = None
        
    # def define_proj_mat(self):
    #     self.proj_mat = self.intrinsics.unsqueeze(0) @ jt.linalg.inv(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center) / self.radius
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'mask': mask}
        return sample
