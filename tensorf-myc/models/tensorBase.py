import jittor as jt
import jittor.nn
import jittor.nn as F
from .sh import eval_sh_bases
import numpy as np
import time


def positional_encoding(positions, freqs):
    
        freq_bands = (2**jt.arange(freqs).float())  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = jt.concat([jt.sin(pts), jt.cos(pts)], dim=-1)
        return pts

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - jt.exp(-sigma*dist)

    T = jt.cumprod(jt.concat([jt.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = jt.relu(jt.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):

    rgb = features
    return rgb

class AlphaGridMask(jt.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=jt.Var(aabb)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = jt.int32([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]])

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        if xyz_sampled.shape[0] > 0:
            alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)
        else:
            alpha_vals = jt.array([])
        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1


class MLPRender_Fea(jt.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = jt.nn.Linear(self.in_mlpC, featureC)
        layer2 = jt.nn.Linear(featureC, featureC)
        layer3 = jt.nn.Linear(featureC,3)

        self.mlp = jt.nn.Sequential(layer1, jt.nn.ReLU(), layer2, jt.nn.ReLU(), layer3)
        jt.nn.init.constant_(self.mlp[-1].bias, 0)

    def execute(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = jt.concat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = jt.sigmoid(rgb)

        return rgb

class MLPRender_PE(jt.nn.Module):
    def __init__(self,inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3+2*viewpe*3)+ (3+2*pospe*3)  + inChanel #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = jt.nn.Linear(self.in_mlpC, featureC)
        layer2 = jt.nn.Linear(featureC, featureC)
        layer3 = jt.nn.Linear(featureC,3)

        self.mlp = jt.nn.Sequential(layer1, jt.nn.ReLU( ), layer2, jt.nn.ReLU( ), layer3)
        jt.nn.init.constant_(self.mlp[-1].bias, 0)

    def execute(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = jt.concat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = jt.sigmoid(rgb)

        return rgb

class MLPRender(jt.nn.Module):
    def __init__(self,inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3+2*viewpe*3) + inChanel
        self.viewpe = viewpe
        
        layer1 = jt.nn.Linear(self.in_mlpC, featureC)
        layer2 = jt.nn.Linear(featureC, featureC)
        layer3 = jt.nn.Linear(featureC,3)

        self.mlp = jt.nn.Sequential(layer1, jt.nn.ReLU( ), layer2, jt.nn.ReLU( ), layer3)
        jt.nn.init.constant_(self.mlp[-1].bias, 0)

    def execute(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = jt.concat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = jt.sigmoid(rgb)

        return rgb



class TensorBase(jt.nn.Module):
    def __init__(self, aabb, gridSize, device, density_n_comp = 8, appearance_n_comp = 24, app_dim = 27,
                    shadingMode = 'MLP_PE', alphaMask = None, near_far=[2.0,20.0],
                    density_shift = -10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                    pos_pe = 6, view_pe = 6, fea_pe = 6, featureC=128, step_ratio=2.0,
                    fea2denseAct = 'softplus'):
        super(TensorBase, self).__init__()

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        # aabb包围盒 （2 3）
        self.aabb = jt.Var(aabb)
        self.alphaMask = alphaMask
        self.device=device

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio


        self.update_stepSize(gridSize)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]


        self.init_svd_volume(gridSize[0], device)

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == 'MLP_PE':
            self.renderModule = MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC)
        elif shadingMode == 'MLP_Fea':
            self.renderModule = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC)
        elif shadingMode == 'MLP':
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC)
        elif shadingMode == 'SH':
            self.renderModule = SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            self.renderModule = RGBRender
        else:
            print("Unrecognized shading module")
            exit()
        self.renderModule.requires_grad_(requires_grad=True)
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        gridSize = [int(i) for i in gridSize]
        self.gridSize= jt.Var(gridSize).int32()
        self.units=self.aabbSize / (self.gridSize-1)
        self.stepSize=jt.mean(self.units)*self.step_ratio
        self.aabbDiag = jt.sqrt(jt.sum(jt.pow(self.aabbSize, 2)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass
    
    def compute_densityfeature(self, xyz_sampled):
        pass
    
    def compute_appfeature(self, xyz_sampled):
        pass
    
    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC
        }

    def save(self, path, global_kwargs=None):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        # 添加保存optimizer的参数
        if global_kwargs is not None:
            ckpt.update(global_kwargs)
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb })
        jt.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = jt.Var(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'], alpha_volume.float() )
        self.load_state_dict(ckpt['state_dict'])
        self.load_parameterlist(ckpt['state_dict'])

    def load_parameterlist(self, params):
        ''' loads parameters to the Module.

        :param params: dictionary of parameter names and parameters.
        '''
        n_failed = 0
        for key in params.keys():
            v = self
            key_ = key.split('.')
            end = 0
            ok = 0
            for k in key_:
                if isinstance(v, jt.nn.ParameterList):
                    ok = 1
                    v = v.params
                    if k.isdigit() and (int(k) < len(v)):
                        v = v[int(k)]
                    else:
                        end=1
                        break
                else:
                    if hasattr(v, k):
                        v = getattr(v, k)
                        assert isinstance(v, (jt.Module, jt.Var)), \
                            f"expect a jittor Module or Var, but got <{v.__class__.__name__}>, key: {key}"
                    else:
                        end=1
                        break
            if ok == 0:
                continue
            if end == 1:
                if not key.endswith("num_batches_tracked"):
                    n_failed += 1
                    print(f'load parameter {key} failed ...')
            else:
                assert isinstance(v, jt.Var), \
                    f"expect a jittor Var, but got <{v.__class__.__name__}>, key: {key}"
                if isinstance(params[key], np.ndarray) or isinstance(params[key], list):
                    param = jt.array(params[key])
                elif isinstance(params[key], jt.Var):
                    param = params[key]
                else:
                    # assume is pytorch tensor
                    param = list(params[key].cpu().detach().numpy())
                if param.shape == v.shape:
                    print(f'load parameter {key} success ...')
                    v.update(param)
                else:
                    n_failed += 1
                    print(f'load parameter {key} failed: expect the shape of {key} to be {v.shape}, but got {param.shape}')
        jt.sync_all()
        if n_failed:
            print(f"load total {len(params)} params, {n_failed} failed")

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        # .to(rays_o)
        interpx = jt.linspace(near, far, N_samples).unsqueeze(0).float32()
        if is_train:
            interpx += jt.rand_like(interpx) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, mask_outbbox.logical_not()

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples>0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        # 在包围盒的范围内采样
        vec = jt.where(rays_d==0, jt.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = jt.minimum(rate_a, rate_b).max(-1).clamp(min_v=near, max_v=far)

        rng = jt.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += jt.rand_like(rng[:,[0]])
        step = stepsize * rng
        interpx = (t_min[...,None] + step)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, mask_outbbox.logical_not()


    def shrink(self, new_aabb, voxel_size):
        pass

    @jt.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize
        if type(gridSize) is jt.Var:
            gridSize = gridSize.numpy().tolist()
        samples = jt.stack(jt.meshgrid(
            jt.linspace(0, 1, gridSize[0]),
            jt.linspace(0, 1, gridSize[1]),
            jt.linspace(0, 1, gridSize[2]),
        ), -1)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = jt.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @jt.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):

        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0,2)
        alpha = alpha.clamp(0,1).transpose(0,2)[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.min(0)
        xyz_max = valid_xyz.max(0)

        new_aabb = jt.stack((xyz_min, xyz_max))

        total = jt.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        return new_aabb

    @jt.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()

        N = jt.Var(list(all_rays.shape[:-1])).prod()

        mask_filtered = []
        idx_chunks = jt.split(jt.arange(int(N)), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk]

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = jt.where(rays_d == 0, jt.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = jt.minimum(rate_a, rate_b).max(-1)#.clamp(min=near, max=far)
                t_max = jt.maximum(rate_a, rate_b).min(-1)#.clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox)

        mask_filtered = jt.concat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {jt.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered]


    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)


    def compute_alpha(self, xyz_locs, length=1):
        """
        计算遮罩（强度）
        """
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = jt.ones_like(xyz_locs[:,0]).bool()
            

        sigma = jt.zeros(xyz_locs.shape[:-1])

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma
        

        alpha = 1 - jt.exp(-sigma*length).view(xyz_locs.shape[:-1])

        return alpha


    def execute(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1, additional_output=False):

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = jt.concat((z_vals[:, 1:] - z_vals[:, :-1], jt.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = jt.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = jt.concat((z_vals[:, 1:] - z_vals[:, :-1], jt.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ray_valid.logical_not()
            ray_invalid[ray_valid] |= (alpha_mask.logical_not())
            ray_valid = ray_invalid.logical_not()


        sigma = jt.zeros(xyz_sampled.shape[:-1])
        rgb = jt.zeros((*xyz_sampled.shape[:2], 3))

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            # 进行插值，计算位置的强度信息（代替了传统nerf中前面多层的MLP）
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma


        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])
            valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
            rgb[app_mask] = valid_rgbs

        acc_map = jt.sum(weight, -1)
        rgb_map = jt.sum(weight[..., None] * rgb, -2)

        if white_bg: # or (is_train and jt.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        
        rgb_map = rgb_map.clamp(0,1)

        with jt.no_grad():
            depth_map = jt.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        if additional_output:
            return rgb_map, depth_map, rgb, sigma, alpha, weight, bg_weight
        else:
            return rgb_map, depth_map # rgb, sigma, alpha, weight, bg_weight

    def set_nerfplusplus(self,bg_freq=4, bg_view_freq=2, bg_D=4, radii=20):
        pass