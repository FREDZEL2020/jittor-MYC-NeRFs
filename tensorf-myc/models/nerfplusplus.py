from .tensoRF import *
import jittor.nn as nn

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision

class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                       log_sampling=True, include_input=True,
                       periodic_fns=( jt.sin,  jt.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. **  jt.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands =  jt.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def execute(self, input):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out =  jt.concat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out

# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


class MLPNet(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_viewdirs=3,
                 skips=[4], use_viewdirs=False):
        '''
        :param D: network depth
        :param W: network width
        :param input_ch: input channels for encodings of (x, y, z)
        :param input_ch_viewdirs: input channels for encodings of view directions
        :param skips: skip connection in network
        :param use_viewdirs: if True, will use the view directions as input
        '''
        super().__init__()

        self.input_ch = input_ch
        self.input_ch_viewdirs = input_ch_viewdirs
        self.use_viewdirs = use_viewdirs
        self.skips = skips

        self.base_layers = []
        dim = self.input_ch
        for i in range(D):
            self.base_layers.append(
                nn.Sequential(nn.Linear(dim, W), nn.ReLU())
            )
            dim = W
            if i in self.skips and i != (D-1):      # skip connection after i^th layer
                dim += input_ch
        self.base_layers = nn.ModuleList(self.base_layers)
        # self.base_layers.apply(weights_init)        # xavier init

        sigma_layers = [nn.Linear(dim, 1), ]       # sigma must be positive
        self.sigma_layers = nn.Sequential(*sigma_layers)
        # self.sigma_layers.apply(weights_init)      # xavier init

        # rgb color
        rgb_layers = []
        base_remap_layers = [nn.Linear(dim, 256), ]
        self.base_remap_layers = nn.Sequential(*base_remap_layers)
        # self.base_remap_layers.apply(weights_init)

        dim = 256 + self.input_ch_viewdirs
        for i in range(1):
            rgb_layers.append(nn.Linear(dim, W // 2))
            rgb_layers.append(nn.ReLU())
            dim = W // 2
        rgb_layers.append(nn.Linear(dim, 3))
        rgb_layers.append(nn.Sigmoid())     # rgb values are normalized to [0, 1]
        self.rgb_layers = nn.Sequential(*rgb_layers)
        # self.rgb_layers.apply(weights_init)

    def execute(self, input):
        '''
        :param input: [..., input_ch+input_ch_viewdirs]
        :return [..., 4]
        '''
        input_pts = input[..., :self.input_ch]

        base = self.base_layers[0](input_pts)
        for i in range(len(self.base_layers)-1):
            if i in self.skips:
                base =  jt.concat((input_pts, base), dim=-1)
            base = self.base_layers[i+1](base)

        sigma = self.sigma_layers(base)
        sigma =  jt.abs(sigma)

        base_remap = self.base_remap_layers(base)
        input_viewdirs = input[..., -self.input_ch_viewdirs:]
        rgb = self.rgb_layers( jt.concat((base_remap, input_viewdirs), dim=-1))

        ret = {
                'rgb': rgb,
                'sigma': sigma.squeeze(-1)
            }
        return ret


class NerfPlusPlus(TensorVMSplit):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(NerfPlusPlus, self).__init__(aabb, gridSize, device, **kargs)

    def set_nerfplusplus(self,bg_freq=4, bg_view_freq=2, bg_D=4, radii=20):
        # 单独给背景的mlp
        # background; bg_pt is (x, y, z, 1/r)
        self.bg_freq = bg_freq
        self.bg_view_freq = bg_view_freq
        self.radii = radii
        self.bg_D = bg_D
        self.bg_embedder_position = Embedder(input_dim=4,
                                             max_freq_log2= self.bg_freq - 1,
                                             N_freqs= self.bg_freq)
        self.bg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2= self.bg_view_freq - 1,
                                            N_freqs= self.bg_view_freq)
        self.bg_net = MLPNet(D=bg_D, W= 128, skips=[int(bg_D/2)],
                             input_ch=self.bg_embedder_position.out_dim,
                             input_ch_viewdirs=self.bg_embedder_viewdir.out_dim,
                             use_viewdirs=True)

    def get_kwargs(self):
        kwargs = super().get_kwargs()
        kwargs['bg_freq'] = self.bg_freq
        kwargs['bg_view_freq'] = self.bg_view_freq
        kwargs['bg_D'] = self.bg_D
        kwargs['radii'] = self.radii
        return kwargs

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = super().get_optparam_groups(lr_init_spatialxyz,lr_init_network)
        grad_vars += [{'params':self.bg_net.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def intersect_sphere(self, ray_o, ray_d, radii):
        '''
        ray_o, ray_d: [..., 3]
        compute the depth of the intersection point between this ray and unit sphere
        '''
        # note: d1 becomes negative if this mid point is behind camera
        d1 = -jt.sum(ray_d * ray_o, dim=-1) / jt.sum(ray_d * ray_d, dim=-1)
        p = ray_o + d1.unsqueeze(-1) * ray_d
        # consider the case where the ray does not intersect the sphere
        ray_d_cos = 1. / jt.norm(ray_d, dim=-1)
        p_norm_sq = jt.sum(p * p, dim=-1)
        if (p_norm_sq >= radii).any():
            print(p_norm_sq.max())
            raise Exception('Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly!')
        d2 = jt.sqrt(radii - p_norm_sq) * ray_d_cos

        return d1 + d2

    def perturb_samples(self, z_vals):
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = jt.concat([mids, z_vals[..., -1:]], dim=-1)
        lower = jt.concat([z_vals[..., 0:1], mids], dim=-1)
        # uniform samples in those intervals
        t_rand = jt.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

        return z_vals

    def depth2pts_outside(self, ray_o, ray_d, depth, radii):
        '''
        ray_o, ray_d: [..., 3]
        depth: [...]; inverse of distance to sphere origin
        '''
        # note: d1 becomes negative if this mid point is behind camera
        # 解方程：求出割线中点距射线断点的距离为d1
        d1 = -jt.sum(ray_d * ray_o, dim=-1) / jt.sum(ray_d * ray_d, dim=-1)
        p_mid = ray_o + d1.unsqueeze(-1) * ray_d
        p_mid_norm = jt.norm(p_mid, dim=-1)
        ray_d_cos = 1. / jt.norm(ray_d, dim=-1)
        d2 = jt.sqrt(radii*radii - p_mid_norm * p_mid_norm) * ray_d_cos
        p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

        rot_axis = jt.cross(ray_o, p_sphere, dim=-1)
        rot_axis = rot_axis / jt.norm(rot_axis, dim=-1, keepdim=True)
        phi = jt.asin(p_mid_norm/radii)
        theta = jt.asin(p_mid_norm * depth / (radii * radii))  # depth is inside [0, radii]
        rot_angle = (phi - theta).unsqueeze(-1)     # [..., 1]

        # now rotate p_sphere
        # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        p_sphere_new = p_sphere * jt.cos(rot_angle) + \
                    jt.cross(rot_axis, p_sphere, dim=-1) * jt.sin(rot_angle) + \
                    rot_axis * jt.sum(rot_axis*p_sphere, dim=-1, keepdims=True) * (1.-jt.cos(rot_angle))
        # p_sphere_new = p_sphere_new / jt.norm(p_sphere_new, dim=-1, keepdim=True)
        pts = jt.concat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

        # now calculate conventional depth
        depth_real = radii / (depth + TINY_NUMBER) * jt.cos(theta) * ray_d_cos + d1
        return pts, depth_real

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        fg_far_depth = self.intersect_sphere(rays_o, rays_d, radii=self.radii*self.radii)  # [...,]
        
        
        stepsize = self.stepSize
        near, far = self.near_far

        # foreground depth
        fg_near_depth = near  # TODO: min_depth [..., ]
        step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
        fg_depth = jt.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)  # [..., N_samples]
        fg_depth = self.perturb_samples(fg_depth)
        # # 在包围盒的范围内采样
        # vec = jt.where(rays_d==0, jt.full_like(rays_d, 1e-6), rays_d)
        # rate_a = (self.aabb[1] - rays_o) / vec
        # rate_b = (self.aabb[0] - rays_o) / vec
        # t_min = jt.minimum(rate_a, rate_b).max(-1).clamp(min_v=near, max_v=far)

        # rng = jt.arange(N_samples)[None].float()
        # if is_train:
        #     rng = rng.repeat(rays_d.shape[-2],1)
        #     rng += jt.rand_like(rng[:,[0]])
        # step = stepsize * rng
        # interpx = (t_min[...,None] + step)
        interpx = fg_depth

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, mask_outbbox.logical_not()
        
    
    def execute(self, rays_chunk, white_bg=False, is_train=False, ndc_ray=False, N_samples=-1, additional_output=True):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        # sample points
        
        rgb_map, depth_map, rgb, sigma, alpha, weight, bg_weight = super().execute(rays_chunk, False, is_train, ndc_ray, N_samples,additional_output)
        T = jt.cumprod(1. - alpha + TINY_NUMBER, dim=-1)   # [..., N_samples]
        bg_lambda = T[..., -1]

        ray_o = rays_chunk[:, :3]
        ray_d = rays_chunk[:, 3:6]
        ray_d_norm = jt.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]
        viewdirs = ray_d / ray_d_norm      # [..., 3]
        dots_sh = list(ray_d.shape[:-1])
        N_samples = 512
        bg_z_vals = jt.linspace(0., self.radii, N_samples).view(
            [1, ] * len(dots_sh) + [N_samples,]).expand(dots_sh + [N_samples,])
        bg_z_vals = self.perturb_samples(bg_z_vals)
        bg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_pts, _ = self.depth2pts_outside(bg_ray_o, bg_ray_d, bg_z_vals, radii=self.radii)  # [..., N_samples, 4]
        
        input = jt.concat((self.bg_embedder_position(bg_pts),
                           self.bg_embedder_viewdir(bg_viewdirs)), dim=-1)
        # near_depth: physical far; far_depth: physical near
        input = jt.flip(input, dim=-2)

        bg_z_vals = jt.flip(bg_z_vals, dim=-1)           # 1--->0
        bg_dists = bg_z_vals[..., :-1] - bg_z_vals[..., 1:]
        bg_dists = jt.concat((bg_dists, HUGE_NUMBER * jt.ones_like(bg_dists[..., 0:1])), dim=-1)  # [..., N_samples]
        bg_raw = self.bg_net(input)
        bg_alpha = 1. - jt.exp(-bg_raw['sigma'] * bg_dists)  # [..., N_samples]
        # Eq. (3): T
        # maths show weights, and summation of weights along a ray, are always inside [0, 1]
        T = jt.cumprod(1. - bg_alpha + TINY_NUMBER, dim=-1)[..., :-1]  # [..., N_samples-1]
        T = jt.concat((jt.ones_like(T[..., 0:1]), T), dim=-1)  # [..., N_samples]
        bg_weights = bg_alpha * T  # [..., N_samples]
        bg_rgb_map = jt.sum(bg_weights.unsqueeze(-1) * bg_raw['rgb'], dim=-2)  # [..., 3]
        bg_depth_map = jt.sum(bg_weights * bg_z_vals, dim=-1)  # [...,]

        # composite foreground and background
        bg_lambda = jt.where(bg_lambda > 0.1,bg_lambda,jt.zeros_like(bg_lambda))
        bg_rgb_map = bg_lambda.unsqueeze(-1) * bg_rgb_map
        bg_depth_map = bg_lambda * bg_depth_map
        rgb_map = rgb_map + bg_rgb_map
        # rgb_map = rgb_map.clamp(0,1)
        return rgb_map, depth_map
        # return bg_rgb_map, bg_depth_map

