from .tensoRF import *
import jittor.nn as nn
from .sh import *

class MLPRender_Fea_Ref(jt.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea_Ref, self).__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 1 + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = jt.nn.Linear(self.in_mlpC, featureC)
        layer2 = jt.nn.Linear(featureC, featureC)
        layer3 = jt.nn.Linear(featureC,3)

        self.mlp = jt.nn.Sequential(layer1, jt.nn.ReLU(), layer2, jt.nn.ReLU(), layer3)
        jt.nn.init.constant_(self.mlp[-1].bias, 0)

    def execute(self, pts, viewdirs, features, dot_product, k):
        indata = [dot_product,features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = jt.concat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = jt.sigmoid(rgb)

        return rgb

class MLPRender_SH_Ref(jt.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_SH_Ref, self).__init__()
        self.viewpe = viewpe
        self.feape = feape
        self.in_mlpC = 2*feape*inChanel + 1 + 3 + inChanel
        for l_base in range(1,self.viewpe+1):
            l = l_base ** 2
            self.in_mlpC += l
        layer1 = jt.nn.Linear(self.in_mlpC, featureC)
        layer2 = jt.nn.Linear(featureC, featureC)
        layer3 = jt.nn.Linear(featureC,3)

        self.mlp = jt.nn.Sequential(layer1, jt.nn.ReLU(), layer2, jt.nn.ReLU(), layer3)
        jt.nn.init.constant_(self.mlp[-1].bias, 0)

    def execute(self, pts, viewdirs, features, dot_product, k):
        indata = [dot_product,features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            for l_base in range(1,self.viewpe+1):
                l = l_base ** 2
                a = jt.exp(- (l * (l+1) ) / (2*k) )
                indata += [a * eval_sh_bases(l_base-1, viewdirs)]
        mlp_in = jt.concat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = jt.sigmoid(rgb)

        return rgb


# 必须用MLPRender_Fea_Ref
class REFTensoRF(TensorVMSplit):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(REFTensoRF, self).__init__(aabb, gridSize, device, **kargs)
        self.norm_n_comp = self.density_n_comp
        self.penalty = jt.Var(0)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == 'SH':
            self.renderModule = MLPRender_SH_Ref(self.app_dim, view_pe, fea_pe, featureC)
        else:
            self.renderModule = MLPRender_Fea_Ref(self.app_dim, view_pe, fea_pe, featureC)
        self.renderModule.requires_grad_(requires_grad=True)
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)


    def init_svd_volume(self, res, device):
        super().init_svd_volume(res, device)
        # density_n_comp分解出多少个 gridSize是分辨率
        # # 添加一项张量用来记录每个点的法向量方向
        # self.norm_plane, self.norm_line = self.init_one_svd(self.norm_n_comp, self.gridSize, 0.1, device)
        self.normal_linear = jt.nn.Linear(sum(self.app_n_comp), 3)
        self.normal_linear.requires_grad_(requires_grad=True)
        # 通过appfeature生成散射颜色值
        self.diffuse_linear = jt.nn.Linear(sum(self.app_n_comp), 3)
        self.diffuse_linear.requires_grad_(requires_grad=True)
        # 通过appfeature生成镜面反射值
        self.specular_linear = jt.nn.Linear(sum(self.app_n_comp), 1)
        self.specular_linear.requires_grad_(requires_grad=True)
        # 通过appfeature生成模糊程度
        self.rho_linear = jt.nn.Linear(sum(self.app_n_comp), 1)
        self.rho_linear.requires_grad_(requires_grad=True)
        

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = super().get_optparam_groups(lr_init_spatialxyz,lr_init_network)
        # {'params': self.norm_line, 'lr': lr_init_spatialxyz}, {'params': self.norm_plane, 'lr': lr_init_spatialxyz},
        grad_vars += [{'params': self.normal_linear.parameters(), 'lr':lr_init_network},
                      {'params': self.diffuse_linear.parameters(), 'lr':lr_init_network},
                      {'params': self.rho_linear.parameters(), 'lr':lr_init_network},
                      {'params': self.specular_linear.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_appfeature(self, xyz_sampled):
        """
        return: (appfeatures, rgb_d, specular_tint, normal_vector)
        """

        # plane + line basis
        coordinate_plane = jt.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = jt.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = jt.stack((jt.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = jt.concat(plane_coef_point), jt.concat(line_coef_point)

        h = (plane_coef_point * line_coef_point).transpose()
        appfeatures = self.basis_mat(h)
        normal_vector = self.normal_linear(h) # 法向量
        rgb_d = self.diffuse_linear(h) # color of diffuse 散射颜色
        specular_tint = self.specular_linear(h) # specular tint 镜面反射值
        specular_tint = jt.nn.relu(specular_tint)
        rho = self.rho_linear(h)
        rho = jt.nn.relu(rho)
        return appfeatures, rgb_d, specular_tint, normal_vector, rho
    
    '''
    @jt.no_grad()
    def upsample_volume_grid(self, res_target):
        super().upsample_volume_grid(res_target)
        self.norm_plane, self.norm_line = self.up_sampling_VM(self.norm_plane, self.norm_line, res_target)

    @jt.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = jt.round(jt.round(t_l)).long(), jt.round(b_r).long() + 1
        b_r = jt.stack([b_r, self.gridSize]).min(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = self.density_line[i][...,int(t_l[mode0]):int(b_r[mode0]),:]
            self.app_line[i] = self.app_line[i][...,int(t_l[mode0]):int(b_r[mode0]),:]
            self.norm_line[i] = self.norm_line[i][...,int(t_l[mode0]):int(b_r[mode0]),:]

            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = self.density_plane[i][...,int(t_l[mode1]):int(b_r[mode1]),int(t_l[mode0]):int(b_r[mode0])]
            self.app_plane[i] = self.app_plane[i][...,int(t_l[mode1]):int(b_r[mode1]),int(t_l[mode0]):int(b_r[mode0])]
            self.norm_plane[i] = self.norm_plane[i][...,int(t_l[mode1]):int(b_r[mode1]),int(t_l[mode0]):int(b_r[mode0])]


        if not jt.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = jt.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))
    '''

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
            app_features, rgb_d, specular_tint, normal_vector, rho = self.compute_appfeature(xyz_sampled[app_mask])
            # 重要！！！
            normal_vector = jt.normalize(normal_vector,dim=-1) # 归一化
            input_views = viewdirs[app_mask]
            d = - input_views
            # 点积
            dot_product = jt.multiply(d, normal_vector).sum(dim=1)
            shape = list(dot_product.shape) + [1]
            dot_product = jt.broadcast(dot_product, shape=shape,dims=[-1])
            # 获取反射方向
            reflection = 2 * dot_product * normal_vector - d
            # Intergrated Directional Encoding
            # 方向MLP的输入是bottle neck、reflection的ide、点积拼起来
            # 镜面反射
            rgb_s = self.renderModule(xyz_sampled[app_mask], reflection, app_features, -dot_product, 1/rho)

            # 乘以镜面反射率 再与空间MLP产生的漫反射颜色相加
            valid_rgbs = (specular_tint) * jt.clamp(rgb_s,0) + rgb_d

            rgb[app_mask] = valid_rgbs

            # 计算法向量的penalty
            penalty = jt.nn.relu(-dot_product) # 取得n*d
            penalty = jt.sqr(penalty).squeeze(-1)
            self.penalty = jt.sum(weight[app_mask] * penalty, -1)

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


