import numpy as np
import os,sys,time
import jittor as jt
import jittor.nn as torch_F
import tqdm
from easydict import EasyDict as edict
import imageio
import json

import util,util_vis
from util import log,debug
from . import base
import camera

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
mse2psnr = lambda x : -10. * jt.log(x) / jt.log(jt.array(np.array([10.])))

# ============================ main engine for training and evaluation ============================

class Model(base.Model):

    def __init__(self,opt):
        super().__init__(opt)

    def load_dataset(self,opt,eval_split="val"):
        super().load_dataset(opt,eval_split=eval_split)
        # prefetch all training data
        self.train_data.prefetch_all_data(opt)
        self.train_data.all = edict(self.train_data.all)

    def setup_optimizer(self,opt):
        log.info("setting up optimizers...")
        optimizer = getattr(jt.optim,opt.optim.algo)
        self.optim = optimizer([dict(params=self.graph.nerf.parameters(),lr=opt.optim.lr)],lr=opt.optim.lr)
        if opt.nerf.fine_sampling:
            self.optim.add_param_group(dict(params=self.graph.nerf_fine.parameters(),lr=opt.optim.lr))
        # set up scheduler
        if opt.optim.sched:
            scheduler = getattr(jt.lr_scheduler,opt.optim.sched.type)
            if opt.optim.lr_end:
                assert(opt.optim.sched.type=="ExponentialLR")
                opt.optim.sched.gamma = (opt.optim.lr_end/opt.optim.lr)**(1./opt.max_iter)
            kwargs = { k:v for k,v in opt.optim.sched.items() if k!="type" }
            self.sched = scheduler(self.optim,**kwargs)

    def train(self,opt):
        # before training
        log.title("TRAINING START")
        jt.gc()
        self.timer = edict(start=time.time(),it_mean=None)
        self.graph.train()
        self.ep = 0 # dummy for timer
        # training
        if self.iter_start==0: self.validate(opt,0)
        loader = tqdm.trange(opt.max_iter,desc="training",leave=False)
        for self.it in loader:
            self.graph.train()
            if self.it<self.iter_start: continue
            # set var to all available images
            var = self.train_data.all
            self.train_iteration(opt,var,loader)
            if opt.optim.sched: self.sched.step()
            if self.it%opt.freq.val==0: self.validate(opt,self.it)
            if self.it%opt.freq.ckpt==0: self.save_checkpoint(opt,ep=None,it=self.it)
        # after training
        if opt.tb:
            self.tb.flush()
            self.tb.close()
        log.title("TRAINING DONE")

    @jt.no_grad()
    def log_scalars(self,opt,var,loss,metric=None,step=0,split="train"):
        super().log_scalars(opt,var,loss,metric=metric,step=step,split=split)
        # log learning rate
        if split=="train":
            lr = self.optim.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split,"lr"),lr,step)
            if opt.nerf.fine_sampling:
                lr = self.optim.param_groups[1]["lr"]
                self.tb.add_scalar("{0}/{1}".format(split,"lr_fine"),lr,step)
        # compute PSNR
        psnr = mse2psnr(loss.render)
        self.tb.add_scalar("{0}/{1}".format(split,"PSNR"),psnr.item(),step)
        if opt.nerf.fine_sampling:
            psnr = mse2psnr(loss.render_fine)
            self.tb.add_scalar("{0}/{1}".format(split,"PSNR_fine"),psnr.item(),step)

    @jt.no_grad()
    def visualize(self,opt,var,step=0,split="train",eps=1e-10):
        if opt.tb:
            util_vis.tb_image(opt,self.tb,step,split,"image",var.image)
            if not opt.nerf.rand_rays or split!="train":
                invdepth = (1-var.depth)/var.opacity if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
                rgb_map = var.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
                invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                util_vis.tb_image(opt,self.tb,step,split,"rgb",rgb_map)
                util_vis.tb_image(opt,self.tb,step,split,"invdepth",invdepth_map)
                if opt.nerf.fine_sampling:
                    invdepth = (1-var.depth_fine)/var.opacity_fine if opt.camera.ndc else 1/(var.depth_fine/var.opacity_fine+eps)
                    rgb_map = var.rgb_fine.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
                    invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                    util_vis.tb_image(opt,self.tb,step,split,"rgb_fine",rgb_map)
                    util_vis.tb_image(opt,self.tb,step,split,"invdepth_fine",invdepth_map)

    @jt.no_grad()
    def get_all_training_poses(self,opt):
        # get ground-truth (canonical) camera poses
        pose_GT = self.train_data.get_all_camera_poses(opt)
        return None,pose_GT

    @jt.no_grad()
    def evaluate_full(self,opt,eps=1e-10):
        self.graph.eval()
        loader = tqdm.tqdm(self.test_loader,desc="evaluating",leave=False)
        res = []
        test_path = "{}/test_view".format(opt.output_path)
        os.makedirs(test_path,exist_ok=True)
        transform = []
        pose_flip = camera.pose(R=jt.diag(jt.array([-1,-1,1])))
        for i,batch in enumerate(loader):
            var = edict(batch)
            imageio.imwrite("{}/gt_{}.png".format(test_path,i), var.image.permute(0,2,3,1)[0].numpy())
            if opt.model=="barf" and opt.optim.test_photo:
                # run test-time optimization to factorize imperfection in optimized poses from view synthesis evaluation
                var = self.evaluate_test_time_photometric_optim(opt,var)
                p = self.graph.get_pose(opt,var,mode="test-optim")
                p = camera.pose.invert(p.squeeze(0))
                p = camera.pose.compose([pose_flip, p])
                p = jt.concat([p,jt.array([[0,0,0,1]])])
                t = {}
                t["file_path"]=  "./val/r_"+str(i)
                t["transform_matrix"]= [[float(i) for i in col.numpy()] for col in p]
                transform.append(t)
            var = self.graph.execute(opt,var,mode="eval")
            # evaluate view synthesis
            invdepth = (1-var.depth)/var.opacity if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
            rgb_map = var.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
            invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
            psnr = mse2psnr(self.graph.MSE_loss(rgb_map,var.image)).item()
            res.append(edict(psnr=psnr))
            rgb_map = rgb_map.permute(0,2,3,1)
            # dump novel views
            imageio.imwrite("{}/rgb_{}.png".format(test_path,i), rgb_map[0].numpy())
        
        file = {
            "camera_angle_x": 1.0471975511965976,
            "frames": transform
        }
        with open(opt.output_path+"/transform_val.json","w") as f:
            json.dump(file,f,indent=4)
        print("PSNR:  {:8.2f}".format(np.mean([r.psnr for r in res])))
        print("--------------------------")
        # dump numbers to file
        quant_fname = "{}/quant.txt".format(opt.output_path)
        with open(quant_fname,"w") as file:
            for i,r in enumerate(res):
                file.write(i,r.psnr)


    @jt.no_grad()
    def generate_videos_synthesis(self,opt,eps=1e-10):
        self.graph.eval()
        if opt.data.dataset=="blender":
            test_path = "{}/test_view".format(opt.output_path)
            # assume the test view synthesis are already generated
            print("writing videos...")
            rgb_vid_fname = "{}/test_view_rgb.mp4".format(opt.output_path)
            depth_vid_fname = "{}/test_view_depth.mp4".format(opt.output_path)
            os.system("ffmpeg -y -framerate 30 -i {0}/rgb_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(test_path,rgb_vid_fname))
            os.system("ffmpeg -y -framerate 30 -i {0}/depth_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(test_path,depth_vid_fname))
        else:
            pose_pred,pose_GT = self.get_all_training_poses(opt)
            poses = pose_pred if opt.model=="barf" else pose_GT
            if opt.model=="barf" and opt.data.dataset=="llff":
                _,sim3 = self.prealign_cameras(opt,pose_pred,pose_GT)
                scale = sim3.s1/sim3.s0
            else: scale = 1
            # rotate novel views around the "center" camera of all poses
            idx_center = (poses-poses.mean(dim=0,keepdim=True))[...,3].norm(dim=-1).argmin()
            pose_novel = camera.get_novel_view_poses(opt,poses[idx_center],N=60,scale=scale)
            # render the novel views
            novel_path = "{}/novel_view".format(opt.output_path)
            os.makedirs(novel_path,exist_ok=True)
            pose_novel_tqdm = tqdm.tqdm(pose_novel,desc="rendering novel views",leave=False)
            intr = edict(next(iter(self.test_loader))).intr[:1] # grab intrinsics
            for i,pose in enumerate(pose_novel_tqdm):
                ret = self.graph.render_by_slices(opt,pose[None],intr=intr) if opt.nerf.rand_rays else \
                      self.graph.render(opt,pose[None],intr=intr)
                invdepth = (1-ret.depth)/ret.opacity if opt.camera.ndc else 1/(ret.depth/ret.opacity+eps)
                rgb_map = ret.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
                invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                imageio.imwrite("{}/rgb_{}.png".format(novel_path,i), to8b(rgb_map))
                # torchvision_F.to_pil_image(invdepth_map[0]).save("{}/depth_{}.png".format(novel_path,i))
            # write videos
            print("writing videos...")
            rgb_vid_fname = "{}/novel_view_rgb.mp4".format(opt.output_path)
            depth_vid_fname = "{}/novel_view_depth.mp4".format(opt.output_path)
            os.system("ffmpeg -y -framerate 30 -i {0}/rgb_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(novel_path,rgb_vid_fname))
            os.system("ffmpeg -y -framerate 30 -i {0}/depth_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(novel_path,depth_vid_fname))

# ============================ computation graph for execute/backprop ============================

class Graph(base.Graph):

    def __init__(self,opt):
        super().__init__(opt)
        self.nerf = NeRF(opt)
        if opt.nerf.fine_sampling:
            self.nerf_fine = NeRF(opt)

    def execute(self,opt,var,mode=None):
        batch_size = len(var.idx)
        pose = self.get_pose(opt,var,mode=mode)
        # render images
        if opt.nerf.rand_rays and mode in ["train","test-optim"]:
            # sample random rays for optimization
            # var.ray_idx = jt.arange(0,opt.H*opt.W)[:opt.nerf.rand_rays//batch_size]

            var.ray_idx = jt.randperm(opt.H*opt.W)[:opt.nerf.rand_rays//batch_size]
            ret = self.render(opt,pose,intr=var.intr,ray_idx=var.ray_idx,mode=mode) # [B,N,3],[B,N,1]
        else:
            # render full image (process in slices)
            ret = self.render_by_slices(opt,pose,intr=var.intr,mode=mode) if opt.nerf.rand_rays else \
                  self.render(opt,pose,intr=var.intr,mode=mode) # [B,HW,3],[B,HW,1]
        var.update(ret)
        return var

    def compute_loss(self,opt,var,mode=None):
        loss = edict()
        batch_size = len(var.idx)
        image = var.image.view(batch_size,3,opt.H*opt.W).permute(0,2,1)
        if opt.nerf.rand_rays > 0 and mode in ["train","test-optim"]:
            image = image[:,var.ray_idx]
        # compute image losses
        if opt.loss_weight.render is not None:
            loss.render = self.MSE_loss(var.rgb,image)
        if opt.loss_weight.render_fine is not None:
            assert(opt.nerf.fine_sampling)
            loss.render_fine = self.MSE_loss(var.rgb_fine,image)
        return loss

    def get_pose(self,opt,var,mode=None):
        return var.pose

    def render(self,opt,pose,intr=None,ray_idx=None,mode=None):
        batch_size = len(pose)
        center,ray = camera.get_center_and_ray(opt,pose,intr=intr) # [B,HW,3]
        while ray.isnan().any(): # TODO: weird bug, ray becomes NaN arbitrarily if batch_size>1, not deterministic reproducible
            center,ray = camera.get_center_and_ray(opt,pose,intr=intr) # [B,HW,3]
        if ray_idx is not None:
            # consider only subset of rays
            center,ray = center[:,ray_idx],ray[:,ray_idx]
        if opt.camera.ndc:
            # convert center/ray representations to NDC
            center,ray = camera.convert_NDC(opt,center,ray,intr=intr)
        # render with main MLP
        depth_samples = self.sample_depth(opt,batch_size,num_rays=ray.shape[1]) # [B,HW,N,1]
        rgb_samples,density_samples = self.nerf.execute_samples(opt,center,ray,depth_samples,mode=mode)
        rgb,depth,opacity,prob = self.nerf.composite(opt,ray,rgb_samples,density_samples,depth_samples)
        ret = edict(rgb=rgb,depth=depth,opacity=opacity) # [B,HW,K]
        # render with fine MLP from coarse MLP
        if opt.nerf.fine_sampling:
            with jt.no_grad():
                # resample depth acoording to coarse empirical distribution
                depth_samples_fine = self.sample_depth_from_pdf(opt,pdf=prob[...,0]) # [B,HW,Nf,1]
                depth_samples = jt.concat([depth_samples,depth_samples_fine],dim=2) # [B,HW,N+Nf,1]
                depth_samples = depth_samples.sort(dim=2).values
            rgb_samples,density_samples = self.nerf_fine.execute_samples(opt,center,ray,depth_samples,mode=mode)
            rgb_fine,depth_fine,opacity_fine,_ = self.nerf_fine.composite(opt,ray,rgb_samples,density_samples,depth_samples)
            ret.update(rgb_fine=rgb_fine,depth_fine=depth_fine,opacity_fine=opacity_fine) # [B,HW,K]
        return ret

    def render_by_slices(self,opt,pose,intr=None,mode=None):
        ret_all = edict(rgb=[],depth=[],opacity=[])
        if opt.nerf.fine_sampling:
            ret_all.update(rgb_fine=[],depth_fine=[],opacity_fine=[])
        # render the image by slices for memory considerations
        for c in range(0,opt.H*opt.W,opt.nerf.rand_rays):
            ray_idx = jt.arange(c,min(c+opt.nerf.rand_rays,opt.H*opt.W))
            ret = self.render(opt,pose,intr=intr,ray_idx=ray_idx,mode=mode) # [B,R,3],[B,R,1]
            for k in ret: ret_all[k].append(ret[k])
        # group all slices of images
        for k in ret_all: ret_all[k] = jt.concat(ret_all[k],dim=1)
        return ret_all

    def sample_depth(self,opt,batch_size,num_rays=None):
        depth_min,depth_max = opt.nerf.depth.range
        num_rays = num_rays or opt.H*opt.W
        rand_samples = jt.rand(batch_size,num_rays,opt.nerf.sample_intvs,1) if opt.nerf.sample_stratified else 0.5
        rand_samples += jt.arange(opt.nerf.sample_intvs)[None,None,:,None].float() # [B,HW,N,1]
        depth_samples = rand_samples/opt.nerf.sample_intvs*(depth_max-depth_min)+depth_min # [B,HW,N,1]
        depth_samples = dict(
            metric=depth_samples,
            inverse=1/(depth_samples+1e-8),
        )[opt.nerf.depth.param]
        return depth_samples

    def sample_depth_from_pdf(self,opt,pdf):
        depth_min,depth_max = opt.nerf.depth.range
        # get CDF from PDF (along last dimension)
        cdf = pdf.cumsum(dim=-1) # [B,HW,N]
        cdf = jt.concat([jt.zeros_like(cdf[...,:1]),cdf],dim=-1) # [B,HW,N+1]
        # take uniform samples
        grid = jt.linspace(0,1,opt.nerf.sample_intvs_fine+1) # [Nf+1]
        unif = 0.5*(grid[:-1]+grid[1:]).repeat(*cdf.shape[:-1],1) # [B,HW,Nf]
        idx = jt.searchsorted(cdf,unif,right=True) # [B,HW,Nf] \in {1...N}
        # inverse transform sampling from CDF
        depth_bin = jt.linspace(depth_min,depth_max,opt.nerf.sample_intvs+1) # [N+1]
        depth_bin = depth_bin.repeat(*cdf.shape[:-1],1) # [B,HW,N+1]
        depth_low = depth_bin.gather(dim=2,index=(idx-1).clamp(min_v=0)) # [B,HW,Nf]
        depth_high = depth_bin.gather(dim=2,index=idx.clamp(max_v=opt.nerf.sample_intvs)) # [B,HW,Nf]
        cdf_low = cdf.gather(dim=2,index=(idx-1).clamp(min_v=0)) # [B,HW,Nf]
        cdf_high = cdf.gather(dim=2,index=idx.clamp(max_v=opt.nerf.sample_intvs)) # [B,HW,Nf]
        # linear interpolation
        t = (unif-cdf_low)/(cdf_high-cdf_low+1e-8) # [B,HW,Nf]
        depth_samples = depth_low+t*(depth_high-depth_low) # [B,HW,Nf]
        return depth_samples[...,None] # [B,HW,Nf,1]

class NeRF(jt.nn.Module):

    def __init__(self,opt):
        super().__init__()
        self.define_network(opt)

    def define_network(self,opt):
        input_3D_dim = 3+6*opt.arch.posenc.L_3D if opt.arch.posenc else 3
        if opt.nerf.view_dep:
            input_view_dim = 3+6*opt.arch.posenc.L_view if opt.arch.posenc else 3
        # point-wise feature
        self.mlp_feat = jt.nn.ModuleList()
        L = util.get_layer_dims(opt.arch.layers_feat)
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = input_3D_dim
            if li in opt.arch.skip: k_in += input_3D_dim
            if li==len(L)-1: k_out += 1
            linear = jt.nn.Linear(k_in,k_out)
            if opt.arch.tf_init:
                self.tensorflow_init_weights(opt,linear,out="first" if li==len(L)-1 else None)
            self.mlp_feat.append(linear)
        # RGB prediction
        self.mlp_rgb = jt.nn.ModuleList()
        L = util.get_layer_dims(opt.arch.layers_rgb)
        feat_dim = opt.arch.layers_feat[-1]
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = feat_dim+(input_view_dim if opt.nerf.view_dep else 0)
            linear = jt.nn.Linear(k_in,k_out)
            if opt.arch.tf_init:
                self.tensorflow_init_weights(opt,linear,out="all" if li==len(L)-1 else None)
            self.mlp_rgb.append(linear)

    def tensorflow_init_weights(self,opt,linear,out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = jt.nn.init.calculate_gain("relu") # sqrt(2)
        if out=="all":
            jt.nn.init.xavier_uniform_(linear.weight)
        elif out=="first":
            jt.nn.init.xavier_uniform_(linear.weight[:1])
            jt.nn.init.xavier_uniform_(linear.weight[1:],gain=relu_gain)
        else:
            jt.nn.init.xavier_uniform_(linear.weight,gain=relu_gain)
        jt.nn.init.zero_(linear.bias)

    def execute(self,opt,points_3D,ray_unit=None,mode=None): # [B,...,3]
        if opt.arch.posenc:
            points_enc = self.positional_encoding(opt,points_3D,L=opt.arch.posenc.L_3D)
            points_enc = jt.concat([points_3D,points_enc],dim=-1) # [B,...,6L+3]
        else: points_enc = points_3D
        feat = points_enc
        # extract coordinate-based features
        for li,layer in enumerate(self.mlp_feat):
            if li in opt.arch.skip: feat = jt.concat([feat,points_enc],dim=-1)
            feat = layer(feat)
            if li==len(self.mlp_feat)-1:
                density = feat[...,0]
                if opt.nerf.density_noise_reg and mode=="train":
                    density += jt.randn_like(density)*opt.nerf.density_noise_reg
                density_activ = getattr(torch_F,opt.arch.density_activ) # relu_,abs_,sigmoid_,exp_....
                density = density_activ(density)
                feat = feat[...,1:]
            feat = torch_F.relu(feat)
        # predict RGB values
        if opt.nerf.view_dep:
            assert(ray_unit is not None)
            if opt.arch.posenc:
                ray_enc = self.positional_encoding(opt,ray_unit,L=opt.arch.posenc.L_view)
                ray_enc = jt.concat([ray_unit,ray_enc],dim=-1) # [B,...,6L+3]
            else: ray_enc = ray_unit
            feat = jt.concat([feat,ray_enc],dim=-1)
        for li,layer in enumerate(self.mlp_rgb):
            feat = layer(feat)
            if li!=len(self.mlp_rgb)-1:
                feat = torch_F.relu(feat)
        rgb = feat.sigmoid() # [B,...,3]
        return rgb,density

    def execute_samples(self,opt,center,ray,depth_samples,mode=None):
        points_3D_samples = camera.get_3D_points_from_depth(opt,center,ray,depth_samples,multi_samples=True) # [B,HW,N,3]
        if opt.nerf.view_dep:
            ray_unit = jt.misc.normalize(ray,dim=-1) # [B,HW,3]
            ray_unit_samples = ray_unit[...,None,:].expand_as(points_3D_samples) # [B,HW,N,3]
        else: ray_unit_samples = None
        rgb_samples,density_samples = self.execute(opt,points_3D_samples,ray_unit=ray_unit_samples,mode=mode) # [B,HW,N],[B,HW,N,3]
        return rgb_samples,density_samples

    def composite(self,opt,ray,rgb_samples,density_samples,depth_samples):
        ray_length = ray.norm(dim=-1,keepdim=True) # [B,HW,1]
        # volume rendering: compute probability (using quadrature)
        depth_intv_samples = depth_samples[...,1:,0]-depth_samples[...,:-1,0] # [B,HW,N-1]
        depth_intv_samples = jt.concat([depth_intv_samples,jt.full_like(depth_intv_samples[...,:1],1e10)],dim=2) # [B,HW,N]
        dist_samples = depth_intv_samples*ray_length # [B,HW,N]
        sigma_delta = density_samples*dist_samples # [B,HW,N]
        alpha = 1-(-sigma_delta).exp() # [B,HW,N]
        T = (-jt.concat([jt.zeros_like(sigma_delta[...,:1]),sigma_delta[...,:-1]],dim=2).cumsum(dim=2)).exp() # [B,HW,N]
        prob = (T*alpha)[...,None] # [B,HW,N,1]
        # integrate RGB and depth weighted by probability
        depth = (depth_samples*prob).sum(dim=2) # [B,HW,1]
        rgb = (rgb_samples*prob).sum(dim=2) # [B,HW,3]
        opacity = prob.sum(dim=2) # [B,HW,1]
        if opt.nerf.setbg_opaque:
            rgb = rgb+opt.data.bgcolor*(1-opacity)
        return rgb,depth,opacity,prob # [B,HW,K]

    def positional_encoding(self,opt,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**jt.arange(L,dtype='float32')*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = jt.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc
