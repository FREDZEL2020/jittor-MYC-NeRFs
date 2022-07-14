import numpy as np
import os,sys,time
import jittor as jt
import jittor.nn as torch_F
import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt

import util,util_vis
from util import log,debug
from . import nerf
import camera
import json

# ============================ main engine for training and evaluation ============================
class MyEmbedding(jt.nn.Module):
    ''' A simple lookup table that stores embeddings of a fixed dictionary and size.

        :param num: size of the dictionary of embeddings
        :type num: int

        :param dim: the size of each embedding vector
        :type dim: int

        Example:
            >>> embedding = nn.Embedding(10, 3)
            >>> x = jt.int32([1, 2, 3, 3])
            >>> embedding(x)
            jt.Var([[ 1.1128596   0.19169547  0.706642]
             [ 1.2047412   1.9668795   0.9932192]
             [ 0.14941819  0.57047683 -1.3217674]
             [ 0.14941819  0.57047683 -1.3217674]], dtype=float32)
    '''
    def __init__(self, num, dim):
        self.num = num
        self.dim = dim
        self.weight = jt.empty([num, dim])

    def execute(self, x):
        res = self.weight[x.flatten()].reshape(x.shape + [self.dim])
        return res

class Model(nerf.Model):

    def __init__(self,opt):
        super().__init__(opt)

    def build_networks(self,opt):
        super().build_networks(opt)
        if opt.camera.noise:
            # pre-generate synthetic pose perturbation
            se3_noise = jt.randn(len(self.train_data),6)*opt.camera.noise
            self.graph.pose_noise = camera.lie.se3_to_SE3(se3_noise)
        self.graph.se3_refine = MyEmbedding(len(self.train_data),6)
        # self.graph.se3_refine.requires_grad_(True)
        # self.graph.se3_refine.weight.requires_grad=True
        jt.nn.init.zero_(self.graph.se3_refine.weight)

    def setup_optimizer(self,opt):
        super().setup_optimizer(opt)
        optimizer = getattr(jt.optim,opt.optim.algo)
        self.optim_pose = optimizer([dict(params=self.graph.se3_refine.parameters(),lr=opt.optim.lr_pose)], lr=opt.optim.lr_pose)
        # set up scheduler
        if opt.optim.sched_pose:
            scheduler = getattr(jt.lr_scheduler,opt.optim.sched_pose.type)
            if opt.optim.lr_pose_end:
                assert(opt.optim.sched_pose.type=="ExponentialLR")
                opt.optim.sched_pose.gamma = (opt.optim.lr_pose_end/opt.optim.lr_pose)**(1./opt.max_iter)
            kwargs = { k:v for k,v in opt.optim.sched_pose.items() if k!="type" }
            self.sched_pose = scheduler(self.optim_pose,**kwargs)

    def train_iteration(self,opt,var,loader):
        self.optim_pose.zero_grad()
        if opt.optim.warmup_pose:
            # simple linear warmup of pose learning rate
            self.optim_pose.param_groups[0]["lr_orig"] = self.optim_pose.param_groups[0]["lr"] # cache the original learning rate
            self.optim_pose.param_groups[0]["lr"] *= min(1,self.it/opt.optim.warmup_pose)
        loss = super().train_iteration(opt,var,loader)
        self.optim_pose.backward(loss.all)
        self.optim_pose.step()
        if opt.optim.warmup_pose:
            self.optim_pose.param_groups[0]["lr"] = self.optim_pose.param_groups[0]["lr_orig"] # reset learning rate
        if opt.optim.sched_pose: self.sched_pose.step()
        self.graph.nerf.progress.data = jt.full(self.graph.nerf.progress.data.shape,(self.it/opt.max_iter))
        if opt.nerf.fine_sampling:
            self.graph.nerf_fine.progress.data = jt.full(self.graph.nerf_fine.progress.data.shape ,self.it/opt.max_iter)
        # jt.gc()
        return loss

    @jt.no_grad()
    def validate(self,opt,ep=None):
        pose,pose_GT = self.get_all_training_poses(opt)
        _,self.graph.sim3 = self.prealign_cameras(opt,pose,pose_GT)
        super().validate(opt,ep=ep)

    @jt.no_grad()
    def log_scalars(self,opt,var,loss,metric=None,step=0,split="train"):
        super().log_scalars(opt,var,loss,metric=metric,step=step,split=split)
        if split=="train":
            # log learning rate
            lr = self.optim_pose.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split,"lr_pose"),lr,step)
        # compute pose error
        if split=="train" and opt.data.dataset in ["blender","llff"]:
            pose,pose_GT = self.get_all_training_poses(opt)
            pose_aligned,_ = self.prealign_cameras(opt,pose,pose_GT)
            error = self.evaluate_camera_alignment(opt,pose_aligned,pose_GT)
            self.tb.add_scalar("{0}/error_R".format(split),error.R.mean().item(),step)
            self.tb.add_scalar("{0}/error_t".format(split),error.t.mean().item(),step)

    @jt.no_grad()
    def visualize(self,opt,var,step=0,split="train"):
        super().visualize(opt,var,step=step,split=split)
        
    @jt.no_grad()
    def get_all_training_poses(self,opt):
        # get ground-truth (canonical) camera poses
        pose_GT = self.train_data.get_all_camera_poses(opt)
        # add synthetic pose perturbation to all training data
        if opt.data.dataset=="blender":
            pose = pose_GT
            if opt.camera.noise:
                pose = camera.pose.compose([self.graph.pose_noise,pose])
        else: pose = self.graph.pose_eye
        # add learned pose correction to all training data
        pose_refine = camera.lie.se3_to_SE3(self.graph.se3_refine.weight)
        pose = camera.pose.compose([pose_refine,pose])
        return pose,pose_GT

    @jt.no_grad()
    def prealign_cameras(self,opt,pose,pose_GT):
        # compute 3D similarity transform via Procrustes analysis
        center = jt.zeros([1,1,3])
        center_pred = camera.cam2world(center,pose)[:,0] # [N,3]
        center_GT = camera.cam2world(center,pose_GT)[:,0] # [N,3]
        sim3 = camera.procrustes_analysis(center_GT,center_pred)
        # align the camera poses
        center_aligned = (center_pred-sim3.t1)/sim3.s1 @ (sim3.R.t()) *sim3.s0+sim3.t0
        R_aligned = pose[...,:3]@ (sim3.R.t())
        t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
        pose_aligned = camera.pose(R=R_aligned,t=t_aligned)
        return pose_aligned,sim3

    @jt.no_grad()
    def evaluate_camera_alignment(self,opt,pose_aligned,pose_GT):
        # measure errors in rotation and translation
        R_aligned,t_aligned = pose_aligned.split([3,1],dim=-1)
        R_GT,t_GT = pose_GT.split([3,1],dim=-1)
        R_error = camera.rotation_distance(R_aligned,R_GT)
        t_error = (t_aligned-t_GT)[...,0].norm(dim=-1)
        error = edict(R=R_error,t=t_error)
        return error

    @jt.no_grad()
    def get_transform(self,p,p_gt):
        pose_flip = camera.pose(R=jt.diag(jt.array([-1,-1,1])))
        p = camera.pose.invert(p)
        p = camera.pose.compose([pose_flip, p])
        p_gt = camera.pose.invert(p_gt)
        p_gt = camera.pose.compose([pose_flip, p_gt])
        p = jt.concat([p,jt.array([[0,0,0,1]])])
        p_gt = jt.concat([p_gt,jt.array([[0,0,0,1]])])
        p_inv = jt.linalg.inv(p)
        world_trans = p_gt @ p_inv
        return world_trans

    def get_pose_transfrom(self,opt,pose,pose_GT):
        pose_flip = camera.pose(R=jt.diag(jt.array([-1,-1,1])))

        for i in range(30,31):
            p=pose[i]
            p_gt = pose_GT[i]
            p = camera.pose.invert(p)
            p = camera.pose.compose([pose_flip, p])
            p_gt = camera.pose.invert(p_gt)
            p_gt = camera.pose.compose([pose_flip, p_gt])
            p = jt.concat([p,jt.array([[0,0,0,1]])])
            p_gt = jt.concat([p_gt,jt.array([[0,0,0,1]])])
            p_inv = jt.linalg.inv(p)
            world_trans = p_gt @ p_inv
            print(world_trans)
            
        
        transform = []
        i=0
        for p in pose:
            p = camera.pose.invert(p)
            p = camera.pose.compose([pose_flip, p])
            p = jt.concat([p,jt.array([[0,0,0,1]])])
            # p = world_trans @ p
            t = {}
            t["file_path"]=  "./train/r_"+str(i)
            t["transform_matrix"]= [[float(i) for i in col.numpy()] for col in p]
            transform.append(t)
            i+=1
        # please change your own camera angle
        file = {
            "camera_angle_x": 1.0471975511965976,
            "frames": transform
        }
        with open(opt.output_path +"/transform_train.json","w") as f:
            json.dump(file,f)

    @jt.no_grad()
    def evaluate_full(self,opt):
        self.graph.eval()
        # evaluate rotation/translation
        pose,pose_GT = self.get_all_training_poses(opt)

        self.get_pose_transfrom(opt,pose,pose_GT)

        pose_aligned,self.graph.sim3 = self.prealign_cameras(opt,pose,pose_GT)
        error = self.evaluate_camera_alignment(opt,pose_aligned,pose_GT)
        print("--------------------------")
        print("rot:   {:8.3f}".format(np.rad2deg(error.R.mean().item())))
        print("trans: {:10.5f}".format(error.t.mean().item()))
        print("--------------------------")
        # dump numbers
        quant_fname = "{}/quant_pose.txt".format(opt.output_path)
        with open(quant_fname,"w") as file:
            for i,(err_R,err_t) in enumerate(zip(error.R,error.t)):
                file.write("{} {} {}\n".format(i,err_R.item(),err_t.item()))
        # evaluate novel view synthesis
        super().evaluate_full(opt)

    @jt.enable_grad()
    def evaluate_test_time_photometric_optim(self,opt,var):
        # use another se3 Parameter to absorb the remaining pose errors
        var.se3_refine_test = jt.nn.Parameter(jt.zeros((1,6)))
        optimizer = getattr(jt.optim,opt.optim.algo)
        optim_pose = optimizer([dict(params=[var.se3_refine_test],lr=opt.optim.lr_pose)], lr=opt.optim.lr)
        opt.optim.test_iter = 10000
        min_l = 100
        min_one = 0
        iterator = tqdm.trange(opt.optim.test_iter,desc="test-time optim.",leave=False,position=1)
        for it in iterator:
            self.graph.train()
            optim_pose.zero_grad()
            var.pose_refine_test = camera.lie.se3_to_SE3(var.se3_refine_test)
            var = self.graph.execute(opt,var,mode="test-optim")
            loss = self.graph.compute_loss(opt,var,mode="test-optim")
            loss = self.summarize_loss(opt,var,loss)
            # loss.all.backward()
            optim_pose.backward(loss.all)
            optim_pose.step()
            iterator.set_postfix(loss="{:.3f}".format(loss.all))
            l = float(loss.all)
            if l < 0.0007:
                break
            if l < min_l - 0.0001:
                min_l = l
                min_one = it
            if it > 5000 and it > min_one + 1000:
                break
        pose = self.graph.get_pose(opt,var,mode="test-optim")
        trans = self.get_transform(pose.squeeze(0), var.pose.squeeze(0))
        detail_name = "{}/loss.txt".format(opt.output_path)
        with open(detail_name,"a") as file:
            file.write(str(loss.all)+" ")
            file.write(str(it)+"\n")
            file.write(str(trans)+"\n")
        return var

    @jt.no_grad()
    def generate_videos_pose(self,opt):
        self.graph.eval()
        fig = plt.figure(figsize=(10,10) if opt.data.dataset=="blender" else (16,8))
        cam_path = "{}/poses".format(opt.output_path)
        os.makedirs(cam_path,exist_ok=True)
        ep_list = []
        for ep in range(0,opt.max_iter+1,opt.freq.ckpt):
            # load checkpoint (0 is random init)
            if ep!=0:
                try: util.restore_checkpoint(opt,self,resume=ep)
                except: continue
            # get the camera poses
            pose,pose_ref = self.get_all_training_poses(opt)
            if opt.data.dataset in ["blender","llff"]:
                pose_aligned,_ = self.prealign_cameras(opt,pose,pose_ref)
                pose_aligned,pose_ref = pose_aligned.detach(),pose_ref.detach()
                dict(
                    blender=util_vis.plot_save_poses_blender,
                    llff=util_vis.plot_save_poses,
                )[opt.data.dataset](opt,fig,pose_aligned,pose_ref=pose_ref,path=cam_path,ep=ep)
            else:
                pose = pose.detach()
                util_vis.plot_save_poses(opt,fig,pose,pose_ref=None,path=cam_path,ep=ep)
            ep_list.append(ep)
        plt.close()
        # write videos
        print("writing videos...")
        list_fname = "{}/temp.list".format(cam_path)
        with open(list_fname,"w") as file:
            for ep in ep_list: file.write("file {}.png\n".format(ep))
        cam_vid_fname = "{}/poses.mp4".format(opt.output_path)
        os.system("ffmpeg -y -r 30 -f concat -i {0} -pix_fmt yuv420p {1} >/dev/null 2>&1".format(list_fname,cam_vid_fname))
        os.remove(list_fname)

# ============================ computation graph for execute/backprop ============================

class Graph(nerf.Graph):

    def __init__(self,opt):
        super().__init__(opt)
        self.nerf = NeRF(opt)
        if opt.nerf.fine_sampling:
            self.nerf_fine = NeRF(opt)
        self.pose_eye = jt.init.eye([3,4])

    def get_pose(self,opt,var,mode=None):
        if mode=="train":
            # add the pre-generated pose perturbations
            if opt.data.dataset=="blender":
                if opt.camera.noise:
                    var.pose_noise = self.pose_noise[var.idx]
                    pose = camera.pose.compose([var.pose_noise,var.pose])
                else: pose = var.pose
            else: pose = self.pose_eye
            # add learnable pose correction
            var.se3_refine = self.se3_refine.weight[var.idx]
            pose_refine = camera.lie.se3_to_SE3(var.se3_refine)
            pose = camera.pose.compose([pose_refine,pose])
        elif mode in ["val","eval","test-optim"]:
            # align test pose to refined coordinate system (up to sim3)
            sim3 = self.sim3
            center = jt.zeros((1,1,3))
            center = camera.cam2world(center,var.pose)[:,0] # [N,3]
            center_aligned = (center-sim3.t0)/sim3.s0@sim3.R*sim3.s1+sim3.t1
            R_aligned = var.pose[...,:3]@self.sim3.R
            t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
            pose = camera.pose(R=R_aligned,t=t_aligned)
            # additionally factorize the remaining pose imperfection
            if opt.optim.test_photo and mode!="val":
                pose = camera.pose.compose([var.pose_refine_test,pose])
        else: pose = var.pose
        return pose

class NeRF(nerf.NeRF):

    def __init__(self,opt):
        super().__init__(opt)
        self.progress = jt.nn.Parameter(jt.Var(0.)) # use Parameter so it could be checkpointed

    def positional_encoding(self,opt,input,L): # [B,...,N]
        input_enc = super().positional_encoding(opt,input,L=L) # [B,...,2NL]
        # coarse-to-fine: smoothly mask positional encoding for BARF
        if opt.barf_c2f is not None:
            # set weights for different frequency bands
            start,end = opt.barf_c2f
            alpha = (self.progress.data-start)/(end-start)*L
            k = jt.arange(L,dtype='float32')
            w = jt.clamp((alpha-k),min_v=0,max_v=1)
            weight = (1-(w.multiply(np.pi)).cos())/2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1,L)*weight).view(*shape)
        return input_enc
