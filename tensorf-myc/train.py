
import os
from turtle import begin_poly
from tqdm.auto import tqdm
from opt import config_parser



import json, random
from renderer import *
from utils import *
from tensorboardX import SummaryWriter
import datetime

from dataLoader import dataset_dict
import sys



device = "cuda" # jt.device("cuda" if jt.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = jt.int32(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


@jt.no_grad()
def export_mesh(args):

    ckpt = jt.load(args.ckpt)
    kwargs = ckpt['kwargs']
    bg_freq =None
    if 'bg_freq' in kwargs:
        bg_D = kwargs.pop('bg_D')
        bg_freq = kwargs.pop('bg_freq')
        radii = kwargs.pop('radii')
        bg_view_freq = kwargs.pop('bg_view_freq')
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    if bg_freq is not None:
        tensorf.set_nerfplusplus(bg_freq,bg_view_freq,bg_D,radii)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    # scarf 0.005即可 coffee为0.0005
    convert_sdf_samples_to_ply(alpha , f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb , level=0.0005)


@jt.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, \
        is_stack=True, bbox=args.bbox, near=args.near, far=args.far, white_bg=args.white_bkgd)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = jt.load(args.ckpt)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    bg_freq =None
    if 'bg_freq' in kwargs:
        bg_D = kwargs.pop('bg_D')
        bg_freq = kwargs.pop('bg_freq')
        radii = kwargs.pop('radii')
        bg_view_freq = kwargs.pop('bg_view_freq')
    tensorf = eval(args.model_name)(**kwargs)
    if bg_freq is not None:
        tensorf.set_nerfplusplus(bg_freq,bg_view_freq,bg_D,radii)
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True, bbox=args.bbox, near=args.near, far=args.far, white_bg=args.white_bkgd)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

def reconstruction(args):

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False, bbox=args.bbox, near=args.near, far=args.far, white_bg=args.white_bkgd)
    val_dataset = dataset(args.datadir, split='val', downsample=args.downsample_train, is_stack=True, bbox=args.bbox, near=args.near, far=args.far, white_bg=args.white_bkgd)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)



    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox, reso_list[0])
    aabb = train_dataset.scene_bbox
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))

    optimizer = None
    if args.ckpt is not None:
        ckpt = jt.load(args.ckpt)
        kwargs = ckpt['kwargs']
        # 添加优化器lr、step的ckpt
        if "global_step" in ckpt:
            global_step = ckpt['global_step']+1
        kwargs.update({'device':device})
        bg_freq = None
        if 'bg_freq' in kwargs:
            bg_D = kwargs.pop('bg_D')
            bg_freq = kwargs.pop('bg_freq')
            radii = kwargs.pop('radii')
            bg_view_freq = kwargs.pop('bg_view_freq')
        tensorf = eval(args.model_name)(**kwargs)
        if bg_freq is not None:
            tensorf.set_nerfplusplus(bg_freq,bg_view_freq,bg_D,radii)
        tensorf.load(ckpt)
        nSamples = min(args.nSamples, cal_n_samples(kwargs['gridSize'],args.step_ratio))
    else:
        global_step = 0
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct,
                    
                    )
        tensorf.set_nerfplusplus(bg_freq=args.bg_freq, bg_view_freq=args.bg_view_freq, bg_D=args.bg_D, radii=args.radii)


    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = jt.optim.Adam(grad_vars, lr=0.001, betas=(0.9,0.99))
    if args.ckpt is not None:
        if "lr" in ckpt:
            lr_group = ckpt['lr']
            for i,param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = lr_group[i]

    if upsamp_list is None:
        upsamp_list = []
    if update_AlphaMask_list is None:
        update_AlphaMask_list = []
    #linear in logrithmic space
    N_voxel_list = (jt.round(jt.exp(jt.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]


    # jt.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")
    normal_vector_penalty_weight = args.normal_vector_penalty_weight

    pbar = tqdm(range(global_step,args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:
        optimizer.zero_grad()
        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx]

        #rgb_map, alphas_map, depth_map, weights, uncertainty
        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, tensorf, chunk=args.batch_size,
                                N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

        loss = jt.mean((rgb_map - rgb_train) ** 2)


        # loss
        total_loss = loss
        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight*loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.item(), global_step=iteration)
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight*loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.item(), global_step=iteration)

        if TV_weight_density>0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.item(), global_step=iteration)
        if TV_weight_app>0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.item(), global_step=iteration)

        if normal_vector_penalty_weight > 0:
            loss_norm = tensorf.penalty
            total_loss += normal_vector_penalty_weight*loss_norm
            summary_writer.add_scalar('train/normal_vector', loss_norm.item(), global_step=iteration)
            tensorf.penalty = jt.Var(0)

        # optimizer.zero_grad()
        optimizer.backward(total_loss)
        optimizer.step()

        loss = loss.item()
        
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)


        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []


        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test = evaluation(val_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)

            jt.clean_graph()
            jt.gc()


        if iteration in update_AlphaMask_list:

            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)


            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays,allrgbs = tensorf.filtering_rays(allrays,allrgbs)
                trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)
            
            jt.clean_graph()
            jt.sync_all()
            jt.gc()


        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            optimizer = jt.optim.Adam(grad_vars, lr=0.001, betas=(0.9, 0.99))
            jt.clean_graph()
            jt.sync_all()
            jt.gc()

        if iteration%args.gc_every == 0: 
            # jt.sync_all()
            jt.gc()

        if (iteration)%(5 * args.vis_every) == 0 and iteration > 0:
            lr_group = []
            for param_group in optimizer.param_groups:
                lr_group.append(param_group['lr'])
            global_kwargs = {
                "lr": lr_group,
                "global_step": iteration,
            }
            tensorf.save(f'{logfolder}/{args.expname}{iteration}.th', global_kwargs)
            if args.render_test:
                os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
                test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True, bbox=args.bbox, near=args.near, far=args.far, white_bg=args.white_bkgd)
                PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                        N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
                summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
                print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    lr_group = []
    for param_group in optimizer.param_groups:
        lr_group.append(param_group['lr'])
    global_kwargs = {
        "lr": lr_group,
        "global_step": iteration,
    }
    tensorf.save(f'{logfolder}/{args.expname}.th', global_kwargs)


    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True, bbox=args.bbox, near=args.near, far=args.far, white_bg=args.white_bkgd)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True, bbox=args.bbox, near=args.near, far=args.far, white_bg=args.white_bkgd)
        PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = val_dataset.render_path
        # c2ws = val_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(val_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


if __name__ == '__main__':

    jt.flags.use_cuda = 1
    DEBUG = False

    os.environ['debug'] = '0'
    os.environ['gdb_attach'] = '0'

    # jt.set_default_dtype(jt.float32)
    jt.set_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    if  args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)

