import numpy as np
import os,sys,time
import jittor as jt
import importlib

import options
from util import log

def main():
    jt.flags.use_cuda = 1
    DEBUG = False

    log.process(os.getpid())
    log.title("[{}] (PyTorch code for evaluating NeRF/BARF)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)

    model = importlib.import_module("model.{}".format(opt.model))
    m = model.Model(opt)

    m.load_dataset(opt,eval_split="val")
    m.build_networks(opt)

    # if opt.model=="barf" or opt.model=='garf':
    #     m.generate_videos_pose(opt)

    m.restore_checkpoint(opt)
    if opt.data.dataset in ["blender","llff"]:
        m.evaluate_full(opt)
    m.generate_videos_synthesis(opt)

if __name__=="__main__":
    main()
