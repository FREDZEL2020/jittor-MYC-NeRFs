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
    log.title("[{}] (PyTorch code for training NeRF/BARF)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    options.save_options_file(opt)

    model = importlib.import_module("model.{}".format(opt.model))
    m = model.Model(opt)

    m.load_dataset(opt)
    m.build_networks(opt)
    m.setup_optimizer(opt)
    m.restore_checkpoint(opt)
    m.setup_visualizer(opt)

    m.train(opt)

if __name__=="__main__":
    main()
