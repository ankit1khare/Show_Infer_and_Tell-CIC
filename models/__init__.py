from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import pdb

import numpy as np
import misc.utils as utils
import torch

from .AttModel import * 

def setup(opt):
    if opt.caption_model == 'topdown_pa':
        model = TopDownModel(opt) #pending change to include all files here and then send the type in opt
#     elif opt.caption_model == 'txtcon_512':
#         model = TxtCon512Model(opt)
    elif opt.caption_model == 'txtcon_1024': #later change to txtcon only
        model = TxtConModel(opt)
    elif opt.caption_model == 'lt_1024_rl':
        model = Lt1024RLModel(opt)
    elif opt.caption_model == 'variation_4':
        model = Variation4Model(opt)
    elif opt.caption_model == 'variation_5':
        model = ContextModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None and opt.caption_model != 'lt_1024_rl':
        # check if all necessary files exist 
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth'),  map_location="cuda:" + str(opt.device_num)))
#         model.to(opt.device_num)
    elif opt.caption_model == 'lt_1024_rl' and vars(opt).get('start_from', None) is not None:
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth'), map_location="cuda:" + str(opt.device_num)), False)
    return model
