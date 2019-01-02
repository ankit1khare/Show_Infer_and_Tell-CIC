from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import models 
from dataloader_server import *
import eval_utils
import argparse
import misc.utils as utils
import torch
import pdb

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--device_num', type=int, default=0,
                    help='gpu id')
parser.add_argument('--model', type=str, default='save/model.pth',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
parser.add_argument('--caption_model', type=str, default="topdown_pa",
                    help=', topdown')
parser.add_argument('--seq_per_img', type=int, default=1,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
parser.add_argument('--rnn_size', type=int, default=1000,
                help='size of the rnn in number of hidden nodes in each layer')
# *********************
parser.add_argument('--num_layers', type=int, default=1,
                help='number of layers in the RNN')
parser.add_argument('--rnn_type', type=str, default='lstm',
                help='rnn, gru, or lstm')
parser.add_argument('--input_encoding_size', type=int, default=1000,
                help='the encoding size of each token in the vocabulary, and the image.')
parser.add_argument('--att_hid_size', type=int, default=512,
                help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
parser.add_argument('--fc_feat_size', type=int, default=2048,
                help='2048 for resnet, 4096 for vgg')
parser.add_argument('--att_feat_size', type=int, default=2048,
                help='2048 for resnet, 512 for vgg')
parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                help='strength of dropout in the Language Model RNN')
parser.add_argument('--use_bn', type=int, default=0,
                    help='If 1, then do batch_normalization first in att_embed, if 2 then do bn both in the beginning and the end of att_embed')
parser.add_argument('--logit_layers', type=int, default=1,
                    help='number of layers in the RNN')

# Basic options
parser.add_argument('--batch_size', type=int, default=10,
                help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--dump_images', type=int, default=0,
                help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--beam_size', type=int, default=1,
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--temperature', type=float, default=1.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='val2014', 
                help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='images/', 
                help='In case the image paths have to be preprended with a root path to an image folder')


# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_fc_dir', type=str, default='data/_fc',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_att_dir', type=str, default='data/_att',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_label_h5', type=str, default='data/cocotalk_label.h5',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='data/cocotalk.json', 
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--input_box_dir', type=str, default='data/_box',
                    help='gpu id')
parser.add_argument('--train_only', type=str, default=1,
                    help='karpathy split / coco split')
parser.add_argument('--split', type=str, default='test', 
                help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='data/image_info_test2014.json', 
                help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
# misc
parser.add_argument('--id', type=str, default='ultra', 
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
parser.add_argument('--vocab_size', type=int, default=0, 
                help='vocab size')
parser.add_argument('--seq_length', type=int, default = 0, help='seq length')


opt = parser.parse_args()


# vocab = infos['vocab'] # ix -> word mapping
with open(os.path.join(opt.infos_path)) as f:
    infos = cPickle.load(f)
    saved_model_opt = infos['opt']
need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"] 
for checkme in need_be_same:
    assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme
        
loader = DataLoaderRaw(opt)
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
opt.vocab_size = 9487 #len(infos['vocab'])
opt.seq_length = 16 #infos['opt']['seq_length']
torch.cuda.set_device(opt.device_num) #AK
print(torch.cuda.current_device()) #AK

# Setup the model
model = models.setup(opt).cuda(opt.device_num) 
'''Dictionary load'''
model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
model.eval()

predictions = []
n=0

#start reading data and fetching from model
while True:
    start = time.time()
    data = loader.get_batch(opt.batch_size) #iterator state handled by data loader
    n = n + loader.batch_size
    print('Read data:', time.time() - start, '--' * 15)
    torch.cuda.synchronize()
    tmp = [data['fc_feats'], data['att_feats'], data['att_masks'] if data['att_masks'] is not None else None]
    tmp = [_ if _ is None else torch.from_numpy(_).cuda(opt.device_num) for _ in tmp]
    fc_feats, att_feats, att_masks = tmp
    
    with torch.no_grad():
        seq = model(fc_feats, att_feats, att_masks, opt={'beam_size': opt.beam_size, 'device_num':opt.device_num}, mode='sample')[0].data
    torch.cuda.synchronize()
    end = time.time()
    print("images done = {}, time/batch = {:.3f}".format(n, end - start))
    if opt.beam_size > 1:
        for i in range(loader.batch_size):
            print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]])) 
            print('--' * 15)
    sents = utils.decode_sequence(loader.get_vocab(), seq)
    for k, sent in enumerate(sents):
        entry = {'image_id': data['infos'][k]['id'], 'caption': str(sent)}
        print('image %s: %s' %(entry['image_id'], entry['caption']))
        predictions.append(entry)
    if n > data['bounds']['it_max']:
        break
for i in range(n - data['bounds']['it_max']):
    predictions.pop()
            
if opt.dump_json == 1:
    # dump the json
    json.dump(predictions, open('vis/test2014_trial.json', 'w'))
    print("Json file ready for server")
    
    