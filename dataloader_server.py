from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
import torch
# import skimage
# import skimage.io
# import scipy.misc
import pdb

class DataLoaderRaw():
    
    def __init__(self, opt):
        self.opt = opt
        self.json_path = getattr(opt, 'coco_json', '')
        self.batch_size = getattr(opt, 'batch_size', 10)
        self.seq_per_img = 1
        self.vocab_size = getattr(opt, 'vocab_size', 0)
        self.coco_json = json.load(open(os.path.join(self.json_path)))
        self.cocotlk = json.load(open(os.path.join(opt.input_json)))
        self.ix_to_word = self.cocotlk['ix_to_word']
        
        #opts for feats 
        self.use_att = getattr(opt, 'use_att', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)
        
       
            
#         self.cnn_model = opt.get('cnn_model', 'resnet101')
#         self.my_resnet = getattr(misc.resnet, self.cnn_model)()
#         self.my_resnet.load_state_dict(torch.load('./data/imagenet_weights/'+self.cnn_model+'.pth'))
#         self.my_resnet = myResnet(self.my_resnet)
        
        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir
        self.input_box_dir = self.opt.input_box_dir
        
        self.files = []
        self.ids = []

#         print(len(self.coco_json))
        #loading image details from coco test file
        if len(self.coco_json) > 0:
            print('reading from ' + opt.coco_json)         
            for k,v in enumerate(self.coco_json['images']):
                fullpath = os.path.join(v['file_name'])
                self.files.append(fullpath)
                self.ids.append(v['id'])
        else:
            # read in all the filenames from the folder
            print('listing all images in directory ' + self.folder_path)
            def isImage(f):
                supportedExt = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.ppm','.PPM']
                for ext in supportedExt:
                    start_idx = f.rfind(ext)
                    if start_idx >= 0 and start_idx + len(ext) == len(f):
                        return True
                return False

            #Getting all the files (sequence doesn't matter)
            n = 1
            for root, dirs, files in os.walk(self.folder_path, topdown=False):
                for file in files:
                    fullpath = os.path.join(self.folder_path, file)
                    if isImage(fullpath):
                        self.files.append(fullpath)
                        self.ids.append(str(n)) # just order them sequentially wrt json not img
                        n = n + 1

        self.N = len(self.files)
        print('DataLoaderRaw found ', self.N, ' images')

        self.iterator = 0
#init ends

    #get feats one by one for batch
    def get_feats(self, index):
        """This function returns features corresponding to given image indexes
        """
        ix = index 
#         fc_feat = np.load(os.path.join(self.input_fc_dir, str(ix) + '.npy'))
        if self.use_att:
            att_feat = np.load(os.path.join(self.input_att_dir, str(ix) + '.npz'))['feat']
            # Reshape to K x C
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            if self.norm_att_feat:
                att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
            if self.use_box:
                box_feat = np.load(os.path.join(self.input_box_dir, str(ix) + '.npy'))
                # divided by image width and height
                x1,y1,x2,y2 = np.hsplit(box_feat, 4)
                h,w = self.coco_json[str(ix)]['height'], self.coco_json[str(ix)]['width']
                box_feat = np.hstack((x1/w, y1/h, x2/w, y2/h, (x2-x1)*(y2-y1)/(w*h)))
                if self.norm_box_feat:
                    box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
                att_feat = np.hstack([att_feat, box_feat])
                # sort the features by the size of boxes
                att_feat = np.stack(sorted(att_feat, key=lambda x:x[-1], reverse=True))
        else:
            att_feat = np.zeros((1,1,1))
        return (np.load(os.path.join(self.input_fc_dir, str(ix) + '.npy')), att_feat)    
    
    def get_batch(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        fc_batch = []
        att_batch = []
        max_index = self.N
        wrapped = False
        infos = []
        seq_per_img =1
#         pdb.set_trace()
        for i in range(batch_size):
            ri = self.iterator
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
                # wrap back around
            self.iterator = ri_next
#             pdb.set_trace()
            img = self.ids[ri] 
            info_struct = {}
            info_struct['id'] = self.coco_json['images'][ri]['id']
            infos.append(info_struct)
            
            tmp_fc, tmp_att = self.get_feats(img)

            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)
            
        data = {}
        data['fc_feats'] = np.stack(reduce(lambda x,y:x+y, [[_]*seq_per_img for _ in fc_batch]))
        # merge att_feats
        max_att_len = max([_.shape[0] for _ in att_batch])
        data['att_feats'] = np.zeros([len(att_batch)*seq_per_img, max_att_len, att_batch[0].shape[1]], dtype = 'float32')
        for i in range(len(att_batch)):
            data['att_feats'][i*seq_per_img:(i+1)*seq_per_img, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i*seq_per_img:(i+1)*seq_per_img, :att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None
            
            
        data['bounds'] = {'it_pos_now': self.iterator, 'it_max': self.N, 'wrapped': wrapped}
        data['infos'] = infos
        return data

    def reset_iterator(self, split):
        self.iterator = 0

    def get_vocab_size(self):
        return len(self.ix_to_word)

    def get_vocab(self):
        return self.ix_to_word
        