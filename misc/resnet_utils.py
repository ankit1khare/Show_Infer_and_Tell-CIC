#resnet applied to images directly to get fc_feats and att_feats

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class myResnet(nn.Module):
    def __init__(self, resnet):
#         pdb.set_trace()
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=10):
#         pdb.set_trace()
        x = img.unsqueeze(0)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        pdb.set_trace()
        fc = x.mean(3).mean(2).squeeze()
        att = F.adaptive_avg_pool2d(x,[att_size,att_size]).squeeze().permute(1, 2, 0)
        
        return fc, att

