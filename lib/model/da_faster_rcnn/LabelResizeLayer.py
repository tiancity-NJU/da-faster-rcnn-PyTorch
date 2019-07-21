
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
import torch.nn as nn
from torch.autograd import Function
import cv2


class ImageLabelResizeLayer(nn.Module):
    """
    Resize label to be the same size with the samples
    """
    def __init__(self):
        super(ImageLabelResizeLayer, self).__init__()


    def forward(self,x,need_backprop):

        feats = x.detach().cpu().numpy()
        lbs = need_backprop.detach().cpu().numpy()
        gt_blob = np.zeros((lbs.shape[0], feats.shape[2], feats.shape[3], 1), dtype=np.float32)
        for i in range(lbs.shape[0]):
            lb=np.array([lbs[i]])
            lbs_resize = cv2.resize(lb, (feats.shape[3] ,feats.shape[2]),  interpolation=cv2.INTER_NEAREST)
            gt_blob[i, 0:lbs_resize.shape[0], 0:lbs_resize.shape[1], 0] = lbs_resize

        channel_swap = (0, 3, 1, 2)
        gt_blob = gt_blob.transpose(channel_swap)
        y=Variable(torch.from_numpy(gt_blob)).cuda()
        y=y.squeeze(1).long()
        return y


class InstanceLabelResizeLayer(nn.Module):


    def __init__(self):
        super(InstanceLabelResizeLayer, self).__init__()
        self.minibatch=256

    def forward(self, x,need_backprop):
        feats = x.data.cpu().numpy()
        lbs = need_backprop.data.cpu().numpy()
        resized_lbs = np.ones((feats.shape[0], 1), dtype=np.float32)
        for i in range(lbs.shape[0]):
            resized_lbs[i*self.minibatch:(i+1)*self.minibatch] = lbs[i]

        y=torch.from_numpy(resized_lbs).cuda()

        return y
