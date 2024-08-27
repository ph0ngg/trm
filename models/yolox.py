#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
import numpy as np
import torch
from .matching import *
from .losses import *

from models.utils.boxes import postprocess



class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        #print(fpn_outs[2].shape)
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
            return outputs
        else:
            outputs = self.head(fpn_outs)
            yolo_outputs, reid_idx = postprocess(outputs, 80)
            #anh 1 ---> id cua anh 1 ---> loss
            #output co dang la (batch, so object, 7)
            
            return fpn_outs, yolo_outputs, reid_idx

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)
class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x 
        return output 

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output
    
class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x

#thong tin dau ra cua model yolo la xin tu backbone
#va outputs du doan cac nguoi co trong anh
#can lam: tu outputs matching voi targets de lay id, sau do tinh loss


class Head2(nn.Module):
    def __init__(self):
        super().__init__()
        self.nids = 1638
        self.embed_length = 128
        self.neck2 = nn.Conv2d(1024, self.embed_length, 1, 1)
        self.neck1 = nn.Conv2d(512, self.embed_length, 1, 1)
        self.neck0 = nn.Conv2d(256, self.embed_length, 1, 1)
        self.cbam1 = CBAM(self.embed_length, r=4)
        self.cbam2 = CBAM(self.embed_length, r=4)
        self.cbam0 = CBAM(self.embed_length, r=4)
        self.linear = nn.Linear(self.embed_length, self.nids)
        self.type_loss = 'triplet_and_ce'

    def forward(self, imgs, xin, yolo_outputs, reid_idx,  targets = None):
        neck_feat_0 = self.cbam0(self.neck0(xin[0]))
        neck_feat_1 = self.cbam1(self.neck1(xin[1]))
        neck_feat_2 = self.cbam2(self.neck2(xin[2]))

        b0, c0, h0, w0 = neck_feat_0.shape
        b1, c1, h1, w1 = neck_feat_1.shape
        b2, c2, h2, w2 = neck_feat_2.shape

        reid_feat = torch.cat((neck_feat_0.view(b0, c0, h0*w0), neck_feat_1.view(b1, c1, h1*w1), neck_feat_2.view(b2, c2, h2*w2)), 2)
        if self.training:
            return self.get_loss(yolo_outputs, targets, reid_feat, reid_idx)           
        else:
            return self.get_emb_vector(yolo_outputs, reid_feat, reid_idx)
    def get_loss(self, yolo_outputs, targets, reid_feat, reid_idx):
        #gt_ids =  targets[:, :, 1]
        total_emb_preds = []
        id_targets = []
        for batch_idx in range(len(yolo_outputs)):
            #print(yolo_outputs[batch_idx])
            matched_gt_inds = matching(yolo_outputs[batch_idx], targets[batch_idx])[1]
            #print('match_gt_inds', matched_gt_inds)
            #match_gt_inds: cai box du doan ung voi cai box thuc te thu bao nhieu
            gt_ids = targets[batch_idx, :, 1]

            id_target = gt_ids[matched_gt_inds] # id cua nguoi dua vao


            emb_preds = reid_feat[batch_idx, :, reid_idx[batch_idx][:len(id_target)]]
            id_targets.extend(id_target)

            for i in range(emb_preds.shape[1]):
                total_emb_preds.append(emb_preds[:, i])


        if (self.type_loss == 'triplet'):
            loss = triplet_loss(torch.stack(total_emb_preds), torch.stack(id_targets))
            return loss
        elif self.type_loss == 'ce':
            #total_emb_preds: n_people, 128
            #id_target: n_people
            emb_vectors = torch.stack(total_emb_preds) #n_peo, 128
            pred_class_output = self.linear(emb_vectors)
            #print(f'pred_class_output: {pred_class_output.shape}, id_target: {torch.stack(new_id_targets).shape}')
            loss = cross_entropy_loss(pred_class_output, torch.stack(id_targets).long())
            return loss    
        else:
            emb_vectors = torch.stack(total_emb_preds) #n_peo, 128
            pred_class_output = self.linear(emb_vectors)
            return 0.5*triplet_loss(torch.stack(total_emb_preds), torch.stack(id_targets)) + 0.5 * cross_entropy_loss(pred_class_output, torch.stack(id_targets).long())

    def get_emb_vector(self, yolo_outputs, reid_feat, reid_idx):
        total_emb_preds = []
        #reid_feat: B, n, 128
        for batch_idx in range(len(yolo_outputs)):
            emb_preds = reid_feat[batch_idx, :, reid_idx[batch_idx]]
            for i in range(emb_preds.shape[1]):
                total_emb_preds.append(emb_preds[:, i])
        return yolo_outputs, total_emb_preds
#test_matching

#training: model1(img) --> output (b, n_object, 7), vi tri reid_idx
