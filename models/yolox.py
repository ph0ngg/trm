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
from .network_blocks import OSNet, OSBlock

from models.utils.boxes import postprocess, cxcywh2xyxy



class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN(depth= 0.67, width= 0.75)
        if head is None:
            head = YOLOXHead(width= 0.75, num_classes= 80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        #print(fpn_outs[0].shape)
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

import torchvision.ops as ops


def change_box_type(box):
    x, y, w, h = box[:]
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return torch.tensor([x1, y1, x2, y2])

class Head2(nn.Module):
    def __init__(self):
        super().__init__()
        self.nids = 1638
        self.embed_length = 512
        # self.neck2 = nn.Conv2d(1024, self.embed_length, 1, 1)
        # self.neck1 = nn.Conv2d(512, self.embed_length, 1, 1)
        self.neck0 = nn.Conv2d(192, 192, 1, 1)
        # self.cbam1 = CBAM(self.embed_length, r=4)
        # self.cbam2 = CBAM(self.embed_length, r=4)
        self.cbam0 = CBAM(192, r=4)

        self.os_block_3 = OSBlock(in_channels= 192, out_channels= 288)
        self.os_block_4 = OSBlock(in_channels= 288, out_channels= 384)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = self._construct_fc_layer(
            self.embed_length, 384, dropout_p=None
        )

        self.linear = nn.Linear(self.embed_length, self.nids)
        self.type_loss = 'triplet_and_ce'

    def forward(self, xin, height, width, targets = None):
        # neck_feat_0 = self.cbam0(self.neck0(xin[0])) # chi lay output tu dark3
        #targets.shape: B, n_peo, 6
        next_feat_0 = self.cbam0(xin[0])
        boxes = list(torch.unbind(targets, dim=0))
        out_boxes = []
        target_ids = targets.view(-1, 6)[:, 1]
        for box in boxes:
            box[:, 2] *= width
            box[:, 3] *= height
            box[:, 4] *= width
            box[:, 5] *= height
            out_boxes.append(cxcywh2xyxy(box[:, 2:6]))
        #print(out_boxes)
        people_feature_map = ops.roi_align(next_feat_0, out_boxes, output_size = (64, 32), spatial_scale = 0.125, sampling_ratio=2)
        #print(people_feature_map.shape)
        x = self.os_block_3(people_feature_map)
        x = self.os_block_4(x) #B, 384, H, W
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        y = self.linear(v)
        #print(target_ids.shape)
        #y: num_peo, num_ids

        if self.training:
            return self.get_loss(v, target_ids, y)           
        else:
            return v
    def get_loss(self, v, target_ids, y):
        if (self.type_loss == 'triplet'):
            loss = triplet_loss(v, target_ids)
            return loss
        elif self.type_loss == 'ce':
            #total_emb_preds: n_people, 128
            #id_target: n_people
            #print(f'pred_class_output: {pred_class_output.shape}, id_target: {torch.stack(new_id_targets).shape}')
            loss = cross_entropy_loss(y, target_ids.long())
            return loss    
        else:

            return 0.5*triplet_loss(v, target_ids) + 0.5 * cross_entropy_loss(y, target_ids.long())
        
    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

        
#test_matching

#training: model1(img) --> output (b, n_object, 7), vi tri reid_idx
