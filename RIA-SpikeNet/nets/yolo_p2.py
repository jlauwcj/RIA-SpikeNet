#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.nn as nn

from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv

from .implicit import*
from .convnext import*


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [128,256, 512, 1024], act = "silu", depthwise = False,):
        super().__init__()
        Conv            = DWConv if depthwise else BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        #self.imA_1=nn.ModuleList()
        #self.imA_2=nn.ModuleList()

        #self.imM=nn.ModuleList()

        #self.imM_1=nn.ModuleList()
        #self.imM_2=nn.ModuleList()
        #self.imM_3=nn.ModuleList()
        for i in range(len(in_channels)):
            #---------------------------------------------
            #self.imA_1.append(ShiftChannel(out_channels = int(in_channels[i] * width)))
            #self.imM.append(ControlChannel(out_channels = int(num_classes+1+4)))

            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            #---------------------------------------------------------
            
            #self.imA_2.append(ShiftChannel(out_channels = int(256 * width)))
            #---------------------------------------------------------
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
             #---------------------------------------------------------
            #self.imM_1.append(ControlChannel(out_channels = int(num_classes)))
             #---------------------------------------------------------
            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act),
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
             #---------------------------------------------------------
            #self.imM_2.append(ControlChannel(out_channels = int(4)))
             #---------------------------------------------------------
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )
             #---------------------------------------------------------
            #self.imM_3.append(ControlChannel(out_channels = int(1)))
             #---------------------------------------------------------

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):

            #------------------------------------------------
            #x=self.imA_1[k](x)

            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)

            #------------------------------------------------
            
            #x=self.imA_2[k](x)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            #cls_output=self.imM_1[k](cls_output)
            #reg_output=self.imM_2[k](reg_output)
            #obj_output=self.imM_3[k](obj_output)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            #-------------------------------------
            #output=self.imM[k](output)

            outputs.append(output)
        return outputs

class YOLOPAFPN(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [128,256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        #self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.backbone=convnext_large()
        self.in_features    = in_features


        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[3] * width), int(in_channels[2] * width), 1, 1, act=act)
    
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        self.bu_conv2       = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)

        

        self.C3_p2=CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )
        self.bu_conv3     = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_n2 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        self.bu_conv1       = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        self.C3_n3= CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )
        self.bu_conv4=Conv(int(in_channels[2] * width), int(in_channels[2] * width), 3, 2, act=act)
        self.C3_n4=CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[3] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )
    def forward(self, input):
        #out_features            = self.backbone.forward(input)
        #[feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]
         
        feat,feat0,feat1,feat2,feat3,feat4=self.backbone(input)
        
       


        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.lateral_conv0(feat4)
        #-------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample) 
        #-------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4) 
        #-------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample    = self.C3_p3(P4_upsample)  
        P3=self.bu_conv2(P4_upsample)

        P3_upsample=self.upsample(P3)
        P3_upsample=torch.cat([P3_upsample,feat0],1)
        P2_out=self.C3_p2(P3_upsample)

        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        P2_downsample   = self.bu_conv3(P2_out) 
        #-------------------------------------------#
        #   40, 40, 256 + 40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P2_downsample   = torch.cat([P2_downsample, P3], 1) 
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P3_out          = self.C3_n2(P2_downsample) 

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        P3_downsample   = self.bu_conv1(P3_out)
        #-------------------------------------------#
        #   20, 20, 512 + 20, 20, 512 -> 20, 20, 1024
        #-------------------------------------------#
        P3_downsample   = torch.cat([P3_downsample, P4], 1)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        P4_out          = self.C3_n3(P3_downsample)

        P4_downsample=self.bu_conv4(P4_out)
        P4_downsample=torch.cat([P4_downsample,P5],1)
        P5_out=self.C3_n4(P4_downsample)

        return (P2_out,P3_out, P4_out, P5_out)

class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,'xl' : 1.33,}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,'xl' : 1.5,}
        depth, width    = depth_dict[phi], width_dict[phi]
        depthwise       = True if phi == 'nano' else False 

        self.backbone   = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.head       = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x):
        fpn_outs    = self.backbone.forward(x)
        outputs     = self.head.forward(fpn_outs)
        return outputs
