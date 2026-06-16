import torch
import torch.nn as nn
from collections import OrderedDict,defaultdict
from backbone import *

# yolov4的neck,返回网络的head

def conv2d(in_channels,out_channels,k_size,stride=1):
    return nn.Sequential(OrderedDict([
        ('conv',nn.Conv2d(in_channels,out_channels,k_size,stride,k_size//2,bias=False)),
        ('bn',nn.BatchNorm2d(out_channels)),
        ('relu',nn.LeakyReLU(0.1))
    ]))

def make_three_conv(filter_list,in_channels):
    return nn.Sequential(
        conv2d(in_channels,filter_list[0],1),# 1*1卷积特征压缩
        conv2d(filter_list[0],filter_list[1],3),
        conv2d(filter_list[1],filter_list[2],1)
    )

def make_five_conv(filter_list,in_channels):
    return nn.Sequential(
        conv2d(in_channels,filter_list[0],1),
        conv2d(filter_list[0],filter_list[1],3),
        conv2d(filter_list[1],filter_list[0],1),
        conv2d(filter_list[0],filter_list[1],3),
        conv2d(filter_list[1],filter_list[0],1)
    )

def yolo_head(in_channels,out_channels):
    return nn.Sequential(
        conv2d(in_channels,in_channels*2,3),
        nn.Conv2d(in_channels*2,out_channels,1)
    )

# 特征融合技巧
class SpatialPyramidPooling(nn.Module):
    def __init__(self,pool_sizes=[5,9,13]):
        super().__init__()
        self.maxpool=nn.ModuleList([
            nn.MaxPool2d(pool_size,1,(pool_size-1)//2) for pool_size in pool_sizes
        ])
    def forward(self,x):
        features = [maxpool(x) for maxpool in self.maxpool[::-1]]
        return torch.cat(features+[x],dim=1)

class Upsample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.upsample=nn.Sequential(
            conv2d(in_channels,out_channels,1),
            nn.Upsample(scale_factor=2)
        )
    def forward(self,x):
        return self.upsample(x)

class YoloBody(nn.Module):
    def __init__(self,num_anchors,num_classes):
        super().__init__()
        self.backbone=DarkNet53()
        self.conv1=make_three_conv([512,1024,512],1024)
        self.SPP=SpatialPyramidPooling()
        self.conv2=make_three_conv([512,1024,512],2048)
        self.upsample1=Upsample(512,256)
        self.upsample2=Upsample(256,128)
        self.make_five_conv1=make_five_conv([256,512],512)
        self.make_five_conv2=make_five_conv([128,256],256)
        self.make_five_conv3=make_five_conv([256,512],512)
        self.make_five_conv4=make_five_conv([512,1024],1024)
        self.conv_for_P4 = conv2d(512,256,1)
        self.conv_for_P3 = conv2d(256,128,1)
        self.down_sample1 = conv2d(128,256,3,stride=2)
        self.down_sample2 = conv2d(256,512,3,stride=2)
        self.yolo_head3=yolo_head(128,num_anchors * (5 + num_classes))# 正常情况都是123，为了对齐参数做一次特例
        self.yolo_head2=yolo_head(256,num_anchors * (5 + num_classes))
        self.yolo_head1=yolo_head(512,num_anchors * (5 + num_classes))
    def forward(self,x):
        b1,b2,b3=self.backbone(x)
        spp=self.conv2(self.SPP(self.conv1(b3)))
        conv51=self.make_five_conv1(torch.cat([self.conv_for_P4(b2),self.upsample1(spp)],dim=1))
        conv52=self.make_five_conv2(torch.cat([self.conv_for_P3(b1),self.upsample2(conv51)],dim=1))
        conv53=self.make_five_conv3(torch.cat([self.down_sample1(conv52),conv51],dim=1))
        out1=self.yolo_head3(conv52)
        out2=self.yolo_head2(conv53)
        out3=self.yolo_head1(self.make_five_conv4(torch.cat([self.down_sample2(conv53),spp],dim=1)))
        return out1,out2,out3

if __name__ == '__main__':
    def load_model_pth(model, pth):
        print('Loading weights into state dict, name: %s'%(pth))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict=model.state_dict()
        pretrained_dict = torch.load(pth, map_location=device)
        matched_dict = {}
        for k,v in pretrained_dict.items():
            if k not in model_dict:
                print("not found:{}".format(k))
                continue
            if np.shape(model_dict[k]) == np.shape(v):
                matched_dict[k] = v
            else:
                print('un matched layers: %s'%k)
        print('%d layers matched,  %d layers miss'%(len(matched_dict.keys()), len(model_dict)-len(matched_dict.keys())))
        model_dict.update(matched_dict)
        model.load_state_dict(model_dict)
        print('Finished!')
        return model
    model = YoloBody(3, 80)
    load_model_pth(model, r"F:\1_project_store\2_dev_info\3_data_set\1_cv\7_car_pedestrain_detection\yolo4_weights_my.pth")