import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x * torch.tanh(F.softplus(x))

class BasicConv(nn.Module):
    def __init__(self,in_channels,out_channels,k_size,stride=1):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,k_size,stride,k_size//2,bias=False)# 无bias;这里padding要么是3核1，1核0;除了下采样stride都是1
        self.bn=nn.BatchNorm2d(out_channels)
        self.mish=Mish()
    def forward(self,x):
        return self.mish(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self,channels,hidden_channels=None):
        super().__init__()
        if hidden_channels==None:
            hidden_channels=channels
        self.block=nn.Sequential(BasicConv(channels,hidden_channels,1),BasicConv(hidden_channels,channels,3))
    def forward(self,x):
        return self.block(x) + x

class CSPBlock(nn.Module):
    def __init__(self,in_channels,num_block,first=False):
        super().__init__()
        self.downsample_conv=BasicConv(in_channels,in_channels*2,3,stride=2)# 进行下采样
        para=2 if first else 1
        self.split_conv0=BasicConv(in_channels*2,in_channels*para,1)
        self.split_conv1=BasicConv(in_channels*2,in_channels*para,1)
        self.blocks_conv=nn.Sequential(
            *[ResBlock(in_channels*para,in_channels) for _ in range(num_block)],
            BasicConv(in_channels*para,in_channels*para,1)
        )
        self.concat_conv=BasicConv(in_channels*2*para,in_channels*2,1)
    def forward(self,x):
        x=self.downsample_conv(x)
        return self.concat_conv(torch.cat([self.blocks_conv(self.split_conv1(x)),self.split_conv0(x)],dim=1))# cat的顺序很重要

class DarkNet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=BasicConv(3,32,3)
        self.stages= nn.ModuleList([
            CSPBlock(32,1,True),
            CSPBlock(64,2),
            CSPBlock(128,8),
            CSPBlock(256,8),
            CSPBlock(512,4)
        ])
    def forward(self,x):
        out1=self.stages[2](self.stages[1](self.stages[0](self.conv1(x))))
        out2=self.stages[3](out1)
        out3=self.stages[4](out2)
        return out1,out2,out3
    
if __name__=="__main__":
    # 将权重赋给CSPDarknet53
    model_path=r"F:\1_project_store\2_dev_info\3_data_set\1_cv\7_car_pedestrain_detection\yolo4_weights_my.pth"
    def load_model_pth(model, pth):
        print('Loading weights into state dict, name: %s' % pth)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pth, map_location=device)
        
        matched_dict = {}
        # 遍历预训练权重中的键名
        for k_pretrain, v in pretrained_dict.items():
            # 只处理 backbone 部分的权重（去掉 'backbone.' 前缀）
            if k_pretrain.startswith('backbone.'):
                k_cur = k_pretrain[9:]   # 去掉 'backbone.' 前缀
                # 检查当前模型中是否存在该键，且形状一致
                if k_cur in model_dict and model_dict[k_cur].shape == v.shape:
                    matched_dict[k_cur] = v
                else:
                    print(f'Skip unmatched layer: {k_pretrain}')
            else:
                # 非 backbone 部分（如头部）在当前模型中不存在，直接跳过
                print(f'Skip non-backbone layer: {k_pretrain}')
        
        print(f'Total loaded keys in pretrained: {len(pretrained_dict)}')
        print(f'Matched keys: {len(matched_dict)}')
        print(f'Missed keys in current model: {len(model_dict) - len(matched_dict)}')
        
        # 更新模型权重（不要求完全匹配）
        model_dict.update(matched_dict)
        model.load_state_dict(model_dict, strict=False)
        print('Finished!')
        return model
    model=DarkNet53()
    model=load_model_pth(model,model_path)