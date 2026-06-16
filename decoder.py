from yolobody import YoloBody
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

# (batch_num,anchor_num,H,W,output)
def yolo_decode(output, num_classes, anchors, num_anchors, scale_x_y):
    device = None
    cuda_check = output.is_cuda
    if cuda_check:device = output.get_device()
    n_ch = 5+num_classes
    A,B,H,W = num_anchors,output.size(0),output.size(2),output.size(3)
    output = output.view(B, A, n_ch, H, W).permute(0,1,3,4,2).contiguous()
    # sigmoid将连续值映射到0，1间；exp映射为大于1的倍数；
    bx = torch.sigmoid(output[..., 0])
    by = torch.sigmoid(output[..., 1])
    grid_x = torch.arange(W).repeat(1, 3, W, 1).to(device)
    grid_y = torch.arange(H).repeat(1, 3, H, 1).permute(0, 1, 3, 2).to(device)
    bx += grid_x
    by += grid_y
    bw = torch.exp(output[..., 2])*scale_x_y - 0.5*(scale_x_y-1)
    bh = torch.exp(output[..., 3])*scale_x_y - 0.5*(scale_x_y-1)
    for i in range(num_anchors):
        bw[:, i, :, :] *= anchors[i*2]
        bh[:, i, :, :] *= anchors[i*2+1]
    bx = (bx / W).unsqueeze(-1)
    by = (by / H).unsqueeze(-1)
    bw = (bw / W).unsqueeze(-1)
    bh = (bh / H).unsqueeze(-1)
    boxes = torch.cat((bx, by, bw, bh), dim=-1).reshape(B, A * H * W, 4)
    det_confs = torch.sigmoid(output[..., 4]).unsqueeze(-1).reshape(B, A*H*W, 1)
    cls_confs = torch.sigmoid(output[..., 5:]).reshape(B, A*H*W, num_classes)
    outputs = torch.cat([boxes, det_confs, cls_confs], dim=-1)
    return outputs

class YoloLayer(nn.Module):
    def __init__(self, img_size, anchor_masks=[], num_classes=80, anchors=[], num_anchors=9, scale_x_y=1):
        super().__init__()
        self.anchor_masks = anchor_masks
        self.num_classes = num_classes
        self.anchors = anchors
        self.anchor_step = len(self.anchors) // num_anchors
        self.scale_x_y = scale_x_y
        self.feature_length = [img_size[0]//8,img_size[0]//16,img_size[0]//32]# 608的图分别为76，38，19
        self.img_size = img_size
    def forward(self, output):
        in_w = output.size(3)
        anchor_index = self.anchor_masks[self.feature_length.index(in_w)]
        stride_w = self.img_size[0] / in_w
        masked_anchors = []
        for m in anchor_index:
            masked_anchors = masked_anchors + [a for a in self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]]
        self.masked_anchors = [anchor / stride_w for anchor in masked_anchors]

        data = yolo_decode(output, self.num_classes, self.masked_anchors, len(anchor_index),scale_x_y=self.scale_x_y)
        return data