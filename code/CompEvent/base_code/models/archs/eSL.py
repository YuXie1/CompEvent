import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from math import sqrt
import torch.nn.functional as F
import numpy as np
import math
# from torchsummary import summary

################################################################################################3
################################################################################################
################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.5  # neuronal threshold
lens = 0.5/3  # hyper-parameters of approximate function
decay = 0.8  # decay constants
batch_size = 1


######################################################################################################
######################################################################################################
######################################################################################################

# class MySCN(nn.Module):
#     def __init__(self):
#         super(MySCN, self).__init__()
        
#         self.W1 = nn.Conv2d(40, 64, 3, 1, 1, bias=False)
#         self.S1 = nn.Conv2d(64, 40, 3, 1, 1, groups=1, bias=False)
#         self.S2 = nn.Conv2d(40, 64, 3, 1, 1, groups=1, bias=False)
#         self.shlu = nn.ReLU(True)


#     def forward(self, input):
#         x1 = input[:,range(0,40),:,:]
#         event_input = input[:,range(40,80),:,:]

#         x1 = torch.mul(x1,event_input)
#         # torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等
#         z = self.W1(x1)
#         tmp = z
#         for i in range(25):
#             ttmp = self.shlu(tmp)
#             x = self.S1(ttmp)
#             x = torch.mul(x,event_input)
#             x = torch.mul(x,event_input)
#             x = self.S2(x)
#             x=ttmp-x
#             tmp = torch.add(x, z)
#         c = self.shlu(tmp)
#         return c
class MySCN(nn.Module):
    def __init__(self):
        super(MySCN, self).__init__()
        self.W1 = nn.Conv2d(40, 128, 3, 1, 1, bias=False)
        self.S1 = nn.Conv2d(128, 40, 3, 1, 1, groups=1, bias=False)
        self.S2 = nn.Conv2d(40, 128, 3, 1, 1, groups=1, bias=False)
        self.shlu = nn.ReLU(True)


    def forward(self, input):
        x1 = input[:,range(0,40),:,:]
        event_input = input[:,range(40,80),:,:]

        x1 = torch.mul(x1,event_input)
        z = self.W1(x1)
        tmp = z
        for i in range(25):
            ttmp = self.shlu(tmp)
            x = self.S1(ttmp)
            x = torch.mul(x,event_input)
            x = torch.mul(x,event_input)
            x = self.S2(x)
            x=ttmp-x
            tmp = torch.add(x, z)
        c = self.shlu(tmp)
        return c

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.scn = nn.Sequential(MySCN())

#         self.image_d=nn.Conv2d(in_channels=1, out_channels=40, kernel_size=1, stride=1, padding=0, bias=False)
#         self.event_c1=nn.Conv2d(in_channels=40, out_channels=40, kernel_size=1, stride=1, padding=0, bias=False)
#         self.event_c2=nn.Conv2d(in_channels=40, out_channels=40, kernel_size=1, stride=1, padding=0, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.shu1=nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
#         self.shu2=nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
#         self.endconv=nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
#         self.ps1=nn.PixelShuffle(2)
#         self.ps2=nn.PixelShuffle(2)
    
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, sqrt(2. / n))

#     def forward(self, x): 
#         x1 = x[:,range(0,1),:,:]
#         # 图像数据仅仅占据1个通道
#         x1=self.image_d(x1)

#         event = x[:,range(1,41),:,:]
#         # 事件数据占据40个通道
#         event_out=self.event_c1(event)
#         event_out=torch.sigmoid(event_out)
#         event_out=self.event_c2(event_out)
#         event_out=torch.sigmoid(event_out)

#         scn_input=torch.cat([x1,event_out],1)
        
#         out = self.scn(scn_input)
#         out=self.shu1(out)
#         out=self.ps1(out)
#         out=self.shu2(out)
#         out=self.ps2(out)
#         out=self.endconv(out)
#         return out


class eSL_Net(nn.Module):
    def __init__(self,scale=4):
        super(eSL_Net, self).__init__()
        self.scale = scale
        self.scn = nn.Sequential(MySCN())
        self.image_d=nn.Conv2d(in_channels=3, out_channels=40, kernel_size=1, stride=1, padding=0, bias=False)
        self.event_c1=nn.Conv2d(in_channels=40, out_channels=40, kernel_size=1, stride=1, padding=0, bias=False)
        self.event_c2=nn.Conv2d(in_channels=40, out_channels=40, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.shu1=nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.shu2=nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.endconv=nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.endconv2 = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        # 
        self.ps1=nn.PixelShuffle(2)
        self.ps2=nn.PixelShuffle(2)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x,aggregrated_event): 
        
        # event_2C = x[:,3:5,:,:]
        event = x[:,5:,:,:]
        x1 = x[:,:3,:,:]
        
        # x1 = x[:,range(0,1),:,:]
        x1=self.image_d(x1)

        # event = x[:,range(1,41),:,:]
        event_out=self.event_c1(event)
        event_out=torch.sigmoid(event_out)
        event_out=self.event_c2(event_out)
        event_out=torch.sigmoid(event_out)

        scn_input=torch.cat([x1,event_out],1)
        
        out = self.scn(scn_input)
        out=self.shu1(out)
        out=self.ps1(out)
        out=self.shu2(out)
        if self.scale == 4:
            out=self.ps2(out)
            out=self.endconv(out)
        elif self.scale == 2:
            out = self.endconv2(out)
        return out
# your_model = Net()
# summary(your_model, input_size=(200, 41, 28,28))