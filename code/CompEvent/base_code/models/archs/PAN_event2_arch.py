import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
from models.archs.se_resnet import SEBottleneck


class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out
    
class PAConv(nn.Module):

    def __init__(self, nf, k_size=3):

        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution

    def forward(self, x):

        y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out
        
class SCPA(nn.Module):
    
    """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
        Github: https://github.com/MCG-NKU/SCNet
    """

    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(SCPA, self).__init__()
        group_width = nf // reduction
        
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        
        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        bias=False)
                    )
        
        self.PAConv = PAConv(group_width)
        
        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x

        out_a= self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual

        return out


class Channel_Att(nn.Module):
    def __init__(self, reduction=16):
        super(Channel_Att, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 该情况相当于torch.mean(input)，所有的数累加起来求一个均值
        
        self.fc = nn.Sequential(
            nn.Linear(512, 40, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(40, 40, bias=False),
            nn.Sigmoid()
        )

    def forward(self, event_40C_feature,aggregrated_event):   # x=[2,40,128,128]
        
        _,event_c,_,_ = event_40C_feature.size()
        aggregrated_event = aggregrated_event[0]
        aggregrated_event = aggregrated_event.transpose(1,2)
        aggregrated_event = aggregrated_event.unsqueeze(3)
        b, c,_, _ = aggregrated_event.size()
        aggregrated_event = self.avg_pool(aggregrated_event)    # y=[2,40,1,1]
        aggregrated_event = aggregrated_event.view(b, c)
        
        
        channel_attention = self.fc(aggregrated_event)

        channel_attention = channel_attention.view(b,event_c,1,1)  # channel_attention = [2,40]    
        
        event_40C_feature = event_40C_feature * channel_attention.expand_as(event_40C_feature)
        

        return event_40C_feature
    
    
#############################################################################
# concate event数据，查看初步结果
#############################################################################

class PAN_Event_2(nn.Module):
        
    def __init__(self, in_nc, out_nc, nf, unf, nb, scale=4):
        super(PAN_Event_2, self).__init__()
        # SCPA
        SCPA_block_f = functools.partial(SCPA, nf=nf, reduction=2)
        self.scale = scale
        
        # inplances=40，表示事件数据通道数
        self.senet = SEBottleneck(inplanes=40,planes=40)
        
        ### first convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        ### main blocks
        self.SCPA_trunk = arch_util.make_layer(SCPA_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
        
        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            
        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        ########################################################################
        ########################################################################
        ########################################################################
        self.channel_att = Channel_Att()
        self.event_first = nn.Conv2d(40, 3, 3, 1, 1, bias=True)

    def forward(self, x,aggregrated_event):
        # x = [16,5,64,64]
        ######################################################
        # event
        ######################################################
        event_2C = x[:,3:5,:,:]
        event_40C = x[:,5:,:,:]
        x = x[:,:3,:,:]
        
        event_40C_feature = self.senet(event_40C)       # event_40C_feature=[2,40,128,128]
        # event_40C_feature = self.channel_att(event_40C_feature,aggregrated_event)
        event_40C_feature = self.event_first(event_40C_feature)
        ###########################################################
        ###########################################################
        ###########################################################
        model_input = torch.cat([event_40C_feature,x],axis=1)
        
        fea = self.conv_first(model_input)
        trunk = self.trunk_conv(self.SCPA_trunk(fea))
        fea = fea + trunk
        
        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(F.interpolate(fea, scale_factor=self.scale, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))
        
        out = self.conv_last(fea)
        
        ILR = F.interpolate(x[:,:3], scale_factor=self.scale, mode='bilinear', align_corners=False)
        out = out + ILR
        return out