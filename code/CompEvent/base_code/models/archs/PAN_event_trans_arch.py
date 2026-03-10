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

        self.conv4 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)
        

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=3)

        self.event_conv = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.gwc_conv = nn.Conv2d(3,20,kernel_size=1, bias=False)

    def forward(self, event_feature, x):
        residual = x

        out_a= self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_event = self.event_conv(event_feature)
        #########################################################
        # 准备特征融合了
        #########################################################
        
        # 融合两种信息
        features_left = self.feature_extraction(event_feature)
        features_right = self.feature_extraction(x)
        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], 3,1)
        
        gwc_volume = gwc_volume.squeeze()
        if gwc_volume.shape[0] == 3:
            gwc_volume = gwc_volume.unsqueeze(dim=0) 
        #############################################################
        #############################################################
        # print('gwc_volume=',gwc_volume.shape)
        gwc_volume = self.gwc_conv(gwc_volume)
        out_a = self.k1(out_a)
        
        out_a = out_a + gwc_volume
        # out_event = out_event + gwc_volume
        out_event = self.conv4(torch.cat([out_event, gwc_volume], dim=1))
        
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual

        return out_event,out





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
#############################################################################
#correlation操作
#############################################################################
def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out



class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(40, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 3, 3, 1, 1, 1)

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(3, 40, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(40, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)


        gwc_feature = x


        concat_feature = self.lastconv(gwc_feature)
        return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}

    
#############################################################################
# concate event数据，查看初步结果
#############################################################################

class PAN_Event_trans(nn.Module):
        
    def __init__(self, in_nc, out_nc, nf, unf, nb, scale=4):
        super(PAN_Event_trans, self).__init__()
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
        self.event_first = nn.Conv2d(40, 40, 3, 1, 1, bias=True)
        self.event_2C_first = nn.Conv2d(2,40,3,1,1,bias=True)
        self.img_frist = nn.Conv2d(3,40,3,1,1,bias=True)
        self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=3)

        # layers = []
        # for _ in range(16):
            # layers.append(SCPA(nf=nf))
        # self.SCPA = nn.Sequential(*layers)

        self.SCPA_1 = SCPA(nf=nf)
        self.SCPA_2 = SCPA(nf=nf)
        self.SCPA_3 = SCPA(nf=nf)
        self.SCPA_4 = SCPA(nf=nf)
        self.SCPA_5 = SCPA(nf=nf)
        self.SCPA_6 = SCPA(nf=nf)
        self.SCPA_7 = SCPA(nf=nf)
        self.SCPA_8 = SCPA(nf=nf)


    def forward(self, x,aggregrated_event):
    # def forward(self, x):
        # x = [16,5,64,64]
        # print('x.shape=',x.shape)
        # print('aggregrated_event.shape=',aggregrated_event.shape)
        ###########################1###########################
        # event
        ######################################################
        event_2C = x[:,3:5,:,:]
        event_40C = x[:,5:,:,:]
        x = x[:,:3,:,:]
        x = self.img_frist(x)
        event_40C_feature = self.senet(event_40C)       # event_40C_feature=[2,40,128,128]
        event_40C_feature = self.channel_att(event_40C_feature,aggregrated_event)
        event_40C_feature = self.event_first(event_40C_feature)
        # event_2C_feature = self.event_2C_first(event_2C)
        event_2C_feature = event_40C_feature
        # [2,3,128,128]
        ###########################################################
        ###########################################################
        ###########################################################
        
        # # 融合两种信息
        # features_left = self.feature_extraction(event_2C_feature)
        # features_right = self.feature_extraction(x)
        # gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], 3,1)
        
        # gwc_volume = gwc_volume.squeeze()
        # [2,3,128,128]
        ######################################################################
        ######################################################################
        # 下面是原始的PAN部分
        ######################################################################

        # fea = self.conv_first(gwc_volume)
        # trunk = self.trunk_conv(self.SCPA_trunk(fea))
        # _,trunk = self.SCPA_trunk(event_2C_feature,x)

        event_2C_feature,x = self.SCPA_1(event_2C_feature,x)
        event_2C_feature,x = self.SCPA_2(event_2C_feature,x)
        event_2C_feature,x = self.SCPA_3(event_2C_feature,x)
        event_2C_feature,x = self.SCPA_4(event_2C_feature,x)
        event_2C_feature,x = self.SCPA_5(event_2C_feature,x)
        event_2C_feature,x = self.SCPA_6(event_2C_feature,x)
        event_2C_feature,x = self.SCPA_7(event_2C_feature,x)
        _,trunk = self.SCPA_8(event_2C_feature,x)
        trunk = self.trunk_conv(trunk)
        fea = x + trunk
        
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
    

