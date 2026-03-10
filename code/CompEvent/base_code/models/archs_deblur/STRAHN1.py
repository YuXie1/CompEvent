import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import slayerSNN as snn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import random

netParams = snn.params('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/models/network.yaml')

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

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
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 1, 1, 1),
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




######################################################################
######################################################################
######################################################################

def getNeuronConfig(type: str='SRMALPHA',
                    theta: float=10.,
                    tauSr: float=1.,
                    tauRef: float=1.,
                    scaleRef: float=2.,
                    tauRho: float=0.3,  # Was set to 0.2 previously (e.g. for fullRes run)
                    scaleRho: float=1.):
    """
    :param type:     neuron type
    :param theta:    neuron threshold
    :param tauSr:    neuron time constant
    :param tauRef:   neuron refractory time constant
    :param scaleRef: neuron refractory response scaling (relative to theta)
    :param tauRho:   spike function derivative time constant (relative to theta)
    :param scaleRho: spike function derivative scale factor
    :return: dictionary
    """
    return {
        'type': type,
        'theta': theta,
        'tauSr': tauSr,
        'tauRef': tauRef,
        'scaleRef': scaleRef,
        'tauRho': tauRho,
        'scaleRho': scaleRho,
    }




class NetworkBasic(torch.nn.Module):
    # def __init__(self, netParams,
    #              theta=[30, 50, 100],
    #              tauSr=[1, 2, 4],
    #              tauRef=[1, 2, 4],
    #              scaleRef=[1, 1, 1],
    #              tauRho=[1, 1, 10],
    #              scaleRho=[10, 10, 100]):
    def __init__(self, netParams,
                 theta=[20, 40, 80],
                 tauSr=[1, 2, 4],
                 tauRef=[1, 2, 4],
                 scaleRef=[1, 1, 1],
                 tauRho=[1, 1, 10],
                 scaleRho=[10, 10, 100]):
        super(NetworkBasic, self).__init__()
        self.global_num = 0
        self.neuron_config = []
        self.neuron_config.append(
            getNeuronConfig(theta=theta[0], tauSr=tauSr[0], tauRef=tauRef[0], scaleRef=scaleRef[0], tauRho=tauRho[0],
                            scaleRho=scaleRho[0]))
        self.neuron_config.append(
            getNeuronConfig(theta=theta[1], tauSr=tauSr[1], tauRef=tauRef[1], scaleRef=scaleRef[1], tauRho=tauRho[1],
                            scaleRho=scaleRho[1]))
        self.neuron_config.append(
            getNeuronConfig(theta=theta[2], tauSr=tauSr[2], tauRef=tauRef[2], scaleRef=scaleRef[2], tauRho=tauRho[2],
                            scaleRho=scaleRho[2]))

        self.slayer1 = snn.layer(self.neuron_config[0], netParams['simulation'])
        self.slayer2 = snn.layer(self.neuron_config[1], netParams['simulation'])
        self.slayer3 = snn.layer(self.neuron_config[2], netParams['simulation'])

        # self.conv1 = self.slayer1.conv(1, 1, 5, padding=2)
        # self.conv2 = self.slayer2.conv(1, 1, 3, padding=1)
        # self.conv3 = self.slayer3.conv(1, 1, 3, padding=1)
        
        # self.conv1 = self.slayer1.conv(2, 2, 5, padding=2)
        # self.conv2 = self.slayer2.conv(2, 2, 3, padding=1)
        # self.conv3 = self.slayer3.conv(2, 2, 3, padding=1)

        self.conv1 = self.slayer1.conv(4, 4, 5, padding=2)
        self.conv2 = self.slayer2.conv(4, 4, 3, padding=1)
        self.conv3 = self.slayer3.conv(4, 4, 3, padding=1)

        # self.upconv1 = self.slayer3.convTranspose(8, 2, kernelSize=2, stride=2)


    def forward(self, spikeInput):

        _spikes_layer_1 = spikeInput[0][0][0].cpu().data.numpy()
        a = np.clip(_spikes_layer_1, 0, 1) # 将numpy数组约束在[0, 1]范围内
        _spikes_layer_1 = (a * 255).astype(np.uint8) # 转换成uint8类型
        img = Image.fromarray(_spikes_layer_1)
        img.save('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/SNN_visual3/{}_input.jpg'.format(self.global_num))

        #############################################################################
        psp1 = self.slayer1.psp(spikeInput)
        _spikes_layer_1 = psp1[0][0][0].cpu().data.numpy()
        print('_spikes_layer_1=',_spikes_layer_1.shape)
        a = np.clip(_spikes_layer_1, 0, 1) # 将numpy数组约束在[0, 1]范围内
        _spikes_layer_1 = (a * 255).astype(np.uint8) # 转换成uint8类型
        img = Image.fromarray(_spikes_layer_1)
        img.save('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/SNN_visual3/{}_spikes.jpg'.format(self.global_num))
        
        #############################################################################
        spikes_layer_1 = self.slayer1.spike(self.conv1(psp1))
        _spikes_layer_1 = spikes_layer_1[0][0][0].cpu().data.numpy()
        a = np.clip(_spikes_layer_1, 0, 1) # 将numpy数组约束在[0, 1]范围内
        _spikes_layer_1 = (a * 255).astype(np.uint8) # 转换成uint8类型
        img = Image.fromarray(_spikes_layer_1)
        img.save('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/SNN_visual3/{}_spikes1.jpg'.format(self.global_num))

        #############################################################################
        spikes_layer_2 = self.slayer2.spike(self.conv2(self.slayer2.psp(spikes_layer_1)))
        _spikes_layer_2 = spikes_layer_2[0][0][0].cpu().data.numpy()
        a = np.clip(_spikes_layer_2, 0, 1) # 将numpy数组约束在[0, 1]范围内
        _spikes_layer_2 = (a * 255).astype(np.uint8) # 转换成uint8类型
        img = Image.fromarray(_spikes_layer_2)
        img.save('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/SNN_visual3/{}_spikes2.jpg'.format(self.global_num))

        #############################################################################
        spikes_layer_3 = self.slayer3.spike(self.conv3(self.slayer3.psp(spikes_layer_2)))
        _spikes_layer_3 = spikes_layer_3[0][0][0].cpu().data.numpy()
        a = np.clip(_spikes_layer_3, 0, 1) # 将numpy数组约束在[0, 1]范围内
        _spikes_layer_3 = (a * 255).astype(np.uint8) # 转换成uint8类型
        img = Image.fromarray(_spikes_layer_3)
        img.save('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/SNN_visual3/{}_spikes3.jpg'.format(self.global_num))

        self.global_num = self.global_num + 1

        # b, t, c, w, h = spikes_layer_2.size()    # [16,8,3,256,128] [b, t, c, w, h]
        # spikes_layer_2 = spikes_layer_2.contiguous().view(b * t, c, w, h)  # x=[128,3,256,128]
        # print('spikes_layer_2=',spikes_layer_2.shape)
        return spikes_layer_2


######################################################################
######################################################################
######################################################################

class STRAHN1(nn.Module):
    
    def __init__(self, in_nc, out_nc, nf, unf, nb, scale=4):
        super(STRAHN1, self).__init__()
        # SCPA
        SCPA_block_f = functools.partial(SCPA, nf=nf, reduction=2)
        self.scale = scale
        self.global_num = 0
        ### first convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        ### main blocks
        self.SCPA_trunk = make_layer(SCPA_block_f, nb)
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
        self.snn = NetworkBasic(netParams=netParams).cuda()


        self.conv_pic = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        

    def forward(self, x):
        print('x=',x.shape)
        img = x[:,0:3,:,:]
        _img = self.conv_pic(img)

        _fea = _img[0,:,:,:].cpu().data.numpy()
        _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
        _a = _a.transpose(1,2,0)
        _a = (_a * 255).astype(np.uint8) # 转换成uint8类型
        _a = Image.fromarray(_a)
        _a.save('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/SNN_visual3/{}_img_feat1.jpg'.format(self.global_num))


        _fea = img[0,:,:,:].cpu().data.numpy()
        _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
        _a = _a.transpose(1,2,0)
        _a = (_a * 255).astype(np.uint8) # 转换成uint8类型
        _a = Image.fromarray(_a)
        _a.save('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/SNN_visual3/{}_img.jpg'.format(self.global_num))


        event = x[:,3:5,:,:]
        event = event.unsqueeze(0)
        event = event.cuda()
        # print('event=',event.shape)
        event = self.snn(event)
        # event的尺寸没有改变
        event = event.squeeze()
        x[:,3:5,:,:] = event

        fea = self.conv_first(x)
        _fea = fea[0,0,:,:].cpu().data.numpy()
        _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
        trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        df = pd.DataFrame(trans_prob_mat)
        plt.figure()
        ax = sns.heatmap(df, cmap='jet')
        plt.xticks(alpha=0)
        plt.tick_params(axis='x', width=0)
        plt.yticks(alpha=0)
        plt.tick_params(axis='y', width=0)
        plt.savefig('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/SNN_visual3/{}_img_fea2.jpg'.format(self.global_num), transparent=True, dpi=800)   

        _fea = fea[0,1,:,:].cpu().data.numpy()
        _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
        trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        df = pd.DataFrame(trans_prob_mat)
        plt.figure()
        ax = sns.heatmap(df, cmap='jet')
        plt.xticks(alpha=0)
        plt.tick_params(axis='x', width=0)
        plt.yticks(alpha=0)
        plt.tick_params(axis='y', width=0)
        plt.savefig('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/SNN_visual3/{}_img_fea3.jpg'.format(self.global_num), transparent=True, dpi=800)   

        _fea = fea[0,2,:,:].cpu().data.numpy()
        _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
        trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        df = pd.DataFrame(trans_prob_mat)
        plt.figure()
        ax = sns.heatmap(df, cmap='jet')
        plt.xticks(alpha=0)
        plt.tick_params(axis='x', width=0)
        plt.yticks(alpha=0)
        plt.tick_params(axis='y', width=0)
        plt.savefig('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/SNN_visual3/{}_img_fea4.jpg'.format(self.global_num), transparent=True, dpi=800)   

        _fea = fea[0,3,:,:].cpu().data.numpy()
        _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
        trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        df = pd.DataFrame(trans_prob_mat)
        plt.figure()
        ax = sns.heatmap(df, cmap='jet')
        plt.xticks(alpha=0)
        plt.tick_params(axis='x', width=0)
        plt.yticks(alpha=0)
        plt.tick_params(axis='y', width=0)
        plt.savefig('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/SNN_visual3/{}_img_fea5.jpg'.format(self.global_num), transparent=True, dpi=800)   

        _fea = fea[0,4,:,:].cpu().data.numpy()
        _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
        trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        df = pd.DataFrame(trans_prob_mat)
        plt.figure()
        ax = sns.heatmap(df, cmap='jet')
        plt.xticks(alpha=0)
        plt.tick_params(axis='x', width=0)
        plt.yticks(alpha=0)
        plt.tick_params(axis='y', width=0)
        plt.savefig('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/SNN_visual3/{}_img_fea6.jpg'.format(self.global_num), transparent=True, dpi=800)   

        _fea = fea[0,5,:,:].cpu().data.numpy()
        _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
        trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        df = pd.DataFrame(trans_prob_mat)
        plt.figure()
        ax = sns.heatmap(df, cmap='jet')
        plt.xticks(alpha=0)
        plt.tick_params(axis='x', width=0)
        plt.yticks(alpha=0)
        plt.tick_params(axis='y', width=0)
        plt.savefig('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/SNN_visual3/{}_img_fea7.jpg'.format(self.global_num), transparent=True, dpi=800)   

        _fea = fea[0,6,:,:].cpu().data.numpy()
        _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
        trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        df = pd.DataFrame(trans_prob_mat)
        plt.figure()
        ax = sns.heatmap(df, cmap='jet')
        plt.xticks(alpha=0)
        plt.tick_params(axis='x', width=0)
        plt.yticks(alpha=0)
        plt.tick_params(axis='y', width=0)
        plt.savefig('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/SNN_visual3/{}_img_fea8.jpg'.format(self.global_num), transparent=True, dpi=800)   

        _fea = fea[0,7,:,:].cpu().data.numpy()
        _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
        trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        df = pd.DataFrame(trans_prob_mat)
        plt.figure()
        ax = sns.heatmap(df, cmap='jet')
        plt.xticks(alpha=0)
        plt.tick_params(axis='x', width=0)
        plt.yticks(alpha=0)
        plt.tick_params(axis='y', width=0)
        plt.savefig('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/SNN_visual3/{}_img_fea9.jpg'.format(self.global_num), transparent=True, dpi=800)   

        _fea = fea[0,8,:,:].cpu().data.numpy()
        _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
        trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        df = pd.DataFrame(trans_prob_mat)
        plt.figure()
        ax = sns.heatmap(df, cmap='jet')
        plt.xticks(alpha=0)
        plt.tick_params(axis='x', width=0)
        plt.yticks(alpha=0)
        plt.tick_params(axis='y', width=0)
        plt.savefig('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/SNN_visual3/{}_img_fea10.jpg'.format(self.global_num), transparent=True, dpi=800)   

    
    
        ####################################################################################
        trunk = self.trunk_conv(self.SCPA_trunk(fea))
        _fea = trunk[0,0,:,:].cpu().data.numpy()
        _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
        trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        df = pd.DataFrame(trans_prob_mat)
        plt.figure()
        ax = sns.heatmap(df, cmap='jet')
        plt.xticks(alpha=0)
        plt.tick_params(axis='x', width=0)
        plt.yticks(alpha=0)
        plt.tick_params(axis='y', width=0)
        plt.savefig('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/SNN_visual3/{}_img_trunk1.jpg'.format(self.global_num), transparent=True, dpi=800)   


        
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
        _fea = trunk[0,0,:,:].cpu().data.numpy()
        _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
        trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        df = pd.DataFrame(trans_prob_mat)
        plt.figure()
        ax = sns.heatmap(df, cmap='jet')
        plt.xticks(alpha=0)
        plt.tick_params(axis='x', width=0)
        plt.yticks(alpha=0)
        plt.tick_params(axis='y', width=0)
        plt.savefig('/ghome/caocz/code/Event_Camera/Event_driven_SR/PAN_IJCV/SNN_visual3/{}_img_out1.jpg'.format(self.global_num), transparent=True, dpi=800)   


        self.global_num = self.global_num + 1

        ILR = F.interpolate(x[:,0:3,:,:], scale_factor=self.scale, mode='bilinear', align_corners=False)
        out = out + ILR
        return out
