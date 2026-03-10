import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

from models.archs_deblur.arch_util import EventImage_ChannelAttentionTransformerBlock

# from .MIMO_2C_e_plus_x import *
##################################################################################
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

###################################################################################################
######################################################################################################


class SFB2(nn.Module):
    def __init__(self, in_channel, out_channel,thita=1e-4):
        super(SFB2, self).__init__()
        self.thita = thita
        self.out_channel = out_channel
        self.value_conv = nn.Sequential(
            BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True),
        )
        
        self.key_conv = nn.Sequential(
            BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True),
        )
        
        self.query_conv = nn.Sequential(    # 提取事件特征
            BasicConv(self.out_channel,self.out_channel,kernel_size=1,stride=1,relu=True),
        )
        #########################################################
        #########################################################
        self.event_feature_extract = nn.Sequential(    # 提取事件特征
            BasicConv(out_channel*2,self.out_channel,kernel_size=1,stride=1,relu=True),
            BasicConv(out_channel*1,self.out_channel,kernel_size=1,stride=1,relu=True),
        )
        self.x_feature_extract = nn.Sequential(
            BasicConv(out_channel*3,self.out_channel,kernel_size=1,stride=1,relu=True),
            BasicConv(out_channel*1,self.out_channel,kernel_size=1,stride=1,relu=True),
        )

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        self.INF = INF
 
    def forward(self, x1, x2,event,last_event):
        x = torch.cat([x1, x2], dim=1)
        x = self.x_feature_extract[0](x)
        event = torch.cat([event,last_event],dim=1)
        event_feature = self.event_feature_extract[0](event)

        x_2 = F.interpolate(x,scale_factor=0.25)
        event_feature = F.interpolate(event_feature,scale_factor=0.25)

        x_2 = self.x_feature_extract[1](x_2)
        event_feature = self.event_feature_extract[1](event_feature)

        
        ################################################################
        #################################################################
        m_batchsize,C,width ,height = x_2.size()
        proj_query = self.query_conv(x_2).view(m_batchsize, -1, width*height).permute(0,2,1) # B X CX(N)
        proj_key = self.key_conv(event_feature).view(m_batchsize, -1, width*height) # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x_2).view(m_batchsize, -1, width*height) # B X C X N
 
        out = torch.bmm(proj_value,attention.permute(0, 2, 1) )
        out = out.view(m_batchsize, C, width, height)
 
        out = F.interpolate(out,scale_factor=4) 
        out = self.gamma * out
        out = self.thita * out + x
        
        event_feature = F.interpolate(event_feature,scale_factor=4)
        return out,event_feature
    


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img
    
class TMB(nn.Module):
    def __init__(self, out_channel,thita=1e-4):
        super(TMB, self).__init__()
        self.thita = thita
        self.out_channel = out_channel
        self.value_conv = nn.Sequential(
            BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True),
        )
        
        self.key_conv = nn.Sequential(
            BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True),
        )
        
        self.query_conv = nn.Sequential(    # 提取事件特征
            BasicConv(self.out_channel,self.out_channel,kernel_size=1,stride=1,relu=True),
        )
        #########################################################
        #########################################################
        self.event_feature_extract = nn.Sequential(    # 提取事件特征
            BasicConv(out_channel*2,self.out_channel,kernel_size=1,stride=1,relu=True),
            BasicConv(out_channel*1,self.out_channel,kernel_size=1,stride=1,relu=True),
        )
        self.x_feature_extract = nn.Sequential(
            BasicConv(out_channel*3,self.out_channel,kernel_size=1,stride=1,relu=True),
            BasicConv(out_channel*1,self.out_channel,kernel_size=1,stride=1,relu=True),
        )

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        self.INF = INF
 
 
        self.conv_neighbor = BasicConv(4,self.out_channel*2,kernel_size=1,stride=1,relu=True)
        self.conv_pro_key = BasicConv(self.out_channel,self.out_channel,kernel_size=1,stride=1,relu=True)
        self.conv_pro_val = BasicConv(self.out_channel,self.out_channel,kernel_size=1,stride=1,relu=True)
        self.conv_lat_key = BasicConv(self.out_channel,self.out_channel,kernel_size=1,stride=1,relu=True)
        self.conv_lat_val = BasicConv(self.out_channel,self.out_channel,kernel_size=1,stride=1,relu=True)
        self.conv_now1 = BasicConv(2,self.out_channel,kernel_size=1,stride=1,relu=True)
        # self.conv_now2 = BasicConv(self.out_channel,self.out_channel,kernel_size=1,stride=1,relu=True)

        self.conv_tmp1 = BasicConv(self.out_channel*2,self.out_channel,kernel_size=1,stride=1,relu=True)
        self.conv_tmp2 = BasicConv(self.out_channel*2,self.out_channel,kernel_size=1,stride=1,relu=True)

        self.conv_now_key = BasicConv(self.out_channel,self.out_channel,kernel_size=1,stride=1,relu=True)
        self.conv_now_val = BasicConv(self.out_channel,self.out_channel,kernel_size=1,stride=1,relu=True)
 
    def forward(self, event):
        event_pro = event[:,range(0,2),:,:]
        event_now = event[:,range(2,4),:,:]
        event_lat = event[:,range(4,6),:,:]

        event_pro = torch.as_tensor(event_pro)
        event_now = torch.as_tensor(event_now)
        event_lat = torch.as_tensor(event_lat)
        event_neighbor = torch.cat((event_pro,event_lat),1)
        event_neighbor = self.conv_neighbor(event_neighbor)
        event_pro_key = self.conv_pro_key(event_neighbor[:,range(0,self.out_channel),:,])
        event_pro_val = self.conv_pro_val(event_neighbor[:,range(self.out_channel,self.out_channel*2),:,:])
        event_lat_key = self.conv_lat_key(event_neighbor[:,range(0,self.out_channel),:,:])
        event_lat_val = self.conv_lat_val(event_neighbor[:,range(self.out_channel,self.out_channel*2),:,:])
        

        event_key_neighbor = torch.cat((event_pro_key,event_lat_key),1)
        event_val_neighbor = torch.cat((event_pro_val,event_lat_val),1)
        event_key_neighbor = self.conv_tmp1(event_key_neighbor)
        event_val_neighbor = self.conv_tmp1(event_val_neighbor)

        
        event_now = self.conv_now1(event_now)

        event_now_key = self.conv_now_key(event_now)
        event_now_val = self.conv_now_val(event_now)

        ################################################################
        #################################################################
        m_batchsize,C,width ,height = event_key_neighbor.size()
        proj_query = self.query_conv(event_key_neighbor).view(m_batchsize, -1, width*height).permute(0,2,1) # B X CX(N)
        proj_key = self.key_conv(event_now_key).view(m_batchsize, -1, width*height) # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(event_val_neighbor).view(m_batchsize, -1, width*height) # B X C X N
 
        out = torch.bmm(proj_value,attention.permute(0, 2, 1) )
        out = out.view(m_batchsize, C, width, height)
 
        out = self.thita * out + event_now_val

        return out

class ME(nn.Module):
    def __init__(self, ev_chn=6, wf=64, depth=3, relu_slope=0.2, fuse_before_downsample=True):
        super(ME, self).__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        # event
        self.down_path_ev = nn.ModuleList()
        self.conv_ev1 = nn.Conv2d(ev_chn, wf, 3, 1, 1)
        prev_channels = wf
        for i in range(depth):
            downsample = True if (i+1) < depth else False 
            if i < self.depth:
                self.down_path_ev.append(UNetEVConvBlock(prev_channels, (2**i) * wf, downsample , relu_slope))

            prev_channels = (2**i) * wf

    def forward(self, event):

        ev = []
        #EVencoder
        e1 = self.conv_ev1(event)
        for i, down in enumerate(self.down_path_ev):
            if i < self.depth-1:
                e1, e1_up = down(e1, self.fuse_before_downsample)
                if self.fuse_before_downsample:
                    ev.append(e1_up)
                else:
                    ev.append(e1)
            else:
                e1 = down(e1, self.fuse_before_downsample)
                ev.append(e1)
        
        return ev





class ESD(nn.Module):
    def __init__(self, in_chn=3, wf=64, depth=3, fuse_before_downsample=True, relu_slope=0.2, num_heads=[1,2,4]):
        super(ESD, self).__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        

        prev_channels = self.get_input_chn(wf)
        for i in range(depth):
            downsample = True if (i+1) < depth else False 

            self.down_path_1.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, num_heads=self.num_heads[i]))
            self.down_path_2.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_emgc=downsample))
            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.up_path_2.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.skip_conv_1.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            prev_channels = (2**i)*wf
        self.sam12 = SAM(prev_channels)

        self.cat12 = nn.Conv2d(prev_channels*2, prev_channels, 1, 1, 0)
        self.last = conv3x3(prev_channels, in_chn, bias=True)

    def forward(self, x, ev, mask=None):
        image = x
      
        

        #stage 1
        x1 = self.conv_01(image)
        encs = []
        decs = []
        masks = []
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:

                x1, x1_up = down(x1, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)
                encs.append(x1_up)

                if mask is not None:
                    masks.append(F.interpolate(mask, scale_factor = 0.5**i))
            
            else:
                x1 = down(x1, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)


        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i-1]))
            decs.append(x1)
        sam_feature, out_1 = self.sam12(x1, image)


        return out_1 + x

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False, num_heads=None): # cat
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc
        self.num_heads = num_heads

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)        

        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        if self.num_heads is not None:
            self.image_event_transformer = EventImage_ChannelAttentionTransformerBlock(out_size, num_heads=self.num_heads, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
        

    def forward(self, x, enc=None, dec=None, mask=None, event_filter=None, merge_before_downsample=True):
        out = self.conv_1(x)

        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)

        if enc is not None and dec is not None and mask is not None:
            assert self.use_emgc
            out_enc = self.emgc_enc(enc) + self.emgc_enc_mask((1-mask)*enc)
            out_dec = self.emgc_dec(dec) + self.emgc_dec_mask(mask*dec)
            out = out + out_enc + out_dec        
            
        if event_filter is not None and merge_before_downsample:
            # b, c, h, w = out.shape
            out = self.image_event_transformer(out, event_filter) 
             
        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample: 
                out_down = self.image_event_transformer(out_down, event_filter) 

            return out_down, out

        else:
            if merge_before_downsample:
                return out
            else:
                out = self.image_event_transformer(out, event_filter)


class UNetEVConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False):
        super(UNetEVConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_before_merge = nn.Conv2d(out_size, out_size , 1, 1, 0) 
        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, merge_before_downsample=True):
        out = self.conv_1(x)

        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)
             
        if self.downsample:

            out_down = self.downsample(out)
            
            if not merge_before_downsample: 
            
                out_down = self.conv_before_merge(out_down)
            else : 
                out = self.conv_before_merge(out)
            return out_down, out

        else:

            out = self.conv_before_merge(out)
            return out


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out






#####################################################################################
#####################################################################################
# class ESD(nn.Module):
#     def __init__(self, num_res,base_channel=16):
#         super(ESD, self).__init__()
#         self.thita = 1e-3
#         # base_channel = 32

#         self.Encoder = nn.ModuleList([
#             EBlock(base_channel, num_res),
#             EBlock(base_channel*2, num_res),
#             EBlock(base_channel*2, num_res),
#         ])

#         self.feat_extract = nn.ModuleList([
#             BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
#             BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
#             BasicConv(base_channel*2, base_channel*2, kernel_size=3, relu=True, stride=2),
#             BasicConv(base_channel*2, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
#             BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
#             BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
#         ])

#         self.Decoder = nn.ModuleList([
#             DBlock(base_channel * 2, num_res),
#             DBlock(base_channel * 2, num_res),
#             DBlock(base_channel, num_res)
#         ])

#         self.Convs = nn.ModuleList([
#             BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
#             BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
#         ])

#         self.ConvsOut = nn.ModuleList(
#             [
#                 BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
#                 BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
#             ]
#         )


#         self.event_c1=nn.Conv2d(in_channels=2, out_channels=10, kernel_size=1, stride=1, padding=0, bias=False)
#         self.event_c2=nn.Conv2d(in_channels=10, out_channels=base_channel, kernel_size=1, stride=1, padding=0, bias=False)


#         self.sfb2 = nn.ModuleList([
#             SFB2(base_channel*3,base_channel*1,self.thita)
#         ])

#         self.tmb = TMB(out_channel=base_channel)

#     def forward(self, x, output_last_feature=None):

#         event = x[:,range(3,9),:,:]
#         x = x[:,range(0,3),:,:]

#         event = self.tmb(event)

#         x_ = self.feat_extract[0](x)
#         res1 = self.Encoder[0](x_)

#         z = self.feat_extract[1](res1)
#         res2 = self.Encoder[1](z)

#         z = self.feat_extract[2](res2)
#         z = self.Encoder[2](z)


#         z21 = F.interpolate(res2, scale_factor=2)

#         if output_last_feature is not None:
#             res1,event_feature = self.sfb2[0](res1,z21,event,output_last_feature)
#         else:
#             return  
#             # res1,event_feature = self.sfb1[0](res1,z21,event)

#         z = self.Decoder[0](z)
#         z = self.feat_extract[3](z)

#         z = torch.cat([z, res2], dim=1)
#         z = self.Convs[0](z)
#         z = self.Decoder[1](z)
#         z = self.feat_extract[4](z)

#         z = torch.cat([z, res1], dim=1)
#         z = self.Convs[1](z)
#         z = self.Decoder[2](z)
#         z = self.feat_extract[5](z)

#         return z+x,event_feature






from torchstat import stat

# stat(model,(3,40,40))

from torchsummary import summary

# model = EFNet().to('cuda:0')
# model = ESD(num_res=2).to('cuda:0')

# summary(model, [(3, 40, 40), (6, 40, 40)])  
# summary(model,(9,40,40))  

input_img = torch.zeros(4,3,40,40).to('cuda:0')
input_ev = torch.zeros(4,6,40,40).to('cuda:0')
# output = model(input_img, input_ev)



# output_img,_ = model(input_img)
