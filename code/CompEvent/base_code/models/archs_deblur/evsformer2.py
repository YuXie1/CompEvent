import torch as th
import torch.nn.functional as F
from torch import nn


import torch
# from .net_util import ChannelAttention, ChannelAttention_softmax, SpatialAttention, SpatialAttention_softmax,EN_Block,DE_Block,Self_Attention
from einops import rearrange

import numbers

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)


def deconv4x4(in_channels, out_channels, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)


def deconv5x5(in_channels, out_channels, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, output_padding=1)

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x
    

def actFunc(act, *args, **kwargs):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'relu6':
        return nn.ReLU6()
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'rrelu':
        return nn.RReLU(0.1, 0.3)
    elif act == 'selu':
        return nn.SELU()
    elif act == 'celu':
        return nn.CELU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError


def make_blocks(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self, in_chs, activation='relu', batch_norm=False):
        super(ResBlock, self).__init__()
        op = []
        for i in range(2):
            op.append(conv3x3(in_chs, in_chs))
            if batch_norm:
                op.append(nn.BatchNorm2d(in_chs))
            if i == 0:
                op.append(actFunc(activation))
        self.main_branch = nn.Sequential(*op)

    def forward(self, x):
        out = self.main_branch(x)
        out += x
        return out


class DenseLayer(nn.Module):
    """
    Dense layer for residual dense block
    """

    def __init__(self, in_chs, growth_rate, activation='relu'):
        super(DenseLayer, self).__init__()
        self.conv = conv3x3(in_chs, growth_rate)
        self.act = actFunc(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class ResDenseBlock(nn.Module):
    """
    Residual Dense Block
    """

    def __init__(self, in_chs, growth_rate, num_layer, activation='relu'):
        super(ResDenseBlock, self).__init__()
        in_chs_acc = in_chs
        op = []
        for i in range(num_layer):
            op.append(DenseLayer(in_chs_acc, growth_rate, activation))
            in_chs_acc += growth_rate
        self.dense_layers = nn.Sequential(*op)
        self.conv1x1 = conv1x1(in_chs_acc, in_chs)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out


class RDNet(nn.Module):
    """
    Middle network of residual dense blocks
    """

    def __init__(self, in_chs, growth_rate, num_layer, num_blocks, activation='relu'):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(ResDenseBlock(in_chs, growth_rate, num_layer, activation))
        self.conv1x1 = conv1x1(num_blocks * in_chs, in_chs)
        self.conv3x3 = conv3x3(in_chs, in_chs)
        self.act = actFunc(activation)

    def forward(self, x):
        out = []
        h = x
        for i in range(self.num_blocks):
            h = self.RDBs[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.act(self.conv1x1(out))
        out = self.act(self.conv3x3(out))
        return out


class SpaceToDepth(nn.Module):
    """
    Pixel Unshuffle
    """

    def __init__(self, block_size=4):
        super().__init__()
        assert block_size in {2, 4}, "Space2Depth only supports blocks size = 4 or 2"
        self.block_size = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        S = self.block_size
        x = x.view(N, C, H // S, S, W // S, S)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * S * S, H // S, W // S)  # (N, C*bs^2, H//bs, W//bs)
        return x

    def extra_repr(self):
        return f"block_size={self.block_size}"


# based on https://github.com/rogertrullo/pytorch_convlstm
class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, input_chans, num_features, filter_size):
        super(CLSTM_cell, self).__init__()

        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2  # in this way the output has the same size
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
                              self.padding)

    def forward(self, input, hidden_state):
        hidden, c = hidden_state  # hidden and c are images with several channels
        combined = torch.cat((input, hidden), 1)  # oncatenate in the channels
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)  # it should return 4 tensors
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self, batch_size, shape):
        return (torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda(),
                torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda())
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out=self.sigmoid(out)
        return out



class ChannelAttention_softmax(nn.Module):
    def __init__(self, in_planes, ratio=16,L=32,M=2):
        super(ChannelAttention_softmax, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.in_planes=in_planes
        self.M=M
        d = max(in_planes // ratio, L)

        self.fc1=nn.Sequential(nn.Conv2d(in_planes,d,1,bias=False),
                               nn.ReLU(inplace=True))
        self.fc2=nn.Conv2d(d,in_planes*2,1,1,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out=self.avg_pool(x)
        max_out=self.max_pool(x)
        out = avg_out + max_out

        out = self.fc1(out)
        out_two = self.fc2(out)

        batch_size = x.size(0)

        out_two=out_two.reshape(batch_size,self.M,self.in_planes,-1)
        out_two = self.softmax(out_two)

        x_i, x_e = out_two[:, 0:1, :, :], out_two[:, 1:2, :, :]

        x_i = x_i.reshape(batch_size, self.in_planes, 1, 1)
        x_e = x_e.reshape(batch_size, self.in_planes, 1, 1)

        return x_i, x_e




class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1,kernel_size=(3,3), padding=(1,1), bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)

        return x


class SpatialAttention_softmax(nn.Module):
    def __init__(self):
        super(SpatialAttention_softmax, self).__init__()

        self.conv1 = nn.Conv2d(2, 1,kernel_size=(3,3), padding=(1,1), bias=False)
        self.conv2 = nn.Conv2d(2, 1,kernel_size=(3,3), padding=(1,1), bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x_i = self.conv1(x)
        x_e = self.conv2(x)
        x = torch.cat([x_i, x_e], dim=1)
        x = self.softmax(x)
        x_i, x_e = x[:, 0:1, :, :], x[:, 1:2, :, :]

        return x_i,x_e


# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res






class shallow_cell(nn.Module):
    def __init__(self,inChannels):
        super(shallow_cell, self).__init__()
        self.n_feats = 64
        act = nn.ReLU(inplace=True)
        bias = False
        reduction = 4
        self.shallow_feat = nn.Sequential(conv(inChannels, self.n_feats, 3, bias=bias),
                                           CAB(self.n_feats, 3, reduction, bias=bias, act=act))

    def forward(self,x):
        feat = self.shallow_feat(x)
        return feat







class EN_Block(nn.Module):

    def __init__(self, in_planes, planes,kernel_size=3, reduction=4, bias=False):
        super(EN_Block, self).__init__()

        act = nn.ReLU(inplace=True)
        self.down = DownSample(in_planes, planes)
        self.encoder = [CAB(planes, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder = nn.Sequential(*self.encoder)


    def forward(self, x):
        x = self.down(x)
        x=self.encoder(x)
        return x


class DE_Block(nn.Module):

    def __init__(self, in_planes, planes, kernel_size=3, reduction=4, bias=False):
        super(DE_Block, self).__init__()
        act = nn.ReLU(inplace=True)

        self.up=SkipUpSample(in_planes, planes)
        self.decoder = [CAB(planes, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder = nn.Sequential(*self.decoder)
        self.skip_attn = CAB(planes, kernel_size, reduction, bias=bias, act=act)


    def forward(self, x, skpCn):

        x = self.up(x, self.skip_attn(skpCn))
        x = self.decoder(x)

        return x


##########################################################################
## Layer Norm
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Recombine_Cross_Transformer(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Recombine_Cross_Transformer, self).__init__()
        self.num_heads = num_heads
        self.temperature_inp1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_inp1 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q_inp1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_inp1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_inp1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.q_inp2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_inp2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_inp2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)



        self.project_out_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, inp1, inp2):
        assert inp1.shape == inp2.shape, 'The shape of feature maps from image and event branch are not equal!'

        b, c, h, w = inp1.shape
        ###################img_attention###########################################
        q_inp1 = self.q_inp1(inp1)  # image
        k_inp1 = self.k_inp1(inp1)  # event
        v_inp1 = self.v_inp1(inp1)  # event

        q_inp1 = rearrange(q_inp1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_inp1 = rearrange(k_inp1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_inp1 = rearrange(v_inp1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_inp1 = torch.nn.functional.normalize(q_inp1, dim=-1)
        k_inp1 = torch.nn.functional.normalize(k_inp1, dim=-1)
        v_inp1 = torch.nn.functional.normalize(v_inp1, dim=-1)

        ###################event_attention###########################################
        q_inp2 = self.q_inp2(inp2)  # image
        k_inp2 = self.k_inp2(inp2)  # event
        v_inp2 = self.v_inp2(inp2)  # event

        q_inp2 = rearrange(q_inp2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_inp2 = rearrange(k_inp2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_inp2 = rearrange(v_inp2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_inp2 = torch.nn.functional.normalize(q_inp2, dim=-1)
        k_inp2 = torch.nn.functional.normalize(k_inp2, dim=-1)
        v_inp2 = torch.nn.functional.normalize(v_inp2, dim=-1)


        ###################cross_attention###########################################
        attn_inp1 = (q_inp1 @ k_inp2.transpose(-2, -1)) * self.temperature_inp1
        attn_inp1 = attn_inp1.softmax(dim=-1)

        out_inp1=(attn_inp1 @ v_inp2)
        out_inp1 = rearrange(out_inp1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_inp1 = self.project_out_1(out_inp1)

        attn_inp2 = (q_inp2 @ k_inp1.transpose(-2, -1)) * self.temperature_inp1
        attn_inp2 = attn_inp2.softmax(dim=-1)

        out_inp2=(attn_inp2 @ v_inp1)
        out_inp2 = rearrange(out_inp2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_inp2 = self.project_out_2(out_inp2)


        return out_inp1,out_inp2



class Divide_Cross_Transformer(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Divide_Cross_Transformer, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, inp1, inp2):

        assert inp1.shape == inp2.shape, 'The shape of feature maps from image and event branch are not equal!'

        b, c, h, w = inp1.shape

        q = self.q(inp2)  # image
        k = self.k(inp1)  # event
        v = self.v(inp1)  # event

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)


        return attn,v




class Coarse_Attention(nn.Module):
    """Modified version of Unet from SuperSloMo.

    Difference :
    1) diff=img-event.
    2)ca and sa
    """

    def __init__(self, inChannels):
        super(Coarse_Attention, self).__init__()
        self.conv = nn.Conv2d(inChannels*2, 2, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.pool=nn.AdaptiveAvgPool2d(1)

    def forward(self, img,event):

        x = torch.cat([img, event], dim=1)
        x=self.conv(x)
        x_i,x_e=x[:, 0:1, :, :], x[:, 1:2, :, :]
        x_i=self.sigmoid(x_i)
        x_e=self.sigmoid(x_e)
        x_i=self.pool(x_i)
        x_e=self.pool(x_e)
        g_img=x_i*img
        g_event=x_e*event

        return g_img, g_event






class Divide_Cross_Attention_Transformer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(Divide_Cross_Attention_Transformer, self).__init__()
        self.num_heads=num_heads
        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.norm1_common = LayerNorm(dim, LayerNorm_type)
        self.norm1_differential = LayerNorm(dim, LayerNorm_type)
        self.attn_common = Divide_Cross_Transformer(dim, num_heads, bias)
        self.attn_differential = Divide_Cross_Transformer(dim, num_heads, bias)

        # mlp
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.FFN_C = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.FFN_D = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, image,event):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = image.shape
        common=image+event
        differential=image-event
        ###########common################################

        attn_img_common,v_img_common=self.attn_common(self.norm1_image(image), self.norm1_common(common))
        attn_event_common,v_event_common=self.attn_common(self.norm1_event(event), self.norm1_common(common))

        attn_common_all=attn_img_common*attn_event_common
        out_common_img=attn_common_all @ v_img_common
        out_common_img = rearrange(out_common_img, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_common_event=attn_common_all @ v_event_common
        out_common_event = rearrange(out_common_event, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_common=out_common_img+out_common_event
        # out_common = rearrange(out_common, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_common = self.project_out1(out_common)
        out_common = to_3d(out_common) # b, h*w, c
        out_common = out_common + self.FFN_C(self.norm1(out_common))
        out_common = to_4d(out_common, h, w)

        ###########differential################################

        attn_img_differential,v_img_differential=self.attn_differential(self.norm1_image(image), self.norm1_differential(differential))
        attn_event_differential,v_event_differential=self.attn_differential(self.norm1_event(event), self.norm1_differential(differential))
        out_differential_img=attn_img_differential @ v_img_differential
        out_differential_img = rearrange(out_differential_img, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_differential_event=attn_event_differential @ v_event_differential
        out_differential_event = rearrange(out_differential_event, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_differential=out_differential_img+out_differential_event
        # out_differential = rearrange(out_differential, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_differential = self.project_out2(out_differential)
        out_differential = to_3d(out_differential) # b, h*w, c
        out_differential = out_differential + self.FFN_D(self.norm2(out_differential))
        out_differential = to_4d(out_differential, h, w)


        return out_common, out_differential


class Recombine_Cross_Attention_Transformer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(Recombine_Cross_Attention_Transformer, self).__init__()

        self.norm1_common = LayerNorm(dim, LayerNorm_type)
        self.norm1_differential = LayerNorm(dim, LayerNorm_type)
        self.attn = Recombine_Cross_Transformer(dim, num_heads, bias)
        # mlp
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.FFN_common = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.FFN_differential = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

        self.project_out = nn.Conv2d(3*dim, dim, kernel_size=1, bias=bias)

    def forward(self, common, differential):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        assert common.shape == differential.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = common.shape

        out_common, out_differential = self.attn(self.norm1_common(common), self.norm1_differential(differential))
        att_common = common + out_common
        att_differential = differential + out_differential

        # Linear_Projection
        att_common = to_3d(att_common)  # b, h*w, c
        att_common = att_common + self.FFN_common(self.norm1(att_common))
        att_common = to_4d(att_common, h, w)

        att_differential = to_3d(att_differential)  # b, h*w, c
        att_differential = att_differential + self.FFN_differential(self.norm2(att_differential))
        att_differential = to_4d(att_differential, h, w)

        fuse_add =att_common+att_differential
        fuse_product =att_common*att_differential
        fuse_max = torch.max(att_common,att_differential)
        fuse_cat = torch.cat((fuse_add, fuse_product,fuse_max), 1)
        fused=self.project_out(fuse_cat)



        return fused






class Encoder(nn.Module):
    """Modified version of Unet from SuperSloMo.

    Difference :
    1) there is an option to skip ReLU after the last convolution.
    2) there is a size adapter module that makes sure that input of all sizes
       can be processed correctly. It is necessary because original
       UNet can process only inputs with spatial dimensions divisible by 32.
    """

    def __init__(self, inChannels):
        super(Encoder, self).__init__()
        # self.inplanes = 32
        self.num_heads=4
        ######encoder部分
        ################################Resnet Image#######################################
        self.head = shallow_cell(inChannels)
        self.down1 = EN_Block(64, 128)  # 128
        self.down2 = EN_Block(128, 256)  # 64
        # self.down3 = EN_Block(128, 256)  # 32
        # self.down4 = EN_Block(256, 512)  # 16

    def forward(self, input):
        # Size adapter spatially augments input to the size divisible by 32.
        # x=torch.cat((input_img, input_event), 1)
        s0 = self.head(input)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        # s3 = self.down3(s2)
        # s4 = self.down4(s3)
        x = [s0, s1, s2]
        return x


class Decoder(nn.Module):
    """Modified version of Unet from SuperSloMo.
    """

    def __init__(self, outChannels):
        super(Decoder, self).__init__()
        ######Decoder
        # self.up1 = DE_Block(512, 256)
        # self.up2 = DE_Block(256, 128)
        self.up3 = DE_Block(256, 128)
        self.up4 = DE_Block(128, 64)

    def forward(self, input,skip):
        x=input
        # x = self.up1(x, skip[3])
        # x = self.up2(x, skip[2])
        x = self.up3(x, skip[1])
        x = self.up4(x, skip[0])

        # x = self.conv(x)
        # x=x+input_img
        return x


class Restoration(nn.Module):
    """Modified version of Unet from SuperSloMo.

    """

    def __init__(self, inChannels_img, inChannels_event,outChannels, args,ends_with_relu=False):
        super(Restoration, self).__init__()
        self._ends_with_relu = ends_with_relu
        self.num_heads=4
        ######encoder
        self.encoder_img=Encoder(inChannels_img)
        self.encoder_event=Encoder(inChannels_event)
        self.decoder = Decoder(outChannels)
        ######fusion
        self.Divide = Divide_Cross_Attention_Transformer(256, num_heads=self.num_heads,ffn_expansion_factor=4, bias=False,
                                                                                   LayerNorm_type='WithBias')

        self.Recombine = Recombine_Cross_Attention_Transformer(256, num_heads=self.num_heads,ffn_expansion_factor=4, bias=False,
                                                                                   LayerNorm_type='WithBias')

        self.conv = nn.Conv2d(64, outChannels, 3, stride=1, padding=1)


    def forward(self, input_img, input_event):


        ####  feature extraction
        en_img=self.encoder_img(input_img)
        en_event=self.encoder_event(input_event)

        #####Divide_and_ Recombine#########################

        ###Divide
        out_common, out_differential=self.Divide(en_img[-1],en_event[-1])
        ###Recombine
        out_fusion=self.Recombine(out_common,out_differential)

        #############decoder
        out=self.decoder(out_fusion,en_img)
        out = self.conv(out)
        out=out+input_img
        return out
    


# my_model = Restoration(inChannels_img=3, inChannels_event=6,outChannels=3,args=None)

# _input = torch.zeros(1,3,128,128)
# _evsinput = torch.zeros(1,6,128,128)
# _output = my_model(_input,_evsinput)
# print('_input=',_input.shape)
# print('_output=',_output.shape)