from pickle import FALSE
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
from torchvision import models


def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)
                     
def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, 
                     stride=stride, padding=2, bias=True)   

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        
    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out
        
class Encoder_input(nn.Module):
    def __init__(self, num_res_blocks, n_feats, img_channel, res_scale=1):
        super(Encoder_input, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(img_channel, n_feats)
        
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats, 
                res_scale=res_scale))
            
        self.conv_tail = conv3x3(n_feats, n_feats)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x

class ResList(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale=1):
        super(ResList, self).__init__()
        self.num_res_blocks = num_res_blocks
        
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats, 
                res_scale=res_scale))
            
        self.conv_tail = conv3x3(n_feats, n_feats)
        
    def forward(self, x):
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x
        
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)

        self.weight.requires_grad = False
        self.bias.requires_grad = False   

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False,In=False,act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, stride=stride, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if In:
            m.append(nn.InstanceNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


######################################################################
######################################################################
######################################################################
class AlignedConv2d(nn.Module):
    def __init__(self, inc, outc=1, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(AlignedConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ReflectionPad2d(padding)

        head = [nn.Conv2d(3, 32, kernel_size=5, padding=2, stride=1), nn.LeakyReLU(0.2, inplace=True), ResBlock(in_channels=32, out_channels=32), nn.LeakyReLU(0.2, inplace=True)]

        head2 = [nn.Conv2d(2*32, 32, kernel_size=5, padding=2, stride=stride), nn.LeakyReLU(0.2, inplace=True), ResBlock(in_channels=32, out_channels=32), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(32, 3, kernel_size=1, padding=0, stride=1)]

        self.p_conv = nn.Sequential(*head2)
        self.conv1 = nn.Sequential(*head)
        self.p_conv.register_backward_hook(self._set_lr)    
        self.conv1.register_backward_hook(self._set_lr)    


        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(2*inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
       grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
       grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x, query,ref):
        query = F.interpolate(query, scale_factor=2, mode='bicubic')
        ref = self.conv1(ref)
        query = self.conv1(query)
    
        affine = self.p_conv( torch.cat((ref, query), 1)) + 1.
        if self.modulation:
            m = torch.sigmoid(self.m_conv(torch.cat((ref, query), 1)))

        dtype = affine.data.type()
        ks = self.kernel_size
        N = ks*ks

        if self.padding:
            x = self.zero_padding(x)

        
        affine = torch.clamp(affine, -3, 3)
        # (b, 2N, h, w)
        p = self._get_p(affine, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        alignment = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(alignment.size(1))], dim=1)
            alignment *= m

        alignment = self._reshape_alignment(alignment, ks)

        return alignment

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(torch.arange(-1*((self.kernel_size-1)//2)-0.5, (self.kernel_size-1)//2+0.6, 1.),  torch.arange(-1*((self.kernel_size-1)//2)-0.5, (self.kernel_size-1)//2+0.6, 1.))          
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)


        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0




    def _get_p(self, affine, dtype):
        N, h, w = self.kernel_size*self.kernel_size, affine.size(2), affine.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)

        p =  p_n.repeat(affine.size(0), 1, h, w) 
        p = p.permute(0,2,3,1) #(1,  h, w, 2N)
        affine = affine.permute(0,2,3,1) #-1xh x w x 3   




        s_x =  affine[:,:,:,0:1]


        s_y =  affine[:,:,:,1:2]


        p[:,:,:,:N] = p[:,:,:,:N].clone()*s_x.type(dtype)
        p[:,:,:,N:] = p[:,:,:,N:].clone()*s_y.type(dtype)
        p = p.view(p.shape[0],p.shape[1], p.shape[2], 1, p.shape[3]) #(1,  h, w, 1, 2N)
        p= torch.cat((p[:,:,:,:,:N], p[:,:,:,:,N:]), 3) #(1,  h, w, 2, N)
        p = p.permute(0,1,2,4,3) #(1,  h, w,  N, 2)

        theta = (affine[:,:,:,2:] - 1.)*1.0472
        rm = torch.cat((torch.cos(theta), torch.sin(theta),-1*torch.sin(theta), torch.cos(theta)), 3)
        rm = rm.view(affine.shape[0],affine.shape[1],affine.shape[2],2,2 ) #-1xh x w x 2x2
        result = torch.matmul(p,rm) #-1xh x w x Nx2
        result= torch.cat((result[:,:,:,:,0], result[:,:,:,:,1]), 3) #(-1,  h, w,  2N)
    
        result = result.permute(0,3,1,2) +(self.kernel_size-1)//2+0.5 + p_0#-1, 2N, h, w



        return result


    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        result = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return result

    @staticmethod
    def _reshape_alignment(alignment, ks):
        b, c, h, w, N = alignment.size()
        alignment = torch.cat([alignment[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        alignment = alignment.contiguous().view(b, c, h*ks, w*ks)

        return alignment    
#########################################################################
#########################################################################
#########################################################################
def normalize(x):
    return x.mul_(2).add_(-1)

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ReflectionPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()
    
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks
def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


##########################################################################
##########################################################################
##########################################################################
class FeatureMatching(nn.Module):
    def __init__(self, ksize=3, k_vsize=1,  scale=2, stride=1, in_channel =3, out_channel =64, conv=default_conv):
        super(FeatureMatching, self).__init__()
        self.ksize = ksize
        self.k_vsize = k_vsize
        self.stride = stride  
        self.scale = scale
        match0 =  BasicBlock(conv, 128, 16, 1,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))

        vgg_pretrained_features = models.vgg19(pretrained=False).features
        self.feature_extract = torch.nn.Sequential()
        
        for x in range(7):
            self.feature_extract.add_module(str(x), vgg_pretrained_features[x])
            
        self.feature_extract.add_module('map', match0)
        
   
        for param in self.feature_extract.parameters():
            param.requires_grad = True
            

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 , 0.224, 0.225 )
        self.sub_mean = MeanShift(1, vgg_mean, vgg_std) 
        self.avgpool = nn.AvgPool2d((self.scale,self.scale),(self.scale,self.scale))            


    def forward(self, query, key,flag_8k):
        #input query and key, return matching
    
        # query = self.sub_mean(query)
        if not flag_8k:
           query  = F.interpolate(query, scale_factor=self.scale, mode='bicubic',align_corners=True)
        # there is a pooling operation in self.feature_extract
        query = self.feature_extract(query)
        shape_query = query.shape
        query = extract_image_patches(query, ksizes=[self.ksize, self.ksize], strides=[self.stride,self.stride], rates=[1, 1], padding='same') 
      

        key = self.avgpool(key)
        # key = self.sub_mean(key)
        if not flag_8k:
           key  = F.interpolate(key, scale_factor=self.scale, mode='bicubic',align_corners=True)
        # there is a pooling operation in self.feature_extract
        key = self.feature_extract(key)
        shape_key = key.shape
        w = extract_image_patches(key, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1], padding='same')
    

        w = w.permute(0, 2, 1)   
        w = F.normalize(w, dim=2) # [N, Hr*Wr, C*k*k]
        query  = F.normalize(query, dim=1) # [N, C*k*k, H*W]
        y = torch.bmm(w, query) #[N, Hr*Wr, H*W]
        relavance_maps, hard_indices = torch.max(y, dim=1) #[N, H*W]   
        relavance_maps = relavance_maps.view(shape_query[0], 1, shape_query[2], shape_query[3])      

        return relavance_maps,  hard_indices


class AlignedAttention(nn.Module):
    def __init__(self,  ksize=3, k_vsize=1,  scale=1, stride=1, align =False):
        super(AlignedAttention, self).__init__()
        self.ksize = ksize
        self.k_vsize = k_vsize
        self.stride = stride
        self.scale = scale
        self.align= align
        if align:
          self.align = AlignedConv2d(inc=128, outc=1, kernel_size=self.scale*self.k_vsize, padding=1, stride=self.scale*1, bias=None, modulation=False)        

    def warp(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, lr, ref, index_map, value):
        # value there can be features or image in ref view

        # b*c*h*w
        shape_out = list(lr.size())   # b*c*h*w
 
        # kernel size on input for matching 
        kernel = self.scale*self.k_vsize

        # unfolded_value is extracted for reconstruction 

        unfolded_value = extract_image_patches(value, ksizes=[kernel, kernel],  strides=[self.stride*self.scale,self.stride*self.scale], rates=[1, 1], padding='same') # [N, C*k*k, L]
        warpped_value = self.warp(unfolded_value, 2, index_map)
        warpped_features = F.fold(warpped_value, output_size=(shape_out[2]*2, shape_out[3]*2), kernel_size=(kernel,kernel), padding=0, stride=self.scale) 
        if self.align:
          unfolded_ref = extract_image_patches(ref, ksizes=[kernel, kernel],  strides=[self.stride*self.scale,self.stride*self.scale], rates=[1, 1], padding='same') # [N, C*k*k, L]
          warpped_ref = self.warp(unfolded_ref, 2, index_map)
          warpped_ref = F.fold(warpped_ref, output_size=(shape_out[2]*2, shape_out[3]*2), kernel_size=(kernel,kernel), padding=0, stride=self.scale)         
          warpped_features = self.align(warpped_features,lr,warpped_ref)        

        return warpped_features     
   

class PatchSelect(nn.Module):
    def __init__(self,  stride=1):
        super(PatchSelect, self).__init__()
        self.stride = stride             

    def forward(self, query, key):
        shape_query = query.shape
        shape_key = key.shape
        
        P = shape_key[3] - shape_query[3] + 1 #patch number per row
        key = extract_image_patches(key, ksizes=[shape_query[2], shape_query[3]], strides=[self.stride, self.stride], rates=[1, 1], padding='valid')

        query = query.view(shape_query[0], shape_query[1]* shape_query[2] *shape_query[3],1)

        y = torch.mean(torch.abs(key - query), 1)

        relavance_maps, hard_indices = torch.min(y, dim=1, keepdim=True) #[N, H*W]   
        

        return  hard_indices.view(-1), P, relavance_maps

####################################################################
####################################################################

##########################################################################
##########################################################################

class DCSR(nn.Module):
    def __init__(self, scale,n_feats, conv= default_conv):
        super(DCSR, self).__init__()

   
        # n_feats = 32
        kernel_size = 3 
        scale = scale 
        self.scale = scale   
        self.flag_8k = FALSE
        # define head module
        m_head1 = [BasicBlock(conv, 3, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True)),
        BasicBlock(conv,n_feats, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))]

        m_head2 = [BasicBlock(conv, n_feats, n_feats, kernel_size,stride=2,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True)),
            BasicBlock(conv, n_feats, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))]            

        # define tail module
        m_tail = [conv3x3(n_feats, n_feats//2), nn.LeakyReLU(0.2, inplace=True), conv3x3(n_feats//2, 3) ]

        fusion1 = [BasicBlock(conv, 2*n_feats, n_feats, 5,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True)) ,
        BasicBlock(conv, n_feats, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))]

        fusion2= [BasicBlock(conv, 2*n_feats, n_feats, 5,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True)) ,
        BasicBlock(conv, n_feats, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))]        

          
        self.feature_match = FeatureMatching(ksize=3,  scale=2, stride=1, in_channel = 3, out_channel = 64)

        self.ref_encoder1 = nn.Sequential(*m_head1) #encoder1
        self.ref_encoder2 = nn.Sequential(*m_head2) #encoder3
      
        self.res1 = ResList(4, n_feats) #res3
        self.res2 = ResList(4, n_feats) #res1
        
        self.input_encoder = Encoder_input(8, n_feats, 3)        

        self.fusion1 = nn.Sequential(*fusion1)
        self.decoder1 = ResList(8, n_feats)     

        
        self.fusion2 = nn.Sequential(*fusion2) #fusion3
        self.decoder2 = ResList(4, n_feats)    #decoder3

        self.decoder_tail = nn.Sequential(*m_tail)

        
        fusion11 = [BasicBlock(conv, 1, 16, 7,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True)) ,
        BasicBlock(conv, 16, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))]
        
        fusion12= [BasicBlock(conv, 1, 16, 7,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True)) ,
        BasicBlock(conv, 16, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))]    
            
        fusion13= [BasicBlock(conv, 4, 32, 7,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True)) ,
        BasicBlock(conv, 32, 3, kernel_size,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))]  
                 
        self.alpha1 = nn.Sequential(*fusion11)  # g layer in paper
        self.alpha2 = nn.Sequential(*fusion12) # alpha3
        self.alpha3 = nn.Sequential(*fusion13) # alpha4      

        if self.flag_8k:
            self.aa1 = AlignedAttention(scale=4, align=True) #swap4
            self.aa2 = AlignedAttention(scale=2, align = False) #swap3
            self.aa3 = AlignedAttention(scale=4, align=True) #swap2
        else:
            self.aa1 = AlignedAttention(scale=2, align=True) #swap4
            self.aa2 = AlignedAttention(scale=1, align = False) #swap3
            self.aa3 = AlignedAttention(scale=2, align=True) #swap2
        
        self.avgpool = nn.AvgPool2d((2,2),(2,2)) 
        self.select = PatchSelect()
        
        self.first_conv = nn.Conv2d(45, 3, 3, 1, 1, bias=True)
        self.second_conv = nn.Conv2d(3,8,3,1,1,bias=True)
        #########################################################
        # 扩大4倍时，需要的东西
        #########################################################
        self.ps1 = nn.PixelShuffle(2)
        self.endconv = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        
        
    def forward(self,input, coarse = False):
        input = self.first_conv(input)
        if self.scale == 4:
            _input = self.second_conv(input)
            _input = self.ps1(_input)
            input = self.endconv(_input)
        
        ref = input
        ref_p = ref    
        
     
        confidence_map,  index_map = self.feature_match(input, ref_p,self.flag_8k)
       
        ref_downsampled = self.avgpool(ref_p)
        ref_hf = ref_p -  F.interpolate(ref_downsampled, scale_factor=2, mode='bicubic')   #extract high freq in ref
        ref_hf_aligned = self.aa1(input, ref_p, index_map, ref_hf)          
       
        ref_features1 = self.ref_encoder1(ref_p) 
        ref_features1 = self.res1(ref_features1)
   
        ref_features2 = self.ref_encoder2(ref_features1)   
        ref_features2 = self.res2(ref_features2) 
        
        input_down =  F.interpolate(input, scale_factor=1/2, mode='bicubic')
        ref_features_matched = self.aa2(input_down, ref_p, index_map, ref_features2)                  
        ref_features_aligned = self.aa3(input, ref_p, index_map, ref_features1)

        input_up = self.input_encoder(input)

        if  self.flag_8k:
             confidence_map = F.interpolate(confidence_map, scale_factor=2, mode='bicubic')
        cat_features = torch.cat((ref_features_matched, input_up), 1)
        fused_features2 = self.alpha1(confidence_map) * self.fusion1(cat_features) + input_up  #adaptive fusion in feature space
        fused_features2 = self.decoder1(fused_features2)
        fused_features2_up = F.interpolate(fused_features2, scale_factor=2, mode='bicubic')

        confidence_map = F.interpolate(confidence_map, scale_factor=2, mode='bicubic')
        cat_features = torch.cat((ref_features_aligned, fused_features2_up), 1)
        fused_features1 = self.alpha2(confidence_map) *self.fusion2(cat_features) + fused_features2_up #adaptive fusion in feature space
        fused_features1 = self.decoder2(fused_features1)
        result = self.decoder_tail(fused_features1) + ref_hf_aligned*self.alpha3(torch.cat((confidence_map, ref_hf_aligned), 1)) #adaptive fusion in image space


        return result

