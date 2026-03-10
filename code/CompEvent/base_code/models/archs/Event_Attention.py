
from __future__ import print_function
from re import L

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial import KDTree


def knn_query(pos_support, pos, k):
    """Dense knn serach
    Arguments:
        pos_support - [B,N,3] support points
        pos - [B,M,3] centre of queries
        k - number of neighboors, needs to be > N
    Returns:
        idx - [B,M,k]
        dist2 - [B,M,k] squared distances
    """
    dist_all = []
    points_all = []
    for x, y in zip(pos_support, pos):
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        kdtree = KDTree(x)
        dist, points = kdtree.query(y, k)   # y=[512,3],k=32
        # 只需要用训练数据建一个kdtree，然后用kdtree的query函数找最近邻
        # dist.size=[512,32]
        dist_all.append(dist)
        points_all.append(points)

#    return torch.tensor(points_all, dtype=torch.int64, device='cuda'), torch.tensor(dist_all, dtype=torch.float64,
#                                                                                    device='cuda')
    return torch.tensor(points_all, dtype=torch.int64), torch.tensor(dist_all, dtype=torch.float64
)   # points_all=[512,32]   dist_all=[512,32]

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def grouping(npoint, K, xyz, new_xyz, points):
    # K=32
    B, N, C = new_xyz.shape
    S = npoint
    idx, _ = knn_query(xyz, new_xyz, K)
    # idx=[1,512,32]
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    # [1,512,32,3]
    grouped_feature = index_points(points, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)


    if points is not None:
        new_points = torch.cat([grouped_xyz, grouped_feature], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz

    return grouped_xyz, new_points



class SampleWeight(nn.Module):
    """Input
        grouped_feature: (batch_size, npoint, nsample, channel) Torch tensor
        grouped_xyz: (batch_size, npoint, nsample, 3)
        new_point: (batch_size, npoint, nsample, channel)
        Output
        (batch_size, npoint, nsample, 1)
    """
    def __init__(self, channel, mlps): # channel is new_point's
        super(SampleWeight, self).__init__()
        self.bottleneck_channel = max(32,channel//2)

        self.transformed_feature = nn.Conv2d(channel+3, self.bottleneck_channel * 2, 1)
        self.tf_bn = nn.BatchNorm2d(self.bottleneck_channel*2)
        self.transformed_new_point = nn.Conv2d(channel+3, self.bottleneck_channel, 1)
        self.tnp_bn = nn.BatchNorm2d(self.bottleneck_channel)

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = self.bottleneck_channel
        for out_channel in mlps:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, new_point, grouped_xyz):
        device = new_point.device
        B, N, S, C = new_point.shape
        normalized_xyz = grouped_xyz - grouped_xyz[:,:,0,:].view(B, N, 1, 3).repeat(1, 1, S, 1)
        new_point = torch.cat([normalized_xyz, new_point], -1).permute(0,3,1,2) #channel last -> channel first

        transformed_feature = self.tf_bn(self.transformed_feature(new_point)).permute(0,2,3,1) #channel first -> channel last
        transformed_new_point = self.tnp_bn(self.transformed_new_point(new_point)).permute(0,2,3,1)

        transformed_feature1 = transformed_feature[:, :, :, :self.bottleneck_channel]
        feature = transformed_feature[:, :, :, self.bottleneck_channel:]

        weights = torch.matmul(transformed_new_point, transformed_feature1.transpose(2, 3))
        weights = weights / np.sqrt(self.bottleneck_channel) #if scaled
        weights = F.softmax(weights, -1)
        C = self.bottleneck_channel


        new_group_features = torch.matmul(weights, feature)
        new_group_features = torch.reshape(new_group_features, (B, N, S, C)).permute(0,3,1,2)
    
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_group_features = F.relu(bn(conv(new_group_features)), inplace=True) if i < len(self.mlp_convs) - 1 else F.softmax(bn(conv(new_group_features)), 2)
        new_group_features = new_group_features.permute(0,2,3,1) 
        return new_group_features 

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class Adaptivesampling(nn.Module):
    def __init__(self, nsample, group_feature_channel, num_neighbor=8): #num channel
        super(Adaptivesampling, self).__init__()
        self.nsample = nsample
        self.num_neighbor = num_neighbor
        self.sampleweight = SampleWeight(group_feature_channel, [32, 1+group_feature_channel])

    def forward(self, group_xyz, group_feature):
        if self.num_neighbor == 0:
            new_xyz = group_xyz[:, :, 0, :]
            new_feature = group_feature[:, :, 0, :]
            return new_xyz, new_feature
        shift_group_xyz = group_xyz[:,:,:self.num_neighbor,:]
        shift_group_points = group_feature[:,:,:self.num_neighbor,:]
        sample_weight = self.sampleweight(shift_group_points, shift_group_xyz).clone().detach()
        sw1, sw2, sw3, _ = sample_weight.shape
        new_weight_xyz = sample_weight[:,:,:,0].view(sw1, sw2, sw3, 1).repeat(1,1,1,3)
        new_weight_feature = sample_weight[:,:,:,1:]
        new_xyz = torch.sum(shift_group_xyz*new_weight_xyz, dim = 2)
        new_feature = torch.sum(shift_group_points*new_weight_feature, dim = 2)

        return new_xyz, new_feature

class weight_net_hidden(nn.Module):
    def __init__(self, in_channel, hidden_units):
        super(weight_net_hidden, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in hidden_units:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    def forward(self, xyz):
        net = xyz.permute(0,3,1,2)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            net = F.relu(bn(conv(net)), inplace=True)
        net = net.permute(0,2,3,1)
        return net


class PointNonLocalCell(nn.Module):
    """Input
        feature: (batch_size, ndataset, channel) Torch tensor
        new_point: (batch_size, npoint, nsample, channel)
        Output
        (batch_size, npoint, nsample, channel)
    """
    def __init__(self, feature_channel, in_channel, mlps):
        super(PointNonLocalCell, self).__init__()
        self.bottleneck_channel = mlps[0]
        # feature_channel=0
        self.transformed_feature = nn.Conv2d(feature_channel, self.bottleneck_channel * 2, 1)
        self.tf_bn = nn.BatchNorm2d(self.bottleneck_channel*2)
        self.transformed_new_point = nn.Conv2d(in_channel, self.bottleneck_channel, 1)
        self.tnp_bn = nn.BatchNorm2d(self.bottleneck_channel)
        self.new_nonlocal_point = nn.Conv2d(self.bottleneck_channel, mlps[-1], 1)
        self.nnp_bn = nn.BatchNorm2d(mlps[-1])

    def forward(self, feature, new_point):
        B, P, S, C = new_point.shape
        FB, FD, FC = feature.shape
        feature = feature.view(FB, FD, 1, FC).permute(0,3,1,2)
        new_point = new_point.permute(0,3,1,2)
        
        transformed_feature = self.tf_bn(self.transformed_feature(feature)).permute(0,2,3,1)
        transformed_new_point = self.tnp_bn(self.transformed_new_point(new_point)).permute(0,2,3,1)
        transformed_new_point = torch.reshape(transformed_new_point, (B, P*S, self.bottleneck_channel))
        transformed_feature1 = torch.squeeze(transformed_feature[:,:,:,:self.bottleneck_channel], 2)
        transformed_feature2 = torch.squeeze(transformed_feature[:,:,:,self.bottleneck_channel:], 2)

        attention_map = torch.matmul(transformed_new_point, transformed_feature1.transpose(1,2)) #mode = 'dot'
        attention_map = attention_map / np.sqrt(self.bottleneck_channel)
        attention_map = F.softmax(attention_map, -1)

        new_nonlocal_point = torch.matmul(attention_map, transformed_feature2)
        new_nonlocal_point = torch.reshape(new_nonlocal_point, (B, P, S, self.bottleneck_channel)).permute(0,3,1,2)
        new_nonlocal_point = self.nnp_bn(self.new_nonlocal_point(new_nonlocal_point)).permute(0,2,3,1)
        new_nonlocal_point = torch.squeeze(new_nonlocal_point, 1)

        return new_nonlocal_point

class PointASNLSetAbstraction(nn.Module): ## use_knn = true, 
    ''' Input:
            xyz: (batch_size, ndataset, 3) Torch tensor
            feature: (batch_size, ndataset, channel) Torch tensor
            point: int32 -- #points sampled in Euclidean space by farthest point sampling
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_xyz: (batch_size, npoint, 3) Torch tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) Torch tensor
    '''
    def __init__(self, npoint, nsample, in_channel, mlp, as_neighbor=8):
        # npoint=512, nsample = 32, in_channel = inchannel, mlp = [64,64,128]
        super(PointASNLSetAbstraction, self).__init__()
        self.npoint = npoint
        self.nsample = nsample

        self.as_neighbor = as_neighbor

        self.Adaptive_smapling = Adaptivesampling(nsample, in_channel, as_neighbor)
        self.PointNonLocalCell = PointNonLocalCell(in_channel-3, in_channel, [max(32, in_channel//2), mlp[-1]])
        # self.PointNonLocalCell = PointNonLocalCell(in_channel-2, in_channel, [max(32, in_channel//2), mlp[-1]])

        self.skip_spatial = nn.Conv1d(in_channel + 3, mlp[-1], 1)
        self.ss_bn = nn.BatchNorm1d(mlp[-1])

        self.weighted_conv = nn.Conv2d(32, mlp[-1], (1, mlp[-1])) ##matrix multiplied weight is (B, N, sample, 32)
        self.wc_bn = nn.BatchNorm2d(mlp[-1])

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weight_net_hidden = weight_net_hidden(3, [32])
        self.feature_fushion = nn.Conv1d(mlp[-1] ,mlp[-1], 1)
        self.ff_bn = nn.BatchNorm1d(mlp[-1])

    # def forward(self, xyz, feature):        # xyz.size=[2,3,1024], feature.size=[2,3,1024]

    def forward(self, event_data, event_feature):        # xyz.size=[2,3,1024], feature.size=[2,3,1024]

        # 这里的feature指的是点云所具有的的属性，比如空间梯度，颜色，反射强度等等
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # xyz = xyz.permute(0, 2, 1)
        # if feature is not None:
        #     feature = feature.permute(0, 2, 1)
        # batch_size, num_points, num_channel = feature.shape
        # if num_points != self.npoint :
        #     new_xyz, new_feature = sampling(self.npoint, xyz, feature)
        #     # xyz=[1,1024,3] --> new_xyz=[1,512,3]
        #     # feature=[1,1024,3] --> new_feature=[1,512,3]
        # else :
        #     new_feature = feature
        #     new_xyz = xyz
        
 
        batch_size, num_points, num_channel = event_feature.shape
        grouped_xyz, new_points = grouping(self.npoint, self.nsample, event_data, event_data, event_feature)

        
        # new_xyz: sampled points position data, [B, npoint, C]
        # grouped_xyz : grouped points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        '''Adaptive Sampling'''
        # if num_points != self.npoint :
        new_xyz, new_feature = self.Adaptive_smapling(grouped_xyz, new_points)
        nB, nN, nC = new_xyz.shape

        grouped_xyz -= new_xyz.view(nB, nN, 1, nC).repeat(1, 1, self.nsample, 1) # translation normalization
        new_points = torch.cat([grouped_xyz, new_points], -1)
        
        '''Point nonlocal Cell''' #NL is true
        nf0, nf1, nf2 = new_feature.shape
        new_nonlocal_point = self.PointNonLocalCell(event_feature, new_feature.view(nf0, 1, nf1, nf2))

        '''skip connection'''
        skip_spartial, _ = torch.max(new_points, 2)
        skip_spartial = skip_spartial.permute(0,2,1)
        skip_spartial = self.ss_bn(self.skip_spatial(skip_spartial)).permute(0,2,1)

        '''Point Local Cell'''
        new_points = new_points.permute(0,3,1,2)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)), inplace=True) 
        new_points = new_points.permute(0,2,3,1)
        weight = self.weight_net_hidden(grouped_xyz)
        new_points = new_points.permute(0,1,3,2)
        new_points = torch.matmul(new_points, weight).permute(0,3,1,2)
        new_points = self.wc_bn(self.weighted_conv(new_points)).permute(0,2,3,1)

        new_points = torch.squeeze(new_points, 2) # (batch_size, npoints, mlp[-1])

        new_points = torch.add(new_points, skip_spartial)
        new_points = torch.add(new_points, new_nonlocal_point).permute(0,2,1)

        '''Feature Fushion'''
        new_points = self.ff_bn(self.feature_fushion(new_points))

        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points
