import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.archs.pointasnl_utils import PointASNLSetAbstraction
# from pointnet2_utils import PointNetSetAbstraction

class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        # inchannel = 6 if normal_channel else 3
        # self.normal_channel = True
        inchannel = 6
        self.sa1 = PointASNLSetAbstraction(npoint=512, nsample = 32, in_channel = inchannel, mlp = [64,64,128]) # npoint, nsample, in_channel, mlp, as_neighbor
        # self.sa2 = PointASNLSetAbstraction(npoint=128, nsample = 64, in_channel = 128 + 3, mlp = [128,128,256])
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, mlp=[128,256,512], in_channel=128 + 3, group_all=True)
        # self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, mlp=[256,512,1024], in_channel=256 + 3, group_all=True)


    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # _, l3_points_res = self.sa3(l1_xyz, l1_points)
        # _, l3_points = self.sa4(l2_xyz, l2_points)

        # l3_points_res = l3_points_res.view(B, l3_points_res.shape[-2])
        # l3_points = l3_points.view(B, l3_points.shape[-2])
        
        # x = torch.cat((l3_points_res, l3_points), -1)

        # return x, l3_points     #x=[2,40],;l3_point=[2,1024]
        return l1_xyz,l1_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))