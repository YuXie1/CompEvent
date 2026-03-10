import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import numpy as np
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, FSLoss, GradientLoss
import models.archs.pointasnl_cls
from scipy.spatial import KDTree

from models.archs.Event_Attention import PointASNLSetAbstraction
from models.archs.se_resnet import SEBottleneck
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


logger = logging.getLogger('base')


############################################################################
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

    return torch.tensor(points_all, dtype=torch.int64, device='cuda'), torch.tensor(dist_all, dtype=torch.float64,device='cuda')
    # return torch.tensor(points_all, dtype=torch.int64), torch.tensor(dist_all, dtype=torch.float64)   # points_all=[512,32]   dist_all=[512,32]


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
###############################################################################

class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        inchannel = 6
        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)

        # self.netG = networks.define_G(opt)

        # self.event_attention = PointASNLSetAbstraction(npoint=512, nsample = 32, in_channel = inchannel, mlp = [64,64,128])

        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)


        # print network
        # self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            self.loss_type = loss_type
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
                # self.cri_pix = nn.L1Loss()
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
                # self.cri_pix = nn.MSELoss()
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
                # self.cri_pix = CharbonnierLoss()
            elif loss_type == 'BCE':
                self.cri_pix = nn.BCELoss()
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):

        # self.var_L = data['LQ']
        # self.event = data['Event']
        # self.event_feature = data['Event_feature']
        # self.event_array = data['Event_array']
        # self.event_array_40 = data['Event_array_40']
        # self.real_H = data['GT']  # GT

        self.var_L = data['LQ'].to(self.device)
        self.event = data['Event'].to(self.device)
        # self.event_feature = data['Event_feature'].to(self.device)
        # self.event_array = data['Event_array'].to(self.device)
        # self.event_array_40 = data['Event_array_40'].to(self.device)
        self.real_H = data['GT'].to(self.device)  # GT
        
        # if need_GT:
        #     self.real_H = data['GT'].to(self.device)  # GT
            # self.real_H = data['GT']
    
    def mixup_data(self, x, y, alpha=1.0, use_cuda=True): 
        '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda''' 
        batch_size = x.size()[0] 
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1 
        index = torch.randperm(batch_size).cuda() if use_cuda else torch.randperm(batch_size) 
        mixed_x = lam * x + (1 - lam) * x[index,:] 
        mixed_y = lam * y + (1 - lam) * y[index,:] 
        return mixed_x, mixed_y
    
    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        
        '''add mixup operation'''
#         self.var_L, self.real_H = self.mixup_data(self.var_L, self.real_H)

        # pred, trans_feat = self.event_attention(self.event)        

        #######################################################
        #######################################################
        #######################################################
        LR_input = self.var_L
        # event_data=[2,512,3]表示事件数据点
        event_data = self.event
        # event_feature = self.event_feature
        # aggregate_event = self.event_attention(event_data.cpu(),event_feature.cpu())
        # aggregate_event=[2,3,512]
        # event_array = self.event_array
        # event_array_40 = self.event_array_40
        
        # aggregate_event = event_data
        # aggregate_event = (event_data,event_feature)  #[2,3,512], [2,128,512]
        #######################################################
        #######################################################
        #######################################################   
        # LR_img_event = torch.cat([LR_input,event_array,event_array_40],axis=1)
        # [2,45,128,128]
        # channel=3(img) + 2(event) + 40(event)
        self.fake_H = self.netG(LR_input, event_data)
        # self.fake_H = self.netG(LR_img_event, aggregate_event)
            
        if self.loss_type == 'fs':

            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H) + self.l_fs_w * self.cri_fs(self.fake_H, self.real_H)
        elif self.loss_type == 'grad':
            l1 = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
            lg = self.l_grad_w * self.gradloss(self.fake_H, self.real_H)
            l_pix = l1 + lg
        elif self.loss_type == 'grad_fs':
            l1 = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
            lg = self.l_grad_w * self.gradloss(self.fake_H, self.real_H)
            lfs = self.l_fs_w * self.cri_fs(self.fake_H, self.real_H)
            l_pix = l1 + lg + lfs
        else:
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        if self.loss_type == 'grad':
            self.log_dict['l_1'] = l1.item()
            self.log_dict['l_grad'] = lg.item()
        if self.loss_type == 'grad_fs':
            self.log_dict['l_1'] = l1.item()
            self.log_dict['l_grad'] = lg.item()
            self.log_dict['l_fs'] = lfs.item()

    def test(self):
        self.netG.eval()
        
        LR_input = self.var_L
        # event_array = self.event_array
        # event_array_40 = self.event_array_40
        event_data = self.event
        # print('LR_input=',LR_input.shape)
        # print('event_array=',event_data.shape)
        # print('event_array_40=',event_array_40.shape)
        # torch_resize = Resize([180,240])
        # event_array = torch_resize(event_array)
        # event_array_40 = torch_resize(event_array_40)
        # print('event_array=',event_array.shape)
        # print('event_array_40=',event_array_40.shape)
        # LR_img_event = torch.cat([LR_input,event_array,event_array_40],axis=1)
        

        with torch.no_grad():
            self.fake_H = self.netG(LR_input, event_data)
            # self.fake_H = self.netG(LR_img_event,None)
        self.netG.train()

    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # ret = torch.Tensor(tfnp)
            
            

            return ret

        lr_list = [self.var_L]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        with torch.no_grad():
            sr_list = [self.netG(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
    
#     def load(self):
#         load_path_G_1 = self.opt['path']['pretrain_model_G_1']
#         load_path_G_2 = self.opt['path']['pretrain_model_G_2']
#         load_path_Gs=[load_path_G_1, load_path_G_2]
        
#         load_path_G = self.opt['path']['pretrain_model_G']
#         if load_path_G is not None:
#             logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
#             self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
#         if load_path_G_1 is not None:
#             logger.info('Loading model for 3net [{:s}] ...'.format(load_path_G_1))
#             logger.info('Loading model for 3net [{:s}] ...'.format(load_path_G_2))
#             self.load_network_part(load_path_Gs, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
