import sys
sys.path.append('/code/CompEvent/CSFL')

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from collections import OrderedDict
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, tensor2img
from basicsr.models.archs import define_network

import importlib
loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

import numpy as np
import os
from tqdm import tqdm
from basicsr.utils.img_util import imwrite
from basicsr.models.archs.CompEvent_arch import ComplexFusion_Model

class ModelManager(BaseModel):
    def __init__(self, opt):
        super(ModelManager, self).__init__(opt)

        if 'test' in opt['datasets']:
            self.num_frames = opt['datasets']['test'].get('num_test_video_frames', 3)
        elif 'train' in opt['datasets']:
            self.num_frames = opt['datasets']['train'].get('num_train_video_frames', 3)
        else:
            self.num_frames = 3
        self.middle_frame_idx = self.num_frames // 2

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            self.init_training_settings()

        if 'train' in opt:
            self.ms_lambda = opt['train'].get('ms_lambda', [1.0, 0.5])
        else:
            self.ms_lambda = [1.0, 0.5]

        self.logger = get_root_logger()

    def init_training_settings(self):

        self.net_g.train()
        train_opt = self.opt['train']

        if train_opt.get('pixel_opt'):
            self.pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, self.pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('lpips_opt'):
            lpips_type = train_opt['lpips_opt'].pop('type')
            cri_lpips_cls = getattr(loss_module, lpips_type)
            self.cri_lpips = cri_lpips_cls(**train_opt['lpips_opt']).to(self.device)
        else:
            self.cri_lpips = None

        if train_opt.get('freq_opt'):
            freq_type = train_opt['freq_opt'].pop('type')
            cri_freq_cls = getattr(loss_module, freq_type)
            self.cri_freq = cri_freq_cls(**train_opt['freq_opt']).to(self.device)
        else:
            self.cri_freq = None

        if train_opt.get('edge_opt'):
            edge_type = train_opt['edge_opt'].pop('type')
            cri_edge_cls = getattr(loss_module, edge_type)
            self.cri_edge = cri_edge_cls(**train_opt['edge_opt']).to(self.device)
        else:
            self.cri_edge = None

        if train_opt.get('ssim_opt'):
            ssim_type = train_opt['ssim_opt'].pop('type')
            cri_ssim_cls = getattr(loss_module, ssim_type)
            self.cri_ssim = cri_ssim_cls(**train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None

        if train_opt.get('structural_opt'):
            structural_type = train_opt['structural_opt'].pop('type')
            cri_structural_cls = getattr(loss_module, structural_type)
            self.cri_structural = cri_structural_cls(**train_opt['structural_opt']).to(self.device)
        else:
            self.cri_structural = None

        if train_opt.get('adaptive_ssim_opt'):
            adaptive_ssim_type = train_opt['adaptive_ssim_opt'].pop('type')
            cri_adaptive_ssim_cls = getattr(loss_module, adaptive_ssim_type)
            self.cri_adaptive_ssim = cri_adaptive_ssim_cls(**train_opt['adaptive_ssim_opt']).to(self.device)
        else:
            self.cri_adaptive_ssim = None

        if train_opt.get('progressive_opt'):
            progressive_type = train_opt['progressive_opt'].pop('type')
            cri_progressive_cls = getattr(loss_module, progressive_type)
            self.cri_progressive = cri_progressive_cls(**train_opt['progressive_opt']).to(self.device)
        else:
            self.cri_progressive = None

        if train_opt.get('multiscale_ssim_opt'):
            multiscale_ssim_type = train_opt['multiscale_ssim_opt'].pop('type')
            cri_multiscale_ssim_cls = getattr(loss_module, multiscale_ssim_type)
            self.cri_multiscale_ssim = cri_multiscale_ssim_cls(**train_opt['multiscale_ssim_opt']).to(self.device)
        else:
            self.cri_multiscale_ssim = None

        if train_opt.get('dynamic_balanced_opt'):
            dynamic_balanced_type = train_opt['dynamic_balanced_opt'].pop('type')
            cri_dynamic_balanced_cls = getattr(loss_module, dynamic_balanced_type)

            if 'loss_functions' in train_opt['dynamic_balanced_opt']:
                loss_functions_str = train_opt['dynamic_balanced_opt']['loss_functions']
                loss_functions_instances = []
                for loss_name in loss_functions_str:
                    loss_cls = getattr(loss_module, loss_name)

                    if loss_name == "CharbonnierLoss":
                        loss_instance = loss_cls(loss_weight=1.0, reduction='mean').to(self.device)
                    elif loss_name == "SSIMLoss":
                        loss_instance = loss_cls(loss_weight=1.0, window_size=11).to(self.device)
                    elif loss_name == "MultiScaleSSIMLoss":
                        loss_instance = loss_cls(loss_weight=1.0, scales=[1, 0.5, 0.25]).to(self.device)
                    elif loss_name == "L1Loss":
                        loss_instance = loss_cls(loss_weight=1.0, reduction='mean').to(self.device)
                    elif loss_name == "StructuralSimilarityLoss":
                        loss_instance = loss_cls(loss_weight=1.0, alpha=0.8, beta=0.2, window_size=11).to(self.device)
                    else:
                        loss_instance = loss_cls(loss_weight=1.0).to(self.device)
                    loss_functions_instances.append(loss_instance)
                train_opt['dynamic_balanced_opt']['loss_functions'] = loss_functions_instances
            
            self.cri_dynamic_balanced = cri_dynamic_balanced_cls(**train_opt['dynamic_balanced_opt']).to(self.device)
        else:
            self.cri_dynamic_balanced = None

        if train_opt.get('adaptive_combination_opt'):
            adaptive_combination_type = train_opt['adaptive_combination_opt'].pop('type')
            cri_adaptive_combination_cls = getattr(loss_module, adaptive_combination_type)

            if 'loss_functions' in train_opt['adaptive_combination_opt']:
                loss_functions_str = train_opt['adaptive_combination_opt']['loss_functions']
                loss_functions_instances = []
                for loss_name in loss_functions_str:
                    loss_cls = getattr(loss_module, loss_name)

                    if loss_name == "L1Loss":
                        loss_instance = loss_cls(loss_weight=1.0, reduction='mean').to(self.device)
                    elif loss_name == "SSIMLoss":
                        loss_instance = loss_cls(loss_weight=1.0, window_size=11).to(self.device)
                    elif loss_name == "StructuralSimilarityLoss":
                        loss_instance = loss_cls(loss_weight=1.0, alpha=0.8, beta=0.2, window_size=11).to(self.device)
                    else:
                        loss_instance = loss_cls(loss_weight=1.0).to(self.device)
                    loss_functions_instances.append(loss_instance)
                train_opt['adaptive_combination_opt']['loss_functions'] = loss_functions_instances
            
            self.cri_adaptive_combination = cri_adaptive_combination_cls(**train_opt['adaptive_combination_opt']).to(self.device)
        else:
            self.cri_adaptive_combination = None

        if train_opt.get('hierarchical_opt'):
            hierarchical_type = train_opt['hierarchical_opt'].pop('type')
            cri_hierarchical_cls = getattr(loss_module, hierarchical_type)

            if 'primary_loss' in train_opt['hierarchical_opt']:
                primary_loss_name = train_opt['hierarchical_opt']['primary_loss']

                if primary_loss_name == "DynamicBalancedLoss" and self.cri_dynamic_balanced is not None:
                    train_opt['hierarchical_opt']['primary_loss'] = self.cri_dynamic_balanced
                else:

                    primary_loss_cls = getattr(loss_module, primary_loss_name)
                    train_opt['hierarchical_opt']['primary_loss'] = primary_loss_cls(loss_weight=1.0).to(self.device)

            if 'auxiliary_losses' in train_opt['hierarchical_opt']:
                auxiliary_losses_str = train_opt['hierarchical_opt']['auxiliary_losses']
                auxiliary_losses_instances = []
                for loss_name in auxiliary_losses_str:
                    if loss_name == "AdaptiveLossCombination" and self.cri_adaptive_combination is not None:
                        auxiliary_losses_instances.append(self.cri_adaptive_combination)
                    else:
                        loss_cls = getattr(loss_module, loss_name)

                        if loss_name == "AdaptiveLossCombination":

                            loss_instance = loss_cls(
                                loss_functions=[
                                    loss_module.L1Loss(loss_weight=1.0, reduction='mean'),
                                    loss_module.SSIMLoss(loss_weight=1.0, window_size=11),
                                    loss_module.StructuralSimilarityLoss(loss_weight=1.0, alpha=0.8, beta=0.2, window_size=11)
                                ],
                                total_steps=200000
                            ).to(self.device)
                        else:
                            loss_instance = loss_cls(loss_weight=1.0).to(self.device)
                        auxiliary_losses_instances.append(loss_instance)
                train_opt['hierarchical_opt']['auxiliary_losses'] = auxiliary_losses_instances
            
            self.cri_hierarchical = cri_hierarchical_cls(**train_opt['hierarchical_opt']).to(self.device)
        else:
            self.cri_hierarchical = None

        if self.cri_pix is None:
            raise ValueError('pixel loss is None.')

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        optim_params_lowlr = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if k.startswith('module.offsets') or k.startswith('module.dcns'):
                    optim_params_lowlr.append(v)
                else:
                    optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio}],
                                                **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio}],
                                                **train_opt['optim_g'])

        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
 
    def feed_data(self, data):

        self.batch = {}
        self.batch['blur_input_clip'] = data['blur_input_clip'].to(self.device)
        self.batch['event_vox_clip'] = data['event_vox_clip'].to(self.device)
        self.batch['clean_middle'] = data['clean_middle'].to(self.device)
        if 'clean_gt_clip' in data:
            self.batch['clean_gt_clip'] = data['clean_gt_clip'].to(self.device)
        self.gt = self.batch['clean_middle']

        b, t, c_frame, h, w = self.batch['blur_input_clip'].shape
        _, _, c_event, _, _ = self.batch['event_vox_clip'].shape

        assert c_frame == 3, f"Expected 3 channels for blur images, got {c_frame}"
        assert c_event == 16, f"Expected 16 channels for event voxels, got {c_event}"
        assert self.batch['clean_middle'].shape[1] == 3, f"Expected 3 channels for GT images, got {self.batch['clean_middle'].shape[1]}"

    def backward_warp(self, x, flo):
        B, C, H, W = x.size()

        xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = torch.autograd.Variable(grid) + flo

        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)
        return output * mask

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        outputs = self.net_g(self.batch)
        if not isinstance(outputs, list):
            outputs = [outputs]
        self.output = outputs[-1]
        loss_total = 0
        loss_dict = OrderedDict()
        B, T = self.batch['blur_input_clip'].shape[:2]
        H, W = self.batch['blur_input_clip'].shape[-2:]

        output = self.output.view(B, T, 3, H, W)
        gt = self.batch['clean_gt_clip']

        l_rec = self.cri_pix(output, gt)
        loss_total += l_rec * self.opt['train'].get('lambda_rec', 1.0)
        loss_dict['l_rec'] = l_rec

        if self.cri_lpips is not None:
            output_flat = output.reshape(B * T, 3, H, W)
            gt_flat = gt.reshape(B * T, 3, H, W)
            l_lpips = self.cri_lpips(output_flat, gt_flat)
            loss_total += l_lpips * self.opt['train'].get('lambda_lpips', 1.0)
            loss_dict['l_lpips'] = l_lpips

        if self.cri_freq is not None:
            output_flat = output.reshape(B * T, 3, H, W)
            gt_flat = gt.reshape(B * T, 3, H, W)
            l_freq = self.cri_freq(output_flat, gt_flat)
            loss_total += l_freq * self.opt['train'].get('lambda_freq', 1.0)
            loss_dict['l_freq'] = l_freq

        if self.cri_edge is not None:

            l_edge = self.cri_edge(output, gt, self.batch['event_vox_clip'])
            loss_total += l_edge * self.opt['train'].get('lambda_edge', 1.0)
            loss_dict['l_edge'] = l_edge

        if self.cri_ssim is not None:
            output_flat = output.reshape(B * T, 3, H, W)
            gt_flat = gt.reshape(B * T, 3, H, W)
            l_ssim = self.cri_ssim(output_flat, gt_flat)
            loss_total += l_ssim * self.opt['train'].get('lambda_ssim', 1.0)
            loss_dict['l_ssim'] = l_ssim

        if self.cri_structural is not None:
            output_flat = output.reshape(B * T, 3, H, W)
            gt_flat = gt.reshape(B * T, 3, H, W)
            l_structural = self.cri_structural(output_flat, gt_flat)
            loss_total += l_structural * self.opt['train'].get('lambda_structural', 1.0)
            loss_dict['l_structural'] = l_structural

        if self.cri_adaptive_ssim is not None:
            output_flat = output.reshape(B * T, 3, H, W)
            gt_flat = gt.reshape(B * T, 3, H, W)
            l_adaptive_ssim = self.cri_adaptive_ssim(output_flat, gt_flat)
            loss_total += l_adaptive_ssim * self.opt['train'].get('lambda_adaptive_ssim', 1.0)
            loss_dict['l_adaptive_ssim'] = l_adaptive_ssim

        if self.cri_progressive is not None:
            output_flat = output.reshape(B * T, 3, H, W)
            gt_flat = gt.reshape(B * T, 3, H, W)
            l_progressive = self.cri_progressive(output_flat, gt_flat, current_iter=current_iter)
            loss_total += l_progressive * self.opt['train'].get('lambda_progressive', 1.0)
            loss_dict['l_progressive'] = l_progressive

        if self.cri_multiscale_ssim is not None:
            output_flat = output.reshape(B * T, 3, H, W)
            gt_flat = gt.reshape(B * T, 3, H, W)
            l_multiscale_ssim = self.cri_multiscale_ssim(output_flat, gt_flat)
            loss_total += l_multiscale_ssim * self.opt['train'].get('lambda_multiscale_ssim', 1.0)
            loss_dict['l_multiscale_ssim'] = l_multiscale_ssim

        if self.cri_dynamic_balanced is not None:
            output_flat = output.reshape(B * T, 3, H, W)
            gt_flat = gt.reshape(B * T, 3, H, W)
            l_dynamic_balanced = self.cri_dynamic_balanced(output_flat, gt_flat)
            loss_total += l_dynamic_balanced * self.opt['train'].get('lambda_dynamic_balanced', 1.0)
            loss_dict['l_dynamic_balanced'] = l_dynamic_balanced

        if self.cri_adaptive_combination is not None:
            output_flat = output.reshape(B * T, 3, H, W)
            gt_flat = gt.reshape(B * T, 3, H, W)
            l_adaptive_combination = self.cri_adaptive_combination(output_flat, gt_flat, current_iter=current_iter)
            loss_total += l_adaptive_combination * self.opt['train'].get('lambda_adaptive_combination', 1.0)
            loss_dict['l_adaptive_combination'] = l_adaptive_combination

        if self.cri_hierarchical is not None:
            output_flat = output.reshape(B * T, 3, H, W)
            gt_flat = gt.reshape(B * T, 3, H, W)
            l_hierarchical = self.cri_hierarchical(output_flat, gt_flat)
            loss_total += l_hierarchical * self.opt['train'].get('lambda_hierarchical', 1.0)
            loss_dict['l_hierarchical'] = l_hierarchical

        loss_total.backward()
        if self.opt['train'].get('use_grad_clip', True):
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.batch)
        self.net_g.train()

    def validation(self, dataloader, current_iter, tb_logger, save_img=True, swanlab_run=None):
        self.net_g.eval()

        assert 'val' in self.opt, "Missing 'val' section in config!"
        assert 'metrics' in self.opt['val'], "Missing 'metrics' in 'val' section!"
        assert 'crop_border' in self.opt['val'], "Missing 'crop_border' in 'val' section!" 
        assert 'save_freq' in self.opt['val'], "Missing 'save_freq' in 'val' section!"
        assert 'test_y_channel' in self.opt['val'], "Missing 'test_y_channel' in 'val' section!"

        metric_results = {'psnr': 0, 'ssim': 0} if self.opt['val']['metrics'] else {}
        loss_results = {}

        crop_border = self.opt['val']['crop_border']
        save_freq = self.opt['val']['save_freq']
        test_y_channel = self.opt['val']['test_y_channel']

        pbar = tqdm(total=len(dataloader), desc='Validation')

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_imgs = visuals['result']
            gt_imgs = visuals['gt']

            if sr_imgs.dim() == 5:
                sr_imgs = sr_imgs[0]
            if gt_imgs.dim() == 5:
                gt_imgs = gt_imgs[0]
            T = sr_imgs.shape[0]

            if save_img and current_iter % save_freq == 0:
                for t in range(T):
                    save_path = os.path.join(
                        self.opt['path']['visualization'],
                        f'iter{current_iter}_idx{idx}_frame{t}_sr.png'
                    )
                    imwrite(tensor2img(sr_imgs[t]), save_path)

            if 'psnr' in metric_results:
                for t in range(T):
                    metric_results['psnr'] += calculate_psnr(
                        tensor2img(sr_imgs[t]), tensor2img(gt_imgs[t]), crop_border=crop_border, test_y_channel=test_y_channel
                    )
            if 'ssim' in metric_results:
                for t in range(T):
                    metric_results['ssim'] += calculate_ssim(
                        tensor2img(sr_imgs[t]), tensor2img(gt_imgs[t]), crop_border=crop_border, test_y_channel=test_y_channel
                    )

            if hasattr(self, 'log_dict') and self.log_dict is not None:
                for k, v in self.log_dict.items():
                    if k not in loss_results:
                        loss_results[k] = 0.0
                    loss_results[k] += v
            pbar.update(1)
        pbar.close()
        log_dict = {}
        num_imgs = len(dataloader) * T
        for metric in metric_results:
            metric_results[metric] /= num_imgs
            log_dict[f'metrics/{metric}'] = metric_results[metric]
            log_dict[f'val/{metric}'] = metric_results[metric]
            if tb_logger is not None:
                tb_logger.add_scalar(f'metrics/{metric}', metric_results[metric], current_iter)

        for k, v in loss_results.items():
            loss_avg = v / len(dataloader)
            log_dict[f'loss/{k}'] = loss_avg
            if tb_logger is not None:
                tb_logger.add_scalar(f'loss/{k}', loss_avg, current_iter)

        if swanlab_run is not None:
            swanlab_run.log(log_dict, step=current_iter)
        self.net_g.train()
        return metric_results.get('psnr', 0), metric_results.get('ssim', 0)

    def get_current_visuals(self):
        def to_visual(x, name):
            if isinstance(x, (list, tuple)):
                x = x[0]
            if not torch.is_tensor(x):
                raise TypeError(f"{name} is not a tensor (got {type(x)})")
            if name == 'output':
                B, T = self.batch['blur_input_clip'].shape[:2]
                H, W = x.shape[-2:]
                x = x.view(B, T, 3, H, W)
                x = x[0]
            elif name == 'lq':
                B, T = x.shape[:2]
                x = x[0]
            elif name == 'gt':
                if x.dim() == 5:
                    x = x[0]
            else:
                raise ValueError(f"Unexpected tensor shape for {name}: {x.shape}")
            return x.detach().cpu()
        return {
            'lq': to_visual(self.batch['blur_input_clip'], 'lq'),
            'result': to_visual(self.output, 'output'),
            'gt': to_visual(self.batch['clean_gt_clip'], 'gt')
        }

    def save_network(self, net, net_label, current_iter=None):
        if current_iter is not None and current_iter != -1:
            save_filename = f'{net_label}_{current_iter}.pth'
        else:
            save_filename = f'{net_label}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        
        if isinstance(net, nn.DataParallel):
            net = net.module
        
        state_dict = net.state_dict()

        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        torch.save(state_dict, save_path)
        pass

    def load_network(self, net, load_path, strict=True):
        if load_path is None:
            return
        state_dict = torch.load(load_path, map_location=lambda storage, loc: storage)

        state_dict = {k: v for k, v in state_dict.items() if isinstance(k, str) and k is not None}

        is_ddp = isinstance(net, torch.nn.DataParallel) or isinstance(net, torch.nn.parallel.DistributedDataParallel)
        if len(state_dict) == 0:
            raise RuntimeError("state_dict is empty after filtering bad keys!")
        first_key = list(state_dict.keys())[0]
        if is_ddp:

            if not first_key.startswith('module.'):
                state_dict = {'module.' + k if not k.startswith('module.') else k: v for k, v in state_dict.items()}
        else:

            if first_key.startswith('module.'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict, strict=strict)

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

        if hasattr(self, 'output') and hasattr(self, 'batch'):
            from basicsr.utils import tensor2img, imwrite
            import os
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            vis_dir = self.opt['path']['visualization']
            os.makedirs(vis_dir, exist_ok=True)
            save_path = os.path.join(vis_dir, f'vis_{current_iter}.png')
            imwrite(sr_img, save_path)