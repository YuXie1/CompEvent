import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()

@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)

class L1Loss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class SRNLoss(nn.Module):

    def __init__(self):
        super(SRNLoss, self).__init__()  

    def forward(self, preds, target):

        gt1 = target
        B,C,H,W = gt1.shape
        gt2 = F.interpolate(gt1, size=(H // 2, W // 2), mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, size=(H // 4, W // 4), mode='bilinear', align_corners=False)

        l1 = mse_loss(preds[0] , gt3)
        l2 = mse_loss(preds[1] , gt2)
        l3 = mse_loss(preds[2] , gt1)

        return l1+l2+l3

class CharbonnierLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)

class WeightedTVLoss(L1Loss):

    def __init__(self, loss_weight=1.0):
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super(WeightedTVLoss, self).forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super(WeightedTVLoss, self).forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss

try:
    import lpips
    _has_lpips = True
except ImportError:
    _has_lpips = False

class LPIPSLoss(nn.Module):
    def __init__(self, loss_weight=1.0, net='vgg', reduction='mean'):
        super().__init__()
        assert _has_lpips, 'lpips 包未安装，请先 pip install lpips'
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.lpips_fn = lpips.LPIPS(net=net)
        self.lpips_fn.eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad = False

    def forward(self, pred, target, **kwargs):

        def norm(x):
            return x * 2 - 1 if x.max() <= 1.0 else x
        loss = self.lpips_fn(norm(pred), norm(target))
        if self.reduction == 'mean':
            return self.loss_weight * loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * loss.sum()
        else:
            return self.loss_weight * loss

class FreqLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, **kwargs):

        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        loss = torch.abs(pred_fft - target_fft)
        if self.reduction == 'mean':
            return self.loss_weight * loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * loss.sum()
        else:
            return self.loss_weight * loss

class EventEdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', threshold=0):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, pred, target, event_vox_clip, **kwargs):

        B, T, C, H, W = event_vox_clip.shape
        event_mid = event_vox_clip[:, T//2]

        event_count = event_mid.sum(dim=1)

        event_mask = (event_count > self.threshold).float()

        def sobel(x):

            if x.dim() == 5:
                B, T, C, H, W = x.shape
                x = x.reshape(B * T, C, H, W)
            C = x.shape[1]
            sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3).repeat(C,1,1,1)
            sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3).repeat(C,1,1,1)
            grad_x = F.conv2d(x, sobel_x, padding=1, groups=C)
            grad_y = F.conv2d(x, sobel_y, padding=1, groups=C)
            return torch.sqrt(grad_x**2 + grad_y**2)
        pred_edge = sobel(pred)
        target_edge = sobel(target)

        mask = event_mask.unsqueeze(1)
        if pred_edge.shape[0] != mask.shape[0]:

            if pred.dim() == 5:
                B, T, C, H, W = pred.shape
            else:
                raise RuntimeError('无法自动推断B,T')
            mask = mask.repeat(1, T, 1, 1)
            mask = mask.reshape(B * T, 1, H, W)
        loss = torch.abs(pred_edge - target_edge) * mask
        if self.reduction == 'mean':
            return self.loss_weight * loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * loss.sum()
        else:
            return self.loss_weight * loss

def gaussian_window(window_size, sigma):
    gauss = torch.Tensor([torch.exp(torch.tensor(-(x - window_size//2)**2/float(2*sigma**2))) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, window_size=11, size_average=True, reduction='mean'):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.window_size = window_size
        self.size_average = size_average
        self.reduction = reduction
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, **kwargs):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        ssim_value = _ssim(img1, img2, window, self.window_size, channel, self.size_average)

        loss = 1 - ssim_value
        
        if self.reduction == 'mean':
            return self.loss_weight * loss
        elif self.reduction == 'sum':
            return self.loss_weight * loss * img1.numel()
        else:
            return self.loss_weight * loss

class StructuralSimilarityLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=0.8, beta=0.2, window_size=11, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.ssim_loss = SSIMLoss(loss_weight=1.0, window_size=window_size)
        
    def forward(self, pred, target, **kwargs):

        ssim_loss = self.ssim_loss(pred, target)

        def sobel_edge(x):
            C = x.shape[1]
            sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3).repeat(C,1,1,1)
            sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3).repeat(C,1,1,1)
            grad_x = F.conv2d(x, sobel_x, padding=1, groups=C)
            grad_y = F.conv2d(x, sobel_y, padding=1, groups=C)
            return torch.sqrt(grad_x**2 + grad_y**2)
        
        pred_edge = sobel_edge(pred)
        target_edge = sobel_edge(target)
        edge_loss = F.l1_loss(pred_edge, target_edge)

        total_loss = self.alpha * ssim_loss + self.beta * edge_loss
        
        if self.reduction == 'mean':
            return self.loss_weight * total_loss
        elif self.reduction == 'sum':
            return self.loss_weight * total_loss * pred.numel()
        else:
            return self.loss_weight * total_loss

class AdaptiveLossWeight(nn.Module):
    def __init__(self, initial_weight=1.0, momentum=0.9, min_weight=0.1, max_weight=2.0):
        super().__init__()
        self.initial_weight = initial_weight
        self.momentum = momentum
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.register_buffer('current_weight', torch.tensor(initial_weight))
        self.register_buffer('loss_history', torch.zeros(100))
        self.register_buffer('step_count', torch.tensor(0))
        
    def update_weight(self, current_loss):

        idx = int(self.step_count.item()) % 100
        self.loss_history[idx] = current_loss.detach()
        self.step_count += 1
        
        if self.step_count > 10:

            recent_losses = self.loss_history[:min(10, int(self.step_count.item()))]
            loss_trend = torch.mean(recent_losses[-5:]) - torch.mean(recent_losses[:5])

            if loss_trend > 0:
                new_weight = self.current_weight * (1 - 0.01)
            else:
                new_weight = self.current_weight * (1 + 0.01)

            new_weight = torch.clamp(new_weight, self.min_weight, self.max_weight)
            self.current_weight = self.momentum * self.current_weight + (1 - self.momentum) * new_weight
            
        return self.current_weight

class AdaptiveSSIMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, window_size=11, adaptive=True, **kwargs):
        super().__init__()
        self.ssim_loss = SSIMLoss(loss_weight=1.0, window_size=window_size)
        self.adaptive_weight = AdaptiveLossWeight(initial_weight=loss_weight) if adaptive else None
        self.base_weight = loss_weight
        self.adaptive = adaptive
        
    def forward(self, pred, target, **kwargs):
        ssim_loss = self.ssim_loss(pred, target)
        
        if self.adaptive and self.adaptive_weight is not None:

            current_weight = self.adaptive_weight.update_weight(ssim_loss)
            return current_weight * ssim_loss
        else:
            return self.base_weight * ssim_loss

class ProgressiveLoss(nn.Module):
    def __init__(self, total_steps=200000, warmup_steps=10000):
        super().__init__()
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.ssim_loss = SSIMLoss(loss_weight=1.0)
        self.l1_loss = L1Loss(loss_weight=1.0)
        
    def forward(self, pred, target, current_step=0, **kwargs):

        progress = min(current_step / self.total_steps, 1.0)
        warmup_progress = min(current_step / self.warmup_steps, 1.0)

        if current_step < self.warmup_steps:
            l1_weight = 1.0
            ssim_weight = 0.1 * warmup_progress
        else:

            l1_weight = 1.0 - 0.5 * progress
            ssim_weight = 0.5 + 0.5 * progress
            
        l1_loss = self.l1_loss(pred, target)
        ssim_loss = self.ssim_loss(pred, target)
        
        total_loss = l1_weight * l1_loss + ssim_weight * ssim_loss
        return total_loss

class MultiScaleSSIMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, scales=[1, 0.5, 0.25]):
        super().__init__()
        self.loss_weight = loss_weight
        self.scales = scales
        self.ssim_losses = nn.ModuleList([
            SSIMLoss(loss_weight=1.0, window_size=11) for _ in scales
        ])
        
    def forward(self, pred, target, **kwargs):
        total_loss = 0
        for i, scale in enumerate(self.scales):
            if scale != 1:

                pred_scaled = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                pred_scaled = pred
                target_scaled = target
                
            loss = self.ssim_losses[i](pred_scaled, target_scaled)
            total_loss += loss * (scale ** 2)
            
        return self.loss_weight * total_loss / len(self.scales)

class DynamicBalancedLoss(nn.Module):
    def __init__(self, loss_functions, initial_weights, momentum=0.9, min_weight=0.1, max_weight=2.0):
        super().__init__()
        self.loss_functions = nn.ModuleList(loss_functions)
        self.initial_weights = torch.tensor(initial_weights)
        self.momentum = momentum
        self.min_weight = min_weight
        self.max_weight = max_weight

        self.register_buffer('current_weights', self.initial_weights.clone())
        self.register_buffer('loss_history', torch.zeros(len(loss_functions), 100))
        self.register_buffer('step_count', torch.tensor(0))
        
    def update_weights(self, current_losses):

        idx = int(self.step_count.item()) % 100
        self.loss_history[:, idx] = current_losses.detach()
        self.step_count += 1
        
        if self.step_count > 10:

            recent_losses = self.loss_history[:, :min(10, int(self.step_count.item()))]
            loss_means = torch.mean(recent_losses, dim=1)

            loss_stds = torch.std(recent_losses, dim=1)

            relative_importance = loss_means / (loss_means.sum() + 1e-8)
            stability_factor = 1.0 / (loss_stds + 1e-8)

            adjustment_factor = relative_importance * stability_factor
            adjustment_factor = adjustment_factor / (adjustment_factor.sum() + 1e-8)

            new_weights = self.initial_weights * adjustment_factor
            new_weights = torch.clamp(new_weights, self.min_weight, self.max_weight)

            self.current_weights = self.momentum * self.current_weights + (1 - self.momentum) * new_weights
            
        return self.current_weights
    
    def forward(self, pred, target, **kwargs):

        losses = []
        for loss_fn in self.loss_functions:
            loss = loss_fn(pred, target, **kwargs)
            losses.append(loss)
        
        losses_tensor = torch.stack(losses)

        current_weights = self.update_weights(losses_tensor)

        total_loss = torch.sum(current_weights * losses_tensor)
        
        return total_loss

class AdaptiveLossCombination(nn.Module):
    def __init__(self, loss_functions, total_steps=200000):
        super().__init__()
        self.loss_functions = nn.ModuleList(loss_functions)
        self.total_steps = total_steps
        self.register_buffer('loss_performance', torch.zeros(len(loss_functions)))
        self.register_buffer('step_count', torch.tensor(0))
        
    def forward(self, pred, target, current_step=0, **kwargs):

        losses = []
        for loss_fn in self.loss_functions:
            loss = loss_fn(pred, target, **kwargs)
            losses.append(loss)
        
        losses_tensor = torch.stack(losses)

        if self.step_count > 0:

            loss_improvement = self.loss_performance - losses_tensor
            self.loss_performance = 0.9 * self.loss_performance + 0.1 * losses_tensor
        else:
            self.loss_performance = losses_tensor
            loss_improvement = torch.zeros_like(losses_tensor)
        
        self.step_count += 1

        progress = current_step / self.total_steps
        
        if progress < 0.3:
            weights = torch.tensor([1.0, 0.1, 0.1])
        elif progress < 0.7:

            improvement_weights = torch.softmax(loss_improvement, dim=0)
            weights = torch.tensor([0.6, 0.3, 0.1]) * improvement_weights
        else:

            best_loss_idx = torch.argmin(self.loss_performance)
            weights = torch.zeros(len(losses))
            weights[best_loss_idx] = 1.0
        
        total_loss = torch.sum(weights * losses_tensor)
        return total_loss

class HierarchicalLossOptimization(nn.Module):
    def __init__(self, primary_loss, auxiliary_losses, primary_weight=1.0):
        super().__init__()
        self.primary_loss = primary_loss
        self.auxiliary_losses = nn.ModuleList(auxiliary_losses)
        self.primary_weight = primary_weight
        self.register_buffer('primary_loss_history', torch.zeros(50))
        self.register_buffer('step_count', torch.tensor(0))
        
    def forward(self, pred, target, **kwargs):

        primary_loss = self.primary_loss(pred, target, **kwargs)

        idx = int(self.step_count.item()) % 50
        self.primary_loss_history[idx] = primary_loss.detach()
        self.step_count += 1

        if self.step_count > 10:
            recent_primary_losses = self.primary_loss_history[:min(10, int(self.step_count.item()))]
            primary_stability = 1.0 / (torch.std(recent_primary_losses) + 1e-8)
        else:
            primary_stability = 1.0

        auxiliary_weight = torch.clamp(primary_stability, 0.0, 1.0)

        auxiliary_losses = []
        for aux_loss_fn in self.auxiliary_losses:
            aux_loss = aux_loss_fn(pred, target, **kwargs)
            auxiliary_losses.append(aux_loss)
        
        auxiliary_total = torch.stack(auxiliary_losses).mean()

        total_loss = self.primary_weight * primary_loss + auxiliary_weight * auxiliary_total
        
        return total_loss