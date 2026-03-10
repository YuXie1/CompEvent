import torch
import torch.nn as nn

import sys
sys.path.append("/code/CompEvent/CSFL")

class ComplexGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super().__init__()
        from basicsr.models.archs.CompEvent_arch import ComplexConv2d
        self.conv_z = ComplexConv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_r = ComplexConv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_h = ComplexConv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)

    def forward(self, x_real, x_imag, h_prev_real, h_prev_imag):
        xh_real = torch.cat([x_real, h_prev_real], dim=1)
        xh_imag = torch.cat([x_imag, h_prev_imag], dim=1)
        z_real, z_imag = self.conv_z(xh_real, xh_imag)
        z_real = torch.sigmoid(z_real)
        z_imag = torch.sigmoid(z_imag)
        r_real, r_imag = self.conv_r(xh_real, xh_imag)
        r_real = torch.sigmoid(r_real)
        r_imag = torch.sigmoid(r_imag)
        rh_real = r_real * h_prev_real
        rh_imag = r_imag * h_prev_imag
        xrh_real = torch.cat([x_real, rh_real], dim=1)
        xrh_imag = torch.cat([x_imag, rh_imag], dim=1)
        h_tilde_real, h_tilde_imag = self.conv_h(xrh_real, xrh_imag)
        h_tilde_real = torch.tanh(h_tilde_real)
        h_tilde_imag = torch.tanh(h_tilde_imag)
        h_real = (1 - z_real) * h_prev_real + z_real * h_tilde_real
        h_imag = (1 - z_imag) * h_prev_imag + z_imag * h_tilde_imag
        return h_real, h_imag

class ComplexBiGRU(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gru_fwd = ComplexGRUCell(input_channels, hidden_channels, kernel_size, padding)
        self.gru_bwd = ComplexGRUCell(input_channels, hidden_channels, kernel_size, padding)

    def forward(self, x_real_seq, x_imag_seq):

        B, T, C, H, W = x_real_seq.shape
        device = x_real_seq.device
        h_fwd_real = torch.zeros(B, self.hidden_channels, H, W, device=device)
        h_fwd_imag = torch.zeros_like(h_fwd_real)
        h_bwd_real = torch.zeros_like(h_fwd_real)
        h_bwd_imag = torch.zeros_like(h_fwd_real)
        outs_fwd_real, outs_fwd_imag = [], []
        outs_bwd_real, outs_bwd_imag = [], []

        for t in range(T):
            h_fwd_real, h_fwd_imag = self.gru_fwd(x_real_seq[:, t], x_imag_seq[:, t], h_fwd_real, h_fwd_imag)
            outs_fwd_real.append(h_fwd_real)
            outs_fwd_imag.append(h_fwd_imag)

        for t in reversed(range(T)):
            h_bwd_real, h_bwd_imag = self.gru_bwd(x_real_seq[:, t], x_imag_seq[:, t], h_bwd_real, h_bwd_imag)
            outs_bwd_real.insert(0, h_bwd_real)
            outs_bwd_imag.insert(0, h_bwd_imag)

        out_real = [torch.cat([f, b], dim=1) for f, b in zip(outs_fwd_real, outs_bwd_real)]
        out_imag = [torch.cat([f, b], dim=1) for f, b in zip(outs_fwd_imag, outs_bwd_imag)]
        out_real = torch.stack(out_real, dim=1)
        out_imag = torch.stack(out_imag, dim=1)
        return out_real, out_imag 