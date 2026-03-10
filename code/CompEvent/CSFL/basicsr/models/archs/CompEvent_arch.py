import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import sys
sys.path.append("/code/CompEvent/CSFL")
from basicsr.models.archs.ComplexBiGRU import ComplexBiGRU

class ComplexConv2d(nn.Module):
    
    def __init__(self, input_channels, output_channels,
             kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_real = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_imag = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, groups, bias)
    
    def forward(self, input_real, input_imag):
        assert input_real.shape == input_imag.shape

        return self.conv_real(input_real) - self.conv_imag(input_imag), self.conv_imag(input_real) + self.conv_real(input_imag)

class ComplexPixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
    
    def forward(self, x_real, x_imag):

        x_complex = torch.cat([x_real, x_imag], dim=1)
        x_upsampled = F.pixel_shuffle(x_complex, self.upscale_factor)

        channels = x_upsampled.shape[1] // 2
        x_real_upsampled = x_upsampled[:, :channels, :, :]
        x_imag_upsampled = x_upsampled[:, channels:, :, :]
        
        return x_real_upsampled, x_imag_upsampled

class ComplexPatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=16, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = ComplexConv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, bias=True)

    def forward(self, x_real, x_imag):
        x_real, x_imag = self.proj(x_real, x_imag)
        return x_real, x_imag

class ComplexPatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=16, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1
        self.proj = ComplexConv2d(embed_dim, out_chans, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=True)
：
    def forward(self, x_real, x_imag):
        x_real, x_imag = self.proj(x_real, x_imag)
        return x_real, x_imag

class ComplexPatchUnEmbed_for_upsample(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, out_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(embed_dim, out_dim * patch_size ** 2 * 2, kernel_size=1, bias=False)
        self.pixel_shuffle = ComplexPixelShuffle(patch_size)

    def forward(self, x_real, x_imag):
        x_real_proj = self.proj(x_real)
        x_imag_proj = self.proj(x_imag)
        x_real_upsampled, x_imag_upsampled = self.pixel_shuffle(x_real_proj, x_imag_proj)
        
        return x_real_upsampled, x_imag_upsampled

class ComplexDownSample(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = output_dim
        self.proj = ComplexConv2d(input_dim, input_dim * 2, kernel_size=2, stride=2)

    def forward(self, x_real, x_imag):
        x_real, x_imag = self.proj(x_real, x_imag)
        return x_real, x_imag

class ComplexFFN(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(ComplexFFN, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        self.conv_init = ComplexConv2d(dim, dim*2, 1)
        self.conv1_1 = ComplexConv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1,
                      groups=self.dim_sp)
        self.conv1_2 = ComplexConv2d(self.dim_sp, self.dim_sp, kernel_size=5, padding=2,
                      groups=self.dim_sp)
        self.conv1_3 = ComplexConv2d(self.dim_sp, self.dim_sp, kernel_size=7, padding=3,
                      groups=self.dim_sp)
        self.gelu = nn.GELU()
        self.conv_fina = ComplexConv2d(dim*2, dim, 1)

    def forward(self, x_real, x_imag):
        x_real, x_imag = self.conv_init(x_real, x_imag)
        x_real_list = list(torch.split(x_real, self.dim_sp, dim=1))
        x_imag_list = list(torch.split(x_imag, self.dim_sp, dim=1))
        
        x_real_list[1], x_imag_list[1] = self.conv1_1(x_real_list[1], x_imag_list[1])
        x_real_list[2], x_imag_list[2] = self.conv1_2(x_real_list[2], x_imag_list[2])
        x_real_list[3], x_imag_list[3] = self.conv1_3(x_real_list[3], x_imag_list[3])
        
        x_real = torch.cat(x_real_list, dim=1)
        x_imag = torch.cat(x_imag_list, dim=1)
        
        x_real = self.gelu(x_real)
        x_imag = self.gelu(x_imag)
        
        x_real, x_imag = self.conv_fina(x_real, x_imag)

        return x_real, x_imag

class ComplexTokenMixer_For_Local(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(ComplexTokenMixer_For_Local, self).__init__()
        self.dim = dim
        self.dim_sp = dim//2
        self.CDilated_1 = ComplexConv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, dilation=1, groups=self.dim_sp)
        self.CDilated_2 = ComplexConv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=2, dilation=2, groups=self.dim_sp)

    def forward(self, x_real, x_imag):
        x_real1, x_real2 = x_real.chunk(2, dim=1)
        x_imag1, x_imag2 = x_imag.chunk(2, dim=1)
        
        cd_real1, cd_imag1 = self.CDilated_1(x_real1, x_imag1)
        cd_real2, cd_imag2 = self.CDilated_2(x_real2, x_imag2)
        
        x_real = torch.cat([cd_real1, cd_real2], dim=1)
        x_imag = torch.cat([cd_imag1, cd_imag2], dim=1)

        return x_real, x_imag

class ComplexFourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=4):
        super(ComplexFourierUnit, self).__init__()
        self.groups = groups
        self.bn = nn.BatchNorm2d(out_channels)
        self.fdc = ComplexConv2d(in_channels, output_channels=out_channels * self.groups, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=True)
        self.fpe = ComplexConv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels, bias=True)

        self.weight = nn.ModuleList([
            ComplexConv2d(in_channels, self.groups, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)]
        )

    def forward(self, x_real, x_imag):

        x = x_real + 1j * x_imag
        batch, c, h, w = x.size()
        ffted = torch.fft.fft2(x, norm='ortho')
        ffted_real = torch.real(ffted)
        ffted_imag = torch.imag(ffted)

        ffted_real = self.bn(ffted_real)
        ffted_imag = self.bn(ffted_imag)

        fpe_real, fpe_imag = self.fpe(ffted_real, ffted_imag)
        ffted_real = fpe_real + ffted_real
        ffted_imag = fpe_imag + ffted_imag

        dy_weight_real , dy_weight_imag = self.weight[0](ffted_real , ffted_imag)
        dy_weight_real = self.weight[1](dy_weight_real)
        dy_weight_imag = self.weight[1](dy_weight_imag)

        ffted_real, ffted_imag = self.fdc(ffted_real, ffted_imag)
        ffted_real = ffted_real.view(batch, self.groups, c, h, -1)
        ffted_imag = ffted_imag.view(batch, self.groups, c, h, -1)

        ffted_real = torch.einsum('bgchw,bghw->bchw', ffted_real, dy_weight_real)
        ffted_imag = torch.einsum('bgchw,bghw->bchw', ffted_imag, dy_weight_imag)

        ffted_real = F.gelu(ffted_real)
        ffted_imag = F.gelu(ffted_imag)

        ffted = ffted_real + 1j * ffted_imag
        output = torch.fft.ifft2(ffted, norm='ortho')
        output_real = torch.real(output)
        output_imag = torch.imag(output)
        return output_real, output_imag

class ComplexTokenMixer_For_Gloal(nn.Module):
    def __init__(self, dim):
        super(ComplexTokenMixer_For_Gloal, self).__init__()
        self.dim = dim
        self.conv_init = ComplexConv2d(dim, dim*2, 1)
        self.gelu_init = nn.GELU()
        self.conv_fina = ComplexConv2d(dim*2, dim, 1)
        self.gelu_fina = nn.GELU()
        self.FFC = ComplexFourierUnit(self.dim*2, self.dim*2)

    def forward(self, x_real, x_imag):
        x_real, x_imag = self.conv_init(x_real, x_imag)
        x_real = self.gelu_init(x_real)
        x_imag = self.gelu_init(x_imag)
        x_real0, x_imag0 = x_real, x_imag
        x_real, x_imag = self.FFC(x_real, x_imag)
        x_real = x_real + x_real0
        x_imag = x_imag + x_imag0
        x_real, x_imag = self.conv_fina(x_real, x_imag)
        x_real = self.gelu_fina(x_real)
        x_imag = self.gelu_fina(x_imag)
        return x_real, x_imag

class ComplexMixer(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer_for_local=ComplexTokenMixer_For_Local,
            token_mixer_for_gloal=ComplexTokenMixer_For_Gloal,
    ):
        super(ComplexMixer, self).__init__()
        self.dim = dim
        self.mixer_local = token_mixer_for_local(dim=self.dim,)
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim,)

        self.ca_conv = ComplexConv2d(2*dim, dim, 1)
        self.ca = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            ComplexConv2d(2*dim, 2*dim//2, 1),
            nn.ReLU(inplace=True),
            ComplexConv2d(2*dim//2, 2*dim, 1),
            nn.Sigmoid()
        ])
        self.gelu = nn.GELU()
        self.conv_init = ComplexConv2d(dim, 2*dim, 1)

    def forward(self, x_real, x_imag):
        x_real, x_imag = self.conv_init(x_real, x_imag)
        x_real_list = list(torch.split(x_real, self.dim, dim=1))
        x_imag_list = list(torch.split(x_imag, self.dim, dim=1))
        x_local_real, x_local_imag = self.mixer_local(x_real_list[0], x_imag_list[0])
        x_gloal_real, x_gloal_imag = self.mixer_gloal(x_real_list[1], x_imag_list[1])
        x_real = torch.cat([x_local_real, x_gloal_real], dim=1)
        x_imag = torch.cat([x_local_imag, x_gloal_imag], dim=1)
        x_real = self.gelu(x_real)
        x_imag = self.gelu(x_imag)

        x_real_pool = self.ca[0](x_real)
        x_imag_pool = self.ca[0](x_imag)

        x_real_pool, x_imag_pool = self.ca[1](x_real_pool, x_imag_pool)

        x_real_pool = self.ca[2](x_real_pool)
        x_imag_pool = self.ca[2](x_imag_pool)

        x_real_pool, x_imag_pool = self.ca[3](x_real_pool, x_imag_pool)

        ca_weights_real = self.ca[4](x_real_pool)
        ca_weights_imag = self.ca[4](x_imag_pool)

        x_real = ca_weights_real * x_real
        x_imag = ca_weights_imag * x_imag

        x_real, x_imag = self.ca_conv(x_real, x_imag)
        return x_real, x_imag

class ComplexBlock(nn.Module):
    def __init__(
            self,
            dim,
            norm_layer=nn.BatchNorm2d,
            token_mixer=ComplexMixer,
    ):
        super(ComplexBlock, self).__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mixer = token_mixer(dim=self.dim)：
        self.ffn = ComplexFFN(dim=self.dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x_real, x_imag):
        copy_real, copy_imag = x_real, x_imag
        
        x_real = self.norm1(x_real)
        x_imag = self.norm1(x_imag)
        x_real, x_imag = self.mixer(x_real, x_imag)
        x_real = x_real * self.beta + copy_real
        x_imag = x_imag * self.beta + copy_imag

        copy_real, copy_imag = x_real, x_imag
        x_real = self.norm2(x_real)
        x_imag = self.norm2(x_imag)
        x_real, x_imag = self.ffn(x_real, x_imag)
        x_real = x_real * self.gamma + copy_real
        x_imag = x_imag * self.gamma + copy_imag

        return x_real, x_imag

class ComplexStage(nn.Module):
    def __init__(
            self,
            depth=int,
            in_channels=int,
    ) -> None:
        super(ComplexStage, self).__init__()
        self.blocks = nn.ModuleList([
                ComplexBlock(
                    dim=in_channels,
                    norm_layer=nn.BatchNorm2d,
                    token_mixer=ComplexMixer,
                )
            for index in range(depth)
        ])

    def forward(self, x_real, x_imag):
        for block in self.blocks:
            x_real, x_imag = block(x_real, x_imag)
        return x_real, x_imag

class ComplexFusion_Model(nn.Module):
    def __init__(self, in_channel=32, out_channel=3, kernel_size=3, padding=1):
        super(ComplexFusion_Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=kernel_size, padding=padding)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=kernel_size, padding=padding)
        self.conv6 = nn.Conv2d(16, out_channel, kernel_size=kernel_size, padding=padding)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):

        x = self.relu(self.conv1(input))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        return x

class ComplexBackbone(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, patch_size=1,
                 embed_dim=[48, 96, 192, 96, 48], depth=[2, 2, 2, 2, 2], embed_kernel_size=3,):
        super(ComplexBackbone, self).__init__()

        self.patch_size = patch_size

        self.image_proj = nn.Conv2d(3, 16, kernel_size=1, bias=True)

        self.fusion_net = ComplexFusion_Model(in_channel=6, out_channel=3, kernel_size=3, padding=1)

        self.complex_bigru = ComplexBiGRU(input_channels=16, hidden_channels=16, kernel_size=3, padding=1)

        self.patch_embed = ComplexPatchEmbed(patch_size=patch_size, in_chans=32,
                                      embed_dim=embed_dim[0], kernel_size=embed_kernel_size)
        self.layer1 = ComplexStage(depth=depth[0], in_channels=embed_dim[0],)
        self.skip1 = ComplexConv2d(embed_dim[1] + embed_dim[0], embed_dim[0], 1)
        self.downsample1 = ComplexDownSample(input_dim=embed_dim[0], output_dim=embed_dim[1],)
        self.layer2 = ComplexStage(depth=depth[1], in_channels=embed_dim[1],)
        self.skip2 = ComplexConv2d(embed_dim[2] + embed_dim[1], embed_dim[1], 1)
        self.downsample2 = ComplexDownSample(input_dim=embed_dim[1], output_dim=embed_dim[2],)
        self.layer3 = ComplexStage(depth=depth[2], in_channels=embed_dim[2],)
        self.upsample3 = ComplexPatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[2],
                                                   out_dim=embed_dim[3])
        self.layer8 = ComplexStage(depth=depth[3], in_channels=embed_dim[3],)
        self.upsample4 = ComplexPatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[3],
                                                   out_dim=embed_dim[4])
        self.layer9 = ComplexStage(depth=depth[4], in_channels=embed_dim[4],)
        self.patch_unembed = ComplexPatchUnEmbed(patch_size=patch_size, out_chans=out_chans,
                                          embed_dim=embed_dim[4], kernel_size=3)

    def forward(self, batch):
        if isinstance(batch, dict):
            x_frame = batch['blur_input_clip']
            x_event = batch['event_vox_clip']

            x_frame_proj = [self.image_proj(x_frame[:, i]) for i in range(x_frame.shape[1])]
            x_frame_proj = torch.stack(x_frame_proj, dim=1)

            x_real_bigru, x_imag_bigru = self.complex_bigru(x_frame_proj, x_event)

            B_gru, T_gru, C_gru, H_gru, W_gru = x_real_bigru.shape
            x_real = x_real_bigru.view(-1, C_gru, H_gru, W_gru)
            x_imag = x_imag_bigru.view(-1, C_gru, H_gru, W_gru)
            x_real, x_imag = self.patch_embed(x_real, x_imag)
            x_real, x_imag = self.layer1(x_real, x_imag)
            copy1_real, copy1_imag = x_real, x_imag
            x_real, x_imag = self.downsample1(x_real, x_imag)
            x_real, x_imag = self.layer2(x_real, x_imag)
            copy2_real, copy2_imag = x_real, x_imag
            x_real, x_imag = self.downsample2(x_real, x_imag)
            x_real, x_imag = self.layer3(x_real, x_imag)
            x_real, x_imag = self.upsample3(x_real, x_imag)
            x_real_concat = torch.cat([x_real, copy2_real], dim=1)
            x_imag_concat = torch.cat([x_imag, copy2_imag], dim=1)
            x_real, x_imag = self.skip2(x_real_concat, x_imag_concat)
            x_real, x_imag = self.layer8(x_real, x_imag)
            x_real, x_imag = self.upsample4(x_real, x_imag)
            x_real_concat = torch.cat([x_real, copy1_real], dim=1)
            x_imag_concat = torch.cat([x_imag, copy1_imag], dim=1)
            x_real, x_imag = self.skip1(x_real_concat, x_imag_concat)
            x_real, x_imag = self.layer9(x_real, x_imag)
            x_real, x_imag = self.patch_unembed(x_real, x_imag)
            network_output_combined = torch.cat([x_real, x_imag], dim=1)
            fusion_output = self.fusion_net(network_output_combined)

            original_input_3ch = batch['blur_input_clip'].reshape(-1, batch['blur_input_clip'].shape[2], batch['blur_input_clip'].shape[3], batch['blur_input_clip'].shape[4])
            final_output = original_input_3ch + fusion_output
            final_output = final_output.view(B_gru, T_gru, 3, H_gru, W_gru)
            return final_output
        else:

            x_frame, x_event = batch

            x_frame_proj = [self.image_proj(x_frame[:, i]) for i in range(x_frame.shape[1])]
            x_frame_proj = torch.stack(x_frame_proj, dim=1)
            x_real_bigru, x_imag_bigru = self.complex_bigru(x_frame_proj, x_event)
            B_gru, T_gru, C_gru, H_gru, W_gru = x_real_bigru.shape
            x_real = x_real_bigru.view(-1, C_gru, H_gru, W_gru)
            x_imag = x_imag_bigru.view(-1, C_gru, H_gru, W_gru)
            x_real, x_imag = self.patch_embed(x_real, x_imag)
            x_real, x_imag = self.layer1(x_real, x_imag)
            copy1_real, copy1_imag = x_real, x_imag
            x_real, x_imag = self.downsample1(x_real, x_imag)
            x_real, x_imag = self.layer2(x_real, x_imag)
            copy2_real, copy2_imag = x_real, x_imag
            x_real, x_imag = self.downsample2(x_real, x_imag)
            x_real, x_imag = self.layer3(x_real, x_imag)
            x_real, x_imag = self.upsample3(x_real, x_imag)
            x_real_concat = torch.cat([x_real, copy2_real], dim=1)
            x_imag_concat = torch.cat([x_imag, copy2_imag], dim=1)
            x_real, x_imag = self.skip2(x_real_concat, x_imag_concat)
            x_real, x_imag = self.layer8(x_real, x_imag)
            x_real, x_imag = self.upsample4(x_real, x_imag)
            x_real_concat = torch.cat([x_real, copy1_real], dim=1)
            x_imag_concat = torch.cat([x_imag, copy1_imag], dim=1)
            x_real, x_imag = self.skip1(x_real_concat, x_imag_concat)
            x_real, x_imag = self.layer9(x_real, x_imag)
            x_real, x_imag = self.patch_unembed(x_real, x_imag)
            network_output_combined = torch.cat([x_real, x_imag], dim=1)
            fusion_output = self.fusion_net(network_output_combined)
            original_input_3ch = x_frame.reshape(-1, x_frame.shape[2], x_frame.shape[3], x_frame.shape[4])
            final_output = original_input_3ch + fusion_output
            final_output = final_output.view(B_gru, T_gru, 3, H_gru, W_gru)
            return final_output

def CompEvent_t():
    return ComplexBackbone(
        patch_size=1,
        embed_dim=[32, 64, 128, 64, 32],
        depth=[1, 1, 2, 1, 1],
        embed_kernel_size=3
    )

def CompEvent_s():
    return ComplexBackbone(
        patch_size=1,
        embed_dim=[32, 64, 128, 64, 32],
        depth=[2, 2, 4, 2, 2],
        embed_kernel_size=3
    )

def CompEvent_m():
    return ComplexBackbone(
        patch_size=1,
        embed_dim=[32, 64, 128, 64, 32],
        depth=[4, 4, 8, 4, 4],
        embed_kernel_size=3
    )

def CompEvent_l():
    return ComplexBackbone(
        patch_size=1,
        in_chans=3,
        embed_dim=[32, 64, 128, 64, 32],
        depth=[6, 6, 10, 6, 6],
        embed_kernel_size=3
    ) 
