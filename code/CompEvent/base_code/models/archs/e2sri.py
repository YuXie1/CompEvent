import torch
import torch.nn
from .base import ConvBlock, DeconvBlock, ResnetBlock, UpBlock, DownBlock
import torch.nn as nn
BLOCK_PARAMS = {
    2: {
        'kernel_size': 6,
        'stride': 2,
        'padding': 2,
    },
    4: {
        'kernel_size': 8,
        'stride': 4,
        'padding': 2,
    },
    8: {
        'kernel_size': 12,
        'stride': 8,
        'padding': 2,
    }
}


class RNet_B(torch.nn.Module):
    def __init__(self,base_2_channels):
        super(RNet_B, self).__init__()
        stage = 5
        # base_2_channels = 64

        modules = [ResnetBlock(num_channels=base_2_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               activation='prelu') for _ in range(stage)]
        modules.append(ConvBlock(in_channels=base_2_channels,
                                 out_channels=base_2_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 activation='prelu'))
        self.rnet_b = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.rnet_b(x)


class RNet_C(torch.nn.Module):
    def __init__(self):
        super(RNet_C, self).__init__()
        stage = 5
        base_1_channels = 256
        base_2_channels = 64
        scale_factor = 2
        kernel_size = BLOCK_PARAMS[scale_factor]['kernel_size']
        stride = BLOCK_PARAMS[scale_factor]['stride']
        padding = BLOCK_PARAMS[scale_factor]['padding']

        modules = [ResnetBlock(num_channels=base_1_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               activation='prelu') for _ in range(stage)]
        modules.append(DeconvBlock(in_channels=base_1_channels,
                                   out_channels=base_2_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   activation='prelu'))
        self.rnet_c = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.rnet_c(x)


class RNet_D(torch.nn.Module):
    def __init__(self,base_1_channels,base_2_channels):
        super(RNet_D, self).__init__()
        stage = 5
        # base_1_channels = 256
        # base_2_channels = 64
        scale_factor = 2
        kernel_size = BLOCK_PARAMS[scale_factor]['kernel_size']
        stride = BLOCK_PARAMS[scale_factor]['stride']
        padding = BLOCK_PARAMS[scale_factor]['padding']

        modules = [ResnetBlock(num_channels=base_2_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               activation='prelu') for _ in range(stage)]
        modules.append(ConvBlock(in_channels=base_2_channels,
                                 out_channels=base_1_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 activation='prelu'))
        self.rnet_d = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.rnet_d(x)


class SRNet(torch.nn.Module):
    def __init__(self,scale=4,base1_channels=256,base2_channels=64):
        super(SRNet, self).__init__()
        self.scale = scale
        
        center_in_channels = 3
        side_in_channels = 3 * 2 + 2
        out_channels = 3
        base_1_channels = base1_channels
        base_2_channels = base2_channels
        sequence_size = 3

        self.first_conv = ConvBlock(in_channels=45,
                                                out_channels=center_in_channels,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                activation='prelu')
        # EFR
        # central stack
        self.center_event_rectifier = ConvBlock(in_channels=center_in_channels,
                                                out_channels=base_1_channels,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                activation='prelu')
        # other stacks
        self.side_event_rectifier = ConvBlock(in_channels=side_in_channels,
                                              out_channels=base_1_channels,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              activation='prelu')


        self.RNet_a = ConvBlock(in_channels=base_1_channels,
                                                 out_channels=base_2_channels,
                                                 kernel_size=1,
                                                 stride=1,
                                                 padding=0,
                                                 activation='prelu')
        # # SRNet
        # self.rnet_a = RNet_A()
        self.rnet_b = RNet_B(base2_channels)
        # self.rnet_c = RNet_C()
        self.rnet_d = RNet_D(base_1_channels=base1_channels,base_2_channels=base2_channels)

        # Mixer (mixes intermediate intensity outputs for final reconstruvtion)
        self.mixer = ConvBlock(in_channels=base2_channels,
                               out_channels=8,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        # for m in self.modules():
        #     classname = m.__class__.__name__
        #     if classname.find('Conv2d') != -1:
        #         torch.nn.init.kaiming_normal_(m.weight)
        #         if m.bias is not None:
        #             m.bias.data.zero_()

        #     elif classname.find('ConvTranspose2d') != -1:
        #         torch.nn.init.kaiming_normal_(m.weight)
        #         if m.bias is not None:
        #             m.bias.data.zero_()

        self.ps1=nn.PixelShuffle(2)
        self.ps2=nn.PixelShuffle(2)

        self.shu2=nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)
        self.endconv=nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.endconv2 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)    
    
        
    def forward(self, x,aggregate_event):
        # EFR

        x = self.first_conv(x)
        
        # central stack feature rectifier without optical flow
        state = self.center_event_rectifier(x)
        # feature rectifier with optical flow for the sequence of stacks except central stack
        # eaxh stack is compared to central stack for calculating optical flow
        # rectified_event_list = []
        # for side_stack, flow in zip(side_stack_list, flow_list):
        #     rectified_event_list.append(self.side_event_rectifier(torch.cat((central_stack, side_stack, flow), 1)))

        # SRNet
        intermediate_images = []

        rnet_a_out = self.RNet_a(state)
        # rnet_c_out = self.rnet_c(rectified_event)
        # e = rnet_a_out - rnet_c_out
        
        e = rnet_a_out
        rnet_b_out = self.rnet_b(e)

        hidden_state = rnet_a_out + rnet_b_out
        intermediate_images.append(hidden_state)
        state = self.rnet_d(hidden_state)

        # Mix
        # Final output intensity image reconstruction by mixing all intermediate outputs
        mix = torch.cat(intermediate_images, 1)
        output = self.mixer(mix)

        out=self.ps1(output)
        out=self.shu2(out)
        
        if self.scale == 4:
            out=self.ps2(out)
            out=self.endconv(out)
        elif self.scale == 2:
            
            out = self.endconv2(out)

        return out


class RNet_A(torch.nn.Module):
    def __init__(self):
        super(RNet_A, self).__init__()
        stage = 0
        scale_factor = 2
        base_1_channels = 256
        base_2_channels = 64

        self.rectified_event_feature = ConvBlock(in_channels=base_1_channels,
                                                 out_channels=base_2_channels,
                                                 kernel_size=1,
                                                 stride=1,
                                                 padding=0,
                                                 activation='prelu')

        # Hour-Glass (increase-decrease scale)
        # self.HG_block = self.make_HG_block()

        # self.HG_block = []
        
        # Initial HR recontstruction from stack
        self.union = ConvBlock(in_channels=base_2_channels * stage,
                               out_channels=base_2_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        

    def forward(self, x):
        # input to RNet-A is rectified events
        # we make features from the rectified events (ref)
        ref = self.rectified_event_feature(x)
        # hour_glass stages increaseing/decreasing the resolution
        hg_out_list = []
        # HG_block = self.make_HG_block()
        # for block in HG_block:
        #     ref = block(ref)
        #     hg_out_list.append(ref)
        # Rnet-A output
        # rnet_a_out = self.union(torch.cat(hg_out_list, 1))
        # return rnet_a_out
        return ref

