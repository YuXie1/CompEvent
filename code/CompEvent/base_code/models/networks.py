import models.archs_deblur.evsformer as EVSFormer
# import models.archs_deblur.evsformer2 as EVSFormer
import models.archs_deblur.EIFNet as EIFNet


import torch
import models.archs.PAN_arch as PAN_arch
import models.archs.PAN_event1_arch as PAN_event1_arch
import models.archs.PAN_event2_arch as PAN_event2_arch
import models.archs.PAN_event3_arch as PAN_event3_arch
import models.archs.PAN_event4_arch as PAN_event4_arch
import models.archs.PAN_event5_arch as PAN_event5_arch
import models.archs.eSL as eSL
import models.archs.e2sri as e2sri
import models.archs.dcsr as dcsr
import models.archs.DPT as DPT
import models.archs.spade_e2v as SPADE
import argparse
import options.options as option
import models.archs.TDAN_model as TDAN
import models.archs.SRResNet_arch as SRResNet_arch
import models.archs.RCAN_arch as RCAN_arch
import models.archs.EVSR_exam as EVSR
# import models.archs.PAN_event5_arch_group as PAN_event5_arch_group
# import models.archs.DDBPN as DDBPN


# Deblur
import models.archs_deblur.eSL_deblur as eSL_deblur
import models.archs_deblur.EVDI as EVDI
import models.archs_deblur.MemDeblur as MemDeblur
import models.archs_deblur.PAN_deblur as PAN_deblur
import models.archs_deblur.STRA1 as STRA1
# import models.archs_deblur.STRAHN1 as STRAHN1
import models.archs_deblur.WGWSNet as WGWSNet

import models.archs_deblur.CCZ as CCZ
import models.archs_deblur.ESD as ESD
# LOL_Blur
import models.archs_deblur.D2HNet as D2HNet
import models.archs_deblur.D2Net as D2Net
import models.archs_deblur.EFNet as EFNet
import models.archs_deblur.ERDN as ERDN
import models.archs_deblur.RED_Net as RED_Net
import models.archs_deblur.STFAN as STFAN
import models.archs_deblur.STRAHN_deblur as STRAHN_deblur
import models.archs_deblur.UEVD as UEVD
import models.archs_deblur.LEDVI as LEDVI


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

  


    ################################### deblur
    if which_model == 'eSL_Net_deblur':
        netG = eSL_deblur.eSL_Net_deblur(scale=opt_net['scale'])
    elif which_model == 'STRA1':
        netG = STRA1.STRA1(num_res=opt_net['num_res'])
    elif which_model == 'EVDI':
        netG = EVDI.EVDI(in_nc=opt_net['in_nc'],out_nc=opt_net['out_nc'],nf=opt_net['nf'],unf=opt_net['unf'],nb=opt_net['nb'],scale=opt_net['scale'])
    elif which_model == 'MemDeblur':
        netG = MemDeblur.MemDeblur(in_nc=opt_net['in_nc'],out_nc=opt_net['out_nc'],nf=opt_net['nf'],unf=opt_net['unf'],nb=opt_net['nb'],scale=opt_net['scale'])
    elif which_model == 'PAN_deblur':
        netG = PAN_deblur.PAN_deblur(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'STRANH1':
        netG = STRAHN1.STRAHN1(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    elif which_model == 'STRANH_visual':
        netG = STRAHN_visual.STRAHN_visual(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
        ##################################
    elif which_model == 'D2HNet':
        netG = D2HNet.D2HNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'D2Net':
        netG = D2Net.D2Net(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    elif which_model == 'EFNet':
        netG = EFNet.EFNet(in_nc=opt_net['in_nc'])
    elif which_model == 'ERDN':
        netG = ERDN.ERDN(in_channels=opt_net['in_nc'])
    elif which_model == 'RED_Net':
        netG = RED_Net.RED_Net(in_nc=opt_net['in_nc'])
    elif which_model == 'STFAN':
        netG = STFAN.STFAN_Net(input_channel=opt_net['in_nc'])
    elif which_model == 'STRAHN_deblur':
        netG = STRAHN_deblur.STRAHN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    elif which_model == 'UEVD':
        netG = UEVD.UEVD(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    elif which_model == 'LEDVI':
        netG = LEDVI.LEDVI(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    elif which_model == 'WGWSNet':
        netG = WGWSNet.WGWSNet(base_channel=24, num_res=6)


    # elif which_model == 'EVSFormer2':
    #     # netG = ESD.ESD(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])

    #     netG = EVSFormer2.EVSFormer(
    #         inp_channels=opt_net["inp_channels"],
    #         out_channels=opt_net["out_channels"],
    #         dim=opt_net["dim"],
    #         num_blocks=opt_net["num_blocks"],
    #         heads=opt_net["heads"],
    #         ffn_expansion_factor=opt_net["ffn_expansion_factor"],
    #         bias=opt_net["bias"],
    #         LayerNorm_type=opt_net["LayerNorm_type"],  ## Other option 'BiasFree'
    #         ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    #     )

    elif which_model == 'EVSFormer':
        # netG = ESD.ESD(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])

        netG = EVSFormer.EVSFormer(
            inp_channels=opt_net["inp_channels"],
            out_channels=opt_net["out_channels"],
            dim=opt_net["dim"],
            num_blocks=opt_net["num_blocks"],
            heads=opt_net["heads"],
            ffn_expansion_factor=opt_net["ffn_expansion_factor"],
            bias=opt_net["bias"],
            LayerNorm_type=opt_net["LayerNorm_type"],  ## Other option 'BiasFree'
            ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        )
    elif which_model == 'EIFNet':
        netG = EIFNet.Restoration(inChannels_img=3, inChannels_event=6,outChannels=3,args=None)
    ################################### self-supervised deblur
        
    elif which_model == 'ESD':
        # netG = ESD.ESD(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])

        netG = ESD.ESD()
    
    
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


