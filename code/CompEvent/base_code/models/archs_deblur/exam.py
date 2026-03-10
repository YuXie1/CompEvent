import torch
from torchvision import models
from thop import profile


if __name__ == '__main__':
    net = models.mobilenet_v2()
    inputs = torch.randn(1, 3, 224, 224)
    flops, params = profile(net, inputs=(inputs, ))
    print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
    print("params=", str(params/1e6)+'{}'.format("M"))
