import sys

sys.path.append("model")
from argparse import Namespace
from invad import invad
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch.nn as nn

class invAD_grad_guide():
    
    def __init__(self, pretrained_path="runs_base/94.748_mvtec_20240912-094816/net_50.pth"):

        reso = 256
        in_chas = [256, 512, 1024]  # [64, 256, 512, 1024, 2048]
        out_cha = 64
        style_chas = [min(in_cha, out_cha) for in_cha in in_chas]
        in_strides = [2 ** (len(in_chas) - i - 1) for i in range(len(in_chas))]
        latent_channel_size = 16
        model_encoder = Namespace()
        model_encoder.name = 'timm_wide_resnet50_2'
        model_encoder.kwargs = dict(pretrained=False, checkpoint_path='/home/hulei/.cache/torch/hub/checkpoints/wide_resnet50_racm-8234f177.pth',
                                    strict=False, features_only=True, out_indices=[1, 2, 3])
        model_fuser = dict(
            type='Fuser', in_chas=in_chas, style_chas=style_chas, in_strides=[4, 2, 1], down_conv=True, bottle_num=1, conv_num=1, conv_type='conv', lr_mul=0.01)

        latent_spatial_size = reso // (2 ** 4)
        model_decoder = dict(in_chas=in_chas, style_chas=style_chas,
                            latent_spatial_size=latent_spatial_size, latent_channel_size=latent_channel_size,
                            blur_kernel=[1, 3, 3, 1], normalize_mode='LayerNorm',
                            lr_mul=0.01, small_generator=True, layers=[2] * len(in_chas))
        sizes = [reso // (2 ** (2 + i)) for i in range(len(in_chas))]
        # model_disor = dict(sizes=sizes, in_chas=in_chas)
        model_disor = None
        self.net = invad(model_encoder=model_encoder, model_fuser=model_fuser, model_decoder=model_decoder, model_disor=model_disor).cuda()
        self.net.load_state_dict(torch.load(pretrained_path))
        
        self.loss_f = nn.MSELoss(reduction='mean')

    
    def forward(self, imgs):
        
        feats, feats_pred = self.net(imgs)

        input1 = feats if isinstance(feats, list) else [feats]
        input2 = feats_pred if isinstance(feats_pred, list) else [feats_pred]
        loss = 0
            
        for in1, in2 in zip(input1, input2):
            
            loss += self.loss_f(in1, in2)
    
        return loss


if __name__ == "__main__":
    
    nets = invAD_grad_guide()
    bs = 2
    reso = 256
    x = torch.randn(bs, 3, reso, reso, requires_grad=True).cuda()
    loss = nets.forward(x)
    grads = torch.autograd.grad(loss, x)[0]
    a=1