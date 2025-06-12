import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
import glob
import PIL
import sys
sys.path.append('textual_inversion-main')

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import random
from mask_transfor import random_rorate,merge_images,little_rorate_and_move,random_3direcions_rotate,visa_candle,radomreset
import torch.nn as nn

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a photo of *",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="textual_inversion-main/output/LD_mvtec_mask/srecw"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="textual_inversion-main/models/model.ckpt", 
        help="Path to pretrained ldm text2img model")

    parser.add_argument(
        "--embedding_path", 
        type=str, 
        help="Path to a pre-trained embedding manager checkpoint",
        default="textual_inversion-main/logs/screw2024-09-07T17-50-29_LD_mvtec_addmask/checkpoints/embeddings_gs-9999.pt")

    parser.add_argument(
        "--class_layer_path", 
        type=str, 
        help="Path to a pre-trained class_layer checkpoint",
        default="textual_inversion-main/logs/screw2024-09-07T17-50-29_LD_mvtec_addmask/checkpoints/mask_linear-9999.pt")
    
    parser.add_argument(
        "--conference_mask_path", 
        type=str, 
        help="Path to conference masks folder",
        default="SAM/data/mvtec_conference/screw")

    opt = parser.parse_args()
    
    mask_images = glob.glob(opt.conference_mask_path+"/*.png")
    class_name = opt.conference_mask_path.split("/")[-1]

    config = OmegaConf.load("textual_inversion-main/configs/latent-diffusion/generate_with_mask.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, opt.ckpt_path)  # TODO: check path
    model.embedding_manager.load(opt.embedding_path)
    

    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    mask_linear_model = nn.Sequential(
            nn.Linear(128, 200),
            nn.ReLU())
    
    mask_linear_model.load_state_dict(torch.load(opt.class_layer_path,map_location=device))
    mask_linear_model.cuda()
    mask_linear_model.eval()
    model = model.to(device)
    mask_linear_model = mask_linear_model.to(device)
    
    

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    all_masks = []
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_samples * [""])
                
            
            for n in trange(opt.n_iter, desc="Sampling"):
                
                #获取mask
                
                masks = []
                
                for i in range(opt.n_samples):
                    
                    idx = random.randint(0,len(mask_images)-1)
                    mask_image = Image.open(mask_images[idx])
                    if not mask_image.mode == "RGB":
                        mask_image = mask_image.convert("RGB")
                    
                    #TODO 对mask进行变换
                    
                    if class_name in ["metal_nut","screw","fryum","hazelnut"]:
                        mask_image = random_rorate(mask_image)
                        
                    elif class_name == "transistor":
                        mask_image = little_rorate_and_move(mask_image)

                    elif class_name == "cashew":
                        mask_image = little_rorate_and_move(mask_image,angles=10)

                    elif class_name == "chewinggum":
                        mask_image = little_rorate_and_move(mask_image,distance=0.1,tranpose=True)
                        
                    elif class_name == "candle":
                        mask_image = visa_candle(mask_image)
                    
                    elif class_name == "macaroni1":
                        
                        mask_image = visa_candle(mask_image,rotate=True)
                    
                    elif class_name in ["grid"]:
                        
                        mask_image = random_3direcions_rotate(mask_image)
                    
                    elif class_name == "macaroni2":
                        
                        mask_image = radomreset(mask_image)
                    
                    elif class_name in ["bottle","cable","capsule","carpet","leather","pill","toothbrush","wood","tile"]:
                        
                        pass
                    
                    else:
                        raise ValueError("输入类别无效")
                
                    mask_image = mask_image.resize((opt.H, opt.W), resample=PIL.Image.BICUBIC)
                    all_masks.append(mask_image)
                    mask_image = np.array(mask_image).astype(np.uint8)
                    mask_image = (mask_image / 127.5 - 1.0).astype(np.float32)
                    masks.append(mask_image)
                    
                
                masks = np.array(masks)
                masks=masks[:opt.n_samples]
                masks = torch.tensor(masks).to(torch.cuda.current_device())
                
                masks = rearrange(masks, 'b h w c -> b c h w')
                masks = masks.to(memory_format=torch.contiguous_format).float()
                

                
                encoder_mask = model.encode_first_stage(masks)
                z_mask = model.get_first_stage_encoding(encoder_mask).detach()
                c = model.get_learned_conditioning(opt.n_samples * [prompt])
                
                z_mask = z_mask.view(z_mask.shape[0],-1,128)
                    
                z_mask = mask_linear_model(z_mask)
                z_mask = z_mask.view(z_mask.shape[0],-1,c.shape[2])
                c = torch.cat([c,z_mask],dim=1)
                
                
                
                shape = [4, opt.H//8, opt.W//8]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.jpg"))
                    base_count += 1
                all_samples.append(x_samples_ddim)


    # additionally, save as grid

    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=opt.n_samples)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.jpg'))
    
    merged_image = merge_images(all_masks, opt.n_samples)
    merged_image.save(os.path.join(outpath, f'{prompt.replace(" ", "-")}_mask.jpg'))

    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")
