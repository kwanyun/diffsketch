import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import json
import torchvision
import logging
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torch.cuda.amp import GradScaler


gt_feature_maps=[]
example_feature_maps=[]

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str, nargs="?",default="configs/train_config/feature-extraction-generated.yaml",help="path to the feature extraction config file")
    parser.add_argument("--ddim_eta",type=float,default=0.0,help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--H",type=int,default=512,help="image height, in pixel space")
    parser.add_argument("--W",type=int,default=512,help="image width, in pixel space")
    parser.add_argument("--C",type=int,default=4,help="latent channels")
    parser.add_argument("--f",type=int,default=8,help="downsampling factor" )
    parser.add_argument("--model_config",type=str,default="configs/stable-diffusion/v1-inference.yaml",help="path to config which constructs model" )
    parser.add_argument("--ckpt",type=str,default="./sd-v1-4.ckpt",help="path to checkpoint of model. may change to v2.1 ")
    parser.add_argument("--sample_layer", type=str, default='481',help="sampling layer splited with ','  ex.41,361,681")
    parser.add_argument("--precision",type=str,help="evaluate at this precision",choices=["full", "autocast"],default="autocast")
    
    opt = parser.parse_args()
    
    setup_config = OmegaConf.load("./configs/train_config/setup.yaml")
    model_config = OmegaConf.load(f"{opt.model_config}")
    exp_config = OmegaConf.load(f"{opt.config}")
    exp_path_root = setup_config.config.exp_path_root

    if exp_config.config.init_img != '':
        exp_config.config.seed = -1
        exp_config.config.prompt = ""
        exp_config.config.scale = 1.0
        
    seed = exp_config.config.seed 
    seed_everything(seed)

    sampling_layers = [int(sample) for sample in opt.sample_layer.split(",")]
    
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    save_feature_timesteps = exp_config.config.ddim_steps if exp_config.config.init_img == '' else exp_config.config.save_feature_timesteps
    scaler = GradScaler()

    outpath = f"{exp_path_root}/{exp_config.config.experiment_name}"

    callback_timesteps_to_save = [save_feature_timesteps]
    if os.path.exists(outpath):
        logging.warning("Experiment directory already exists, previously saved content will be overriden")
        if exp_config.config.init_img != '':
            with open(os.path.join(outpath, "args.json"), "r") as f:
                args = json.load(f)
            callback_timesteps_to_save = args["save_feature_timesteps"] + callback_timesteps_to_save

    predicted_samples_path = os.path.join(outpath, "predicted_samples")
    feature_maps_path = os.path.join(outpath, "feature_maps")
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(predicted_samples_path, exist_ok=True)
    os.makedirs(feature_maps_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)
    
    
    # save parse_args in experiment dir
    with open(os.path.join(outpath, "args.json"), "w") as f:
        args_to_save = OmegaConf.to_container(exp_config.config)
        args_to_save["save_feature_timesteps"] = callback_timesteps_to_save
        json.dump(args_to_save, f)

    assert exp_config.config.prompt is not None
    
    prompts = [exp_config.config.prompt]
    if isinstance(prompts, tuple):
        prompts = list(prompts)
    pivotal_condition = model.get_learned_conditioning(prompts)
    
    transform_image = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),  # Resize the image to 224x224
    torchvision.transforms.Normalize(           # Normalize the image using precomputed mean and std
        mean=[0.485, 0.456, 0.406],  # Mean values for RGB channels
        std=[0.229, 0.224, 0.225]    # Standard deviation values for RGB channels
    )])
    
    def ddim_sampler_callback_gt(pred_x0, xt, i):
        pass

    def ddim_sampler_callback_example(pred_x0, xt, i):
        pass



    def diffusion_sample(c=pivotal_condition,is_gt=False):
        with model.ema_scope():
            uc = model.get_learned_conditioning([""])
            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
            # z encode
            z_enc = torch.randn([1, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
            torch.save(z_enc, f"{outpath}/z_enc.pt")
            if is_gt:
                samples_ddim, _ = sampler.sample(S=exp_config.config.ddim_steps,
                            conditioning=c,
                            batch_size=1,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=exp_config.config.scale,
                            unconditional_conditioning=uc,
                            eta=opt.ddim_eta,
                            x_T=z_enc,
                            img_callback=ddim_sampler_callback_gt,
                            callback_ddim_timesteps=save_feature_timesteps,
                            outpath=outpath)
            else:
                samples_ddim, _ = sampler.sample(S=exp_config.config.ddim_steps,
                            conditioning=c,
                            batch_size=1,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=exp_config.config.scale,
                            unconditional_conditioning=uc,
                            eta=opt.ddim_eta,
                            x_T=z_enc,
                            img_callback=ddim_sampler_callback_example,
                            callback_ddim_timesteps=save_feature_timesteps,
                            outpath=outpath)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            if True:
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                sample_idx = 0
                for x_sample in x_image_torch[-2:]:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(os.path.join(sample_path, f"{sample_idx}.png"))
                    sample_idx += 1
            return x_samples_ddim
                    
                    
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    #prepare the feature to train fusing_diffusion_feature_for_sketch
    diffusion_sample(pivotal_condition)
    

if __name__ == "__main__":
    main()
