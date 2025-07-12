import argparse, os
import torch
import math
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
import joblib
import random


from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from models.generator import SketchGenerator
#from models.generator import SketchGenerator

from models.hed import estimate as extractor
from models.xdog import xdog_garygrossi as extractor_xdog
from models.informative import estimate as extractor_info

from torch.cuda.amp import GradScaler
import torch.nn.functional as F

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
    parser.add_argument("--config",type=str, nargs="?",default="configs/train_config/1-horse.yaml",help="path to the feature extraction config file")
    parser.add_argument("--ddim_eta",type=float,default=0.0,help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--H",type=int,default=512,help="image height, in pixel space")
    parser.add_argument("--W",type=int,default=512,help="image width, in pixel space")
    parser.add_argument("--C",type=int,default=4,help="latent channels")
    parser.add_argument("--f",type=int,default=8,help="downsampling factor" )
    parser.add_argument("--model_config",type=str,default="configs/stable-diffusion/v1-inference.yaml",help="path to config which constructs model" )
    parser.add_argument("--ckpt",type=str,default="./sd-v1-4.ckpt",help="path to checkpoint of model. change to v2.1 if needed")
    parser.add_argument("--sample_layer", type=str, default='1,61,161,241,321,421,501,561,641,701,781,861,941',help="sampling layer splited with ','  ex.41,361,681")
    parser.add_argument("--precision",type=str,help="evaluate at this precision",choices=["full", "autocast"],default="autocast")
    parser.add_argument("--exp_path_root",type=str,help="exp_path_root")
    parser.add_argument("--extractor",type=str,help="type of extractor", choices=["hed", "xdog","anime_style","opensketch_style","contour_style"],default="hed")
    parser.add_argument("--iter",type=int,default=200,help="iteration")
    parser.add_argument("--hiter",type=int,default=0,help="h_iteration_until")
    parser.add_argument("--style_pt",type=str,default="a pho_xdog_iter_1350.pt",help="path to checkpoint of generator.")
    
    opt = parser.parse_args()
    
    model_config = OmegaConf.load(f"{opt.model_config}")
    exp_config = OmegaConf.load(f"{opt.config}")
    exp_path_root = opt.exp_path_root

    if exp_config.config.init_img != '':
        exp_config.config.seed = -1
        exp_config.config.prompt = ""
        exp_config.config.scale = 1.0
        
    seed = exp_config.config.seed 
    seed_everything(0)

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

    random_real_path = os.path.join(outpath, "random_real")
    feature_maps_path = os.path.join(outpath, "feature_maps")
    generated_sketch_path = os.path.join(outpath, "generated_sketch")
    gt_path = os.path.join(outpath, "gt")
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(random_real_path, exist_ok=True)
    os.makedirs(feature_maps_path, exist_ok=True)
    os.makedirs(generated_sketch_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)
    
    # save parse_args in experiment dir
    with open(os.path.join(outpath, "args.json"), "w") as f:
        args_to_save = OmegaConf.to_container(exp_config.config)
        args_to_save["save_feature_timesteps"] = callback_timesteps_to_save
        json.dump(args_to_save, f)
    
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
        extract_feature_maps_gt(unet_model.output_blocks , i, "output_block")
        if i < 1000 // exp_config.config.ddim_steps - 1:
            x_samples_ddim, extracted_features_from_decoder= model.decode_first_stage_with_features(pred_x0)
            gt_feature_maps.extend(extracted_features_from_decoder)

    def ddim_sampler_callback_example(pred_x0, xt, i):
        #save_feature_maps_callback(i)  ## SAVE FEATURE MAP##
        extract_feature_maps_example(unet_model.output_blocks , i, "output_block")
        if i < 1000 // exp_config.config.ddim_steps - 1 :
            x_samples_ddim, extracted_features_from_decoder= model.decode_first_stage_with_features(pred_x0)
            example_feature_maps.extend(extracted_features_from_decoder)

    def extract_feature_maps_gt(blocks, i, feature_type="input_block"):
        if i in sampling_layers:
            for block in blocks:
                if "ResBlock" in str(type(block[0])):
                    #concat_features = torch.cat([block[0].out_layers_features[0],block[0].out_layers_features[1]], dim=0)
                    gt_feature_maps.append(block[0].out_layers_features.unsqueeze(0))

    def extract_feature_maps_example(blocks, i, feature_type="input_block"):
        if i in sampling_layers:
            for block in blocks:
                if "ResBlock" in str(type(block[0])):
                    #concat_features = torch.cat([block[0].out_layers_features[0],block[0].out_layers_features[1]], dim=0)
                    example_feature_maps.append(block[0].out_layers_features.unsqueeze(0))

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block in tqdm(blocks, desc="Saving input blocks feature maps"):
            if "ResBlock" in str(type(block[0])):
                if i in sampling_layers:
                    save_feature_map(block[0].out_layers_features, f"{feature_type}_{block_idx}_out_layers_features_time_{i}")
            block_idx += 1

    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks , i, "output_block")

    def save_feature_map(feature_map, filename):
        save_path = os.path.join(feature_maps_path, f"{filename}.pt")
        torch.save(feature_map, save_path)

    def diffusion_sample(c=pivotal_condition,is_gt=False):
        with model.ema_scope():
            uc = model.get_learned_conditioning([""])
            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
            # z encode
            z_enc = torch.randn([1, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

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

            return x_samples_ddim

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
            
    iteration = opt.iter
    h_iter = opt.hiter

    generator = SketchGenerator()
    generator.to(device)
    
    generator = torch.load(opt.style_pt,map_location=device)

    generator.eval()

    mean_pca = np.load('weight/mean_pca.npy').squeeze()
    cov_pca = np.load('weight/cov_pca.npy')
    pca = joblib.load('weight/pca_model.joblib')

    samples = np.random.multivariate_normal(mean_pca, cov_pca, iteration)

    for i in range(iteration):
        
        sampling_layers= sorted(random.sample(range(50), 13))
        sampling_layers = [sv*20+1 for sv in sampling_layers]

        example_feature_maps=[]
        sample = pca.inverse_transform(samples[i])
        samples_condition = torch.FloatTensor(sample)
        rand_condition = samples_condition.reshape(1,77,768).float().to(device)
        
        if i < h_iter :
            alpha = math.sqrt(1 - (i / h_iter))
            beta = math.sqrt(i / h_iter)
            rand_condition = (alpha / (alpha + beta)) * pivotal_condition + (beta / (alpha + beta)) * rand_condition
        else:
            rand_condition = rand_condition
        
        with torch.no_grad():
            with precision_scope("cuda"):
                rand_image = diffusion_sample(c=rand_condition,is_gt=False)
                if i%10==0:
                    print("Iteration:", i)
                if opt.extractor == "hed":
                    gt_sketch_to_save = extractor(rand_image.squeeze()) 
                elif opt.extractor == "xdog":
                    gt_sketch_to_save = extractor_xdog(rand_image.squeeze())
                else :
                    assert (opt.extractor in ['anime_style','contour_style']), 'Wrong sketch style for informative sketch'
                    gt_sketch_to_save = extractor_info(rand_image.squeeze(),opt.extractor)
                
                torchvision.utils.save_image(gt_sketch_to_save, os.path.join(gt_path, str(i).zfill(4) + ".png"))
                torchvision.utils.save_image(rand_image, os.path.join(random_real_path, str(i).zfill(4) + ".png"))
                sketch_gen = generator(example_feature_maps)

                torchvision.utils.save_image(sketch_gen[0], os.path.join(generated_sketch_path, str(i).zfill(4) + ".png"))

if __name__ == "__main__":
    main()