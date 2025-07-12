import torch
import torch.nn as nn
from archs.aggregation_network import AggregationNetwork
import torch.nn.functional as F

class Conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=False),
            nn.GroupNorm(4,ch_out),
            nn.GELU(),
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class Conv_res(nn.Module):
    def __init__(self,ch_in):
        super(Conv_res,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3,stride=1,padding=1,bias=False),
            nn.GroupNorm(16,ch_in),
            nn.GELU(),
        )
    def forward(self,x):
        c = self.conv(x)
        return c+x

class Fusion(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Fusion,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=False),
		    nn.GroupNorm(16,ch_out),
			nn.GELU(),
        )

    def forward(self,x):
        x = self.up(x)
        return x
    

class SketchGenerator(nn.Module):
    def __init__(self,num_layer_extracted=12,num_timesteps=13,use_VAE=True):
        super(SketchGenerator, self).__init__()
        
        #aggregation lower
        self.aggnet_lower = AggregationNetwork(
        projection_dim=960,
        num_norm_groups = 16,
        feature_dims=[1280, 1280, 1280, 1280, 1280, 1280,640,640,640],
        device="cuda",
        save_timestep=[0 for len_timestep in range(num_timesteps)],
        )
        
        #aggregation higher
        self.aggnet_higher = AggregationNetwork(
        projection_dim=384,
        feature_dims=[320, 320, 320],
        device="cuda",
        save_timestep=[0 for len_timestep in range(num_timesteps)],
        )

    
        self.num_layer_extracted = num_layer_extracted
        self.num_timesteps  =num_timesteps
        self.use_VAE = use_VAE
        self.num_vae_feat = 14
        
        #channel reduction for FFD
        if use_VAE:
            self.channel_reduction_layer= []
            self.conv_blocks_16=[]
            self.conv_blocks_8=[]
            self.conv_blocks_4=[]

            for _ in range(self.num_vae_feat-6):
                self.channel_reduction_layer.append(nn.Conv2d(512, 16, 1, stride=1).cuda())
                self.conv_blocks_16.append(Conv_block(16,16).cuda())
            for _ in range(self.num_vae_feat-11):
                self.channel_reduction_layer.append(nn.Conv2d(256, 8, 1, stride=1).cuda())
                self.conv_blocks_8.append(Conv_block(8,8).cuda())
            for _ in range(self.num_vae_feat-11):
                self.channel_reduction_layer.append(nn.Conv2d(128, 8, 1, stride=1).cuda())
                self.conv_blocks_4.append(Conv_block(8,8).cuda())


            self.fuse128 = Fusion(848,256).cuda()
            self.fuse256 = Fusion(304,128).cuda()
            self.fuse512 = Fusion(152,64).cuda()
        
            self.final = nn.Sequential(
            Conv_block(91,64),
            nn.Conv2d(64,1,1,1)
            )
        else:
            self.fuse128 = Fusion(768,256).cuda()
            self.fuse256 = Fusion(256,128).cuda()
            self.fuse512 = Fusion(128,64).cuda()
        
            self.final = nn.Sequential(
            Conv_block(64,64),
            nn.Conv2d(64,1,1,1)
            )

        self.final = self.final.cuda()

    def forward(self, input):
        y = []
        feats=[]
        for start_idx in range((len(input)-self.num_vae_feat-2)//self.num_layer_extracted):
            latent_feats = input[start_idx*self.num_layer_extracted:start_idx*self.num_layer_extracted+9]
            latent_feats = [torch.nn.functional.interpolate(latent_feat.squeeze(0), size=32, mode="bilinear") for latent_feat in latent_feats]
            latent_feats = torch.cat(latent_feats, dim=1)
            feats.append(latent_feats)  # feats[0].shape = torch.Size([2, 9600, 64, 64])

        feats = torch.stack(feats, dim=1)
        aggregated_features_low = self.aggnet_lower(torch.flip(feats, dims=(1,)).float().view(2, -1, 32, 32))

        feats=[torch.nn.functional.interpolate(aggregated_features_low, size=64, mode="bilinear")]
        for start_idx in range((len(input)-16)//self.num_layer_extracted):
            latent_feats = input[start_idx*self.num_layer_extracted+9:start_idx*self.num_layer_extracted+12]
            latent_feats = [torch.nn.functional.interpolate(latent_feat.squeeze(0), size=64, mode="bilinear") for latent_feat in latent_feats]
            latent_feats = torch.cat(latent_feats, dim=1)
            feats.append(latent_feats)  # feats[0].shape = torch.Size([2, 960, 64, 64])

        feats = torch.stack(feats, dim=1)
        aggregated_features = self.aggnet_higher(torch.flip(feats, dims=(1,)).float().view(2, -1, 64, 64))
        concat_features = torch.cat([aggregated_features[0],aggregated_features[1]], dim=0).unsqueeze(0)

        if self.use_VAE:
            for decoder_features in range(self.num_vae_feat):
                y.append(self.channel_reduction_layer[decoder_features](input[self.num_timesteps*self.num_layer_extracted+1+decoder_features]))

            cat64_2 = torch.cat([concat_features,self.conv_blocks_16[0](y[0]),self.conv_blocks_16[1](y[1]),self.conv_blocks_16[2](y[2]),
                                 self.conv_blocks_16[3](y[3]),self.conv_blocks_16[4](y[4])],dim=1)
            up128 = self.fuse128(cat64_2) 

            cat128 = torch.cat([up128,self.conv_blocks_16[5](y[5]),self.conv_blocks_16[6](y[6]),self.conv_blocks_16[7](y[7])],dim=1)
            up256 = self.fuse256(cat128) 
            
            cat256 = torch.cat([up256,self.conv_blocks_8[0](y[8]),self.conv_blocks_8[1](y[9]),self.conv_blocks_8[2](y[10])],dim=1)
            up512 = self.fuse512(cat256)

            cat512 = torch.cat([up512,self.conv_blocks_4[0](y[11]),self.conv_blocks_4[1](y[12]),self.conv_blocks_4[2](y[13]),input[-1]],dim=1)
            sketch = self.final(cat512)
        else:
            up128 = self.fuse128(concat_features)
            up256 = self.fuse256(up128) 
            up512 = self.fuse512(up256)
            sketch = self.final(up512)
        return sketch

        