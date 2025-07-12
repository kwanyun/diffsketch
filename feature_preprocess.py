import argparse
import os
import natsort
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from concurrent.futures import ProcessPoolExecutor
import shutil

def load_features(timestep, layer, features_dir):
    load_dir = os.path.join(features_dir, f"output_block_{layer}_out_layers_features_time_{timestep}.pt")
    feature = torch.load(load_dir)
    return feature.view(1, -1).cpu().numpy()  # Flatten feature

def process_timestep(timestep):
    timestep_features = [load_features(timestep, layer, features_dir) for layer in range(12)]  # All layers
    return np.concatenate(timestep_features, axis=1)  # Concatenate along axis 1


parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int)

opt = parser.parse_args()

features_dir = os.path.join(f'experiments_no_prompt_1000/random1000_t50_{opt.num}/feature_maps')
features_list = natsort.natsorted(os.listdir(features_dir))
original_features = []

print('Load original features in', features_dir,'...')
with ProcessPoolExecutor() as executor:
    original_features = list(executor.map(process_timestep, range(1, 982, 20)))

# Combine flattened feature vectors for all timesteps
flatten_dir = "Flatten_features"
os.makedirs(flatten_dir, exist_ok=True)
flattened_values = np.concatenate(original_features, axis=0)  # (100, 7045120)
np.save(os.path.join(flatten_dir, f"{opt.num}.npy"), flattened_values)
print('Saved in', os.path.join(flatten_dir, f'{opt.num}.npy'))
shutil.rmtree(features_dir)
