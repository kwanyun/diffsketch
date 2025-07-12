import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from PIL import Image 
import argparse
from joblib import dump


parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int)
parser.add_argument("--version", default=1,type=int)
parser.add_argument("--timesteps", default=50,type=int)

opt = parser.parse_args()

num_data = int(opt.num)

if opt.version==1:
    feature_dim = 32071680
else:
    feature_dim = 32071680
    
diffusion_timesteps = opt.timesteps

pca_dir = f"PCA_t{diffusion_timesteps}_{num_data}s"
os.makedirs(pca_dir, exist_ok=True)

# List of PCA components
PCA_component_list = [25]
combined_feat = np.empty((num_data*diffusion_timesteps, feature_dim), dtype=np.float16)

print('Loading for PCA')

# Loop through numbers
for num in range(num_data):
    flattened_values = np.load(f'Flatten_features/{num}.npy')
    assert flattened_values.shape == (diffusion_timesteps,feature_dim)
    combined_feat[num*diffusion_timesteps:num*diffusion_timesteps+diffusion_timesteps] = flattened_values

    if num % 10 == 0 and num:
        print(num)



assert combined_feat.shape == (diffusion_timesteps*num_data, feature_dim)


print('PCA start')
for n_components in PCA_component_list:
    pca_dir_component = os.path.join(pca_dir, str(n_components))
    os.makedirs(pca_dir_component, exist_ok=True)

    pca = PCA(n_components=n_components, svd_solver='randomized')
    reduced = pca.fit_transform(combined_feat)

    np.save(os.path.join(pca_dir_component, f"combined_{n_components}.npy"), reduced)

    dump(pca, os.path.join(pca_dir_component, f"pca_{n_components}.joblib"))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=np.array(range(len(reduced)))%diffusion_timesteps, cmap='coolwarm', linewidth=0.5)
    ax.set_xlabel('First Component')
    ax.set_ylabel('Second Component')
    plt.colorbar(scatter, ax=ax, label='Timestep Value', shrink=0.5)
    
    ax.set_title('PCA Result in 2D with timesteps')
    plt.savefig(os.path.join(pca_dir_component, "2D_PCA_combined_result_timestep.png"), dpi=500)
    plt.close()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=np.array(range(len(reduced)))%diffusion_timesteps, cmap='coolwarm', linewidth=0.5)
    ax.set_xlabel('First Component')
    ax.set_ylabel('Second Component')
    ax.set_zlabel('Third Component')
    fig.colorbar(scatter, ax=ax, label='Timestep', shrink=0.5)
    ax.set_title('PCA Result in 3D with timesteps')
    plt.savefig(os.path.join(pca_dir_component, "3D_PCA_combined_result_timestep.png"), dpi=500)
    plt.close()

    