import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from PIL import Image 
import argparse
import joblib
from joblib import dump

device = "cuda" if torch.cuda.is_available() else "cpu"


#cluster with dino or manually
Cluster1 = ['00006', '00012', '00047', '00062', '00069', '00082']
Cluster2 = ['00015', '00017', '00024', '00037', '00038', '00039', '00042', '00044', '00055', '00076']
Cluster3 = ['00003', '00011', '00016', '00025', '00026', '00036', '00043', '00045', '00053', '00086', '00095', '00098']
Cluster4 = ['00001', '00004', '00007', '00033', '00072', '00073', '00083', '00097']
Cluster5 = ['00009', '00014', '00023', '00067', '00074', '00092', '00094']
Cluster6 = ['00008', '00027', '00048', '00051', '00057', '00060', '00071', '00079', '00091']
Cluster7 = ['00020', '00022', '00029', '00035', '00068', '00077', '00078', '00081', '00084']
Cluster8 = ['00000', '00002', '00018', '00021', '00031', '00032', '00034', '00059', '00088']
Cluster9 = ['00028', '00066', '00080', '00087', '00089']
Cluster10 = ['00030', '00065', '00093']
Cluster11 = ['00005', '00046', '00049', '00050', '00052', '00054', '00056', '00063', '00070', '00085', '00090']
Cluster12=['00010', '00013', '00041', '00058', '00061', '00064', '00096', '00099']

class_lists = {
    "Cluster1": Cluster1,
    "Cluster2": Cluster2,
    "Cluster3": Cluster3,
    "Cluster4": Cluster4,
    "Cluster5": Cluster5,
    "Cluster6": Cluster6,
    "Cluster7": Cluster7,
    "Cluster8": Cluster8,
    "Cluster9": Cluster9,
    "Cluster10": Cluster10,
    "Cluster11": Cluster11,
    "Cluster12": Cluster12,
}

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int)
parser.add_argument("--classes", type=str)
parser.add_argument("--version", default=1,type=int)
parser.add_argument("--timesteps", default=50,type=int)


# Parse arguments
opt = parser.parse_args()

# Split the classes and combine the corresponding lists
class_names = opt.classes.split(',')
combined_list = []
for name in class_names:
    combined_list += class_lists.get(name, [])[:3]

num_data = int(opt.num)
if opt.version==1:
    feature_dim = 32071680
else:
    feature_dim = 32071680
diffusion_timesteps = opt.timesteps
CLIP_dim = 512


colors = ['red'] * 3*diffusion_timesteps + ['green'] * 3*diffusion_timesteps


pca_dir = f"PCA_{num_data}s_class{opt.classes}"
os.makedirs(pca_dir, exist_ok=True)

# List of PCA components
PCA_component_list = [25]
combined_feat = np.empty((num_data*diffusion_timesteps, feature_dim), dtype=np.float16)

print('Loading for PCA')

# Loop through numbers
for idx,num in enumerate(combined_list):
    flattened_values = np.load(f'/sketch/kwan/stablediffusion/Flatten_features/{int(num)}.npy')
    combined_feat[idx*diffusion_timesteps:idx*diffusion_timesteps+diffusion_timesteps] = flattened_values


    if idx % 10 == 0 and idx:
        print(idx)



assert combined_feat.shape == (diffusion_timesteps*num_data, feature_dim)

pca = joblib.load('PCA_t50_8s/25/pca_25.joblib')


print('PCA start')
for n_components in PCA_component_list:
    pca_dir_component = os.path.join(pca_dir, str(n_components))
    os.makedirs(pca_dir_component, exist_ok=True)

    reduced = pca.fit_transform(combined_feat)

    #color with timestep value. 
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=np.array(range(len(reduced))) % diffusion_timesteps, cmap='coolwarm', linewidth=0.5)
    ax.set_xlabel('First Component', fontsize=14, labelpad=13)
    ax.set_ylabel('Second Component', fontsize=14, labelpad=13)
    ax.set_zlabel('Third Component', fontsize=14, labelpad=13)
    cbar_ax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Position colorbar axes
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Timestep', size=15)  # Set font size for the label
    cbar.ax.tick_params(labelsize=12)    # Set font size for the tick labels

    plt.savefig(os.path.join(pca_dir_component, "3D_PCA_combined_result_timestep.png"), dpi=500)
    plt.close()

    #color with timestep value. 
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=colors, edgecolors='w', linewidth=0.5)
    ax.set_xlabel('First Component', fontsize=14, labelpad=13)
    ax.set_ylabel('Second Component', fontsize=14, labelpad=13)
    ax.set_zlabel('Third Component', fontsize=14, labelpad=13)
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label=class_names[0]),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label=class_names[1]),
                       ],
               loc='best', fontsize=14)
    plt.savefig(os.path.join(pca_dir_component, "3D_PCA_combined_result_semantic.png"), dpi=500)
    plt.close()
