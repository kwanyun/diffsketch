
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Settings
image_folder = "None/horse in mud_wo/predicted_samples"  # Path to your folder with 100 images
device = "cuda" if torch.cuda.is_available() else "cpu"
import clip

# Load CLIP
model, preprocess =  clip.load('ViT-B/32', device=device, jit=False,download_root='/data/kwan/.cache/clip')

# Read image paths and sort for consistency
img_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)
                    if f.lower().endswith(('png', 'jpg', 'jpeg'))])


# Encode all images
image_features = []
for path in img_paths:
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image).float().squeeze()
        image_features.append(features / features.norm())

image_features = torch.stack(image_features)  # Shape: [N, 512]

# Reference image: let's use the first image

ref_im = preprocess(Image.open('horse.png')).unsqueeze(0).to(device)
ref_feat =  model.encode_image(ref_im).float().squeeze()

image_features = image_features / image_features.norm(dim=1, keepdim=True)
ref_feat = ref_feat / ref_feat.norm()  # (already normed if above, but explicit is better)


sims = image_features @ ref_feat


# Plot
plt.figure(figsize=(10, 4))
plt.plot(sims.detach().cpu().numpy(), marker='o')
plt.title("CLIP Image Similarity Without CDST")
plt.xlabel("Image Index During Training")
plt.ylabel("Cosine Similarity")
plt.ylim(0.25, 0.95)
plt.tight_layout()
plt.savefig("clipsim_ours.png", dpi=300)   # ← saves the figure as a PNG
