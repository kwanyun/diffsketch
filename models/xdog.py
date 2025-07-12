import cv2
import numpy as np
import os
from einops import rearrange
import torch


# Difference of Gaussians applied to img input
def dog(img,size=(0,0),k=1.6,sigma=0.5,gamma=1):
	img1 = cv2.GaussianBlur(img,size,sigma)
	img2 = cv2.GaussianBlur(img,size,sigma*k)
	return (img1-gamma*img2)


# garygrossi xdog version
def xdog_garygrossi(tenInput, sigma=0.5, k=200, gamma=0.98, epsilon=0.1, phi=10):
    intWidth = tenInput.shape[2]
    intHeight = tenInput.shape[1]

    # Convert PyTorch tensor to NumPy
    img = tenInput.cpu().numpy().transpose(1, 2, 0) * 255.0
    img = np.mean(img, axis=2).astype(np.uint8)

    aux = dog(img, sigma=sigma, k=k, gamma=gamma) / 255.0

    # Apply threshold and mapping
    aux = np.where(aux >= epsilon, 1.0, 1.0 + np.tanh(phi * (aux - epsilon)))

    # Convert back to PyTorch tensor
    aux = torch.tensor(aux * 255.0).to(tenInput.device).unsqueeze(0).unsqueeze(0)

    return aux / 255.0