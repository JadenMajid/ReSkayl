import os
import urllib.request
import cv2
import numpy as np
import torch
from src.cli import load_generator

os.makedirs('showcase', exist_ok=True)

# 4 High Quality animal local files from pjreddie/darknet
urls = [
    "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg",
    "https://raw.githubusercontent.com/pjreddie/darknet/master/data/eagle.jpg",
    "https://raw.githubusercontent.com/pjreddie/darknet/master/data/giraffe.jpg",
    "https://raw.githubusercontent.com/pjreddie/darknet/master/data/horses.jpg"
]

print("Downloading exemplar animal images...")
hr_patches = []
patch_size = 240 # Larger patch to zoom out by 0.5x

for i, url in enumerate(urls):
    tmp_path = f"showcase/tmp_{i}.jpg"
    # Use curl to bypass python urllib blocks
    cmd = f'curl -sSfL -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/110.0.0.0 Safari/537.36" -o {tmp_path} "{url}"'
    res = os.system(cmd)
    if res != 0:
        print(f"Failed to download {url}")
        continue
        
    img = cv2.imread(tmp_path)
    if img is None:
        print(f"Failed to read image from {tmp_path}")
        continue
        
    # Crop a central patch
    h, w = img.shape[:2]
    start_x = max(0, w // 2 - patch_size // 2)
    start_y = max(0, h // 2 - patch_size // 2)
    patch = img[start_y:start_y+patch_size, start_x:start_x+patch_size]
    
    # If the image was very small, pad it
    if patch.shape[:2] != (patch_size, patch_size):
        patch = cv2.resize(patch, (patch_size, patch_size))
        
    hr_patches.append(patch)

        
print("Images prepared. Loading SRGAN model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_generator("model/srgan.pth", device)

display_size = 400 # Scale up for display so pixels are huge

for i, hr_patch in enumerate(hr_patches):
    print(f"Processing animal example {i+1}...")
    
    # 1. Create heavily pixelated LR image (60x60)
    lr_img = cv2.resize(hr_patch, (patch_size // 2, patch_size // 2), interpolation=cv2.INTER_CUBIC)
    
    # 2. To compare properly, Bicubic also upscales the 60x60 LR back to 120x120
    bicubic_img = cv2.resize(lr_img, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
    
    # 3. Create SRGAN Upscaled image from the 60x60 LR image to 120x120
    lr_rgb = lr_img[:, :, ::-1]
    lr_norm = (lr_rgb / 127.5) - 1.0
    img_tensor = torch.from_numpy(lr_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        sr_tensor = model(img_tensor)
        
    sr_img = sr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    sr_img = ((sr_img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    sr_img = sr_img[:, :, ::-1]
    
    # 4. Enlarge using Nearest Neighbor to make differences ultra-obvious (display_size x display_size)
    hr_disp = cv2.resize(hr_patch, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
    bicubic_disp = cv2.resize(bicubic_img, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
    sr_disp = cv2.resize(sr_img, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
    
    # Concatenate side by side: Original HR, Bicubic, SRGAN
    combined = np.hstack((hr_disp, bicubic_disp, sr_disp))
    
    # Add Text Labels with background
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.rectangle(combined, (0, 0), (display_size*3, 40), (0,0,0), -1)
    
    cv2.putText(combined, "Original (Crop)", (10, 25), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined, "Bicubic x2", (display_size + 10, 25), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined, "SRGAN x2", (display_size*2 + 10, 25), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    save_path = f"showcase/example_{i+1}.jpg"
    cv2.imwrite(save_path, combined)
    print(f"Saved {save_path}")

print("Showcase generation complete.")
