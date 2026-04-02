import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class SRDataset(Dataset):
    def __init__(self, root_dir, hr_size=128, upscale_factor=2):
        self.root_dir = root_dir
        self.image_filenames = [os.path.join(root_dir, x) for x in os.listdir(root_dir) 
                                if x.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.hr_size = hr_size
        self.lr_size = hr_size // upscale_factor

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # OpenCV is faster than PIL for large images
        img = cv2.imread(self.image_filenames[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        
        # Fast numpy-based random crop
        top = np.random.randint(0, h - self.hr_size + 1)
        left = np.random.randint(0, w - self.hr_size + 1)
        hr_img = img[top:top+self.hr_size, left:left+self.hr_size]
        
        # Fast numpy-based horizontal flip
        if np.random.random() > 0.5:
            hr_img = np.ascontiguousarray(hr_img[:, ::-1, :])
            
        # Fast OpenCV resize
        lr_img = cv2.resize(hr_img, (self.lr_size, self.lr_size), interpolation=cv2.INTER_CUBIC)
        
        # Convert to tensor and normalize to [-1, 1]
        # (H, W, C) -> (C, H, W)
        hr_img = (torch.from_numpy(hr_img.transpose(2, 0, 1)).float() / 127.5) - 1.0
        lr_img = (torch.from_numpy(lr_img.transpose(2, 0, 1)).float() / 127.5) - 1.0

        return lr_img, hr_img
