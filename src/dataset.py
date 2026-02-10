import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SRDataset(Dataset):
    def __init__(self, root_dir, hr_size=128, upscale_factor=2):
        """
        Args:
            root_dir (string): Directory with all the images.
            hr_size (int): The size of the high-res crop.
            upscale_factor (int): How much smaller the LR image should be.
        """
        self.root_dir = root_dir
        self.image_filenames = [os.path.join(root_dir, x) for x in os.listdir(root_dir) 
                                if x.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.hr_size = hr_size
        self.lr_size = hr_size // upscale_factor

        # HR images should be normalized to [-1, 1] for the Tanh output of the generator
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(self.hr_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # LR images are downsampled from the HR crop
        self.lr_transform = transforms.Compose([
            transforms.Resize((self.lr_size, self.lr_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img = Image.open(self.image_filenames[idx]).convert('RGB')
        
        # Apply crop and flip to create HR crop (WITHOUT normalization yet)
        hr_crop_transform = transforms.Compose([
            transforms.RandomCrop(self.hr_size),
            transforms.RandomHorizontalFlip()
        ])
        hr_crop = hr_crop_transform(img)
        
        # Create LR from HR crop, then normalize both
        lr_crop = hr_crop.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        hr_img = normalize(hr_crop)
        lr_img = normalize(lr_crop)

        return lr_img, hr_img