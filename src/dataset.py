import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SRDataset(Dataset):
    def __init__(self, root_dir, hr_size=96, upscale_factor=4):
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
        
        # Apply HR transformations first
        hr_img = self.hr_transform(img)
        
        # To create the LR image from the HR crop, we convert back to PIL temporarily
        # This ensures the LR and HR images are perfectly aligned
        hr_pil = transforms.ToPILImage()(hr_img)
        lr_img = self.lr_transform(hr_pil)

        return lr_img, hr_img