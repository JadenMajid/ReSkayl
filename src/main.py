import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from gen_model import GeneratorModel
from discrim_model import DiscriminatorModel
from dataset import SRDataset
import os
from torchvision.models import vgg19, VGG19_Weights

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 100
SAVE_PATH = "srgan_checkpoint.pth"



class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:36].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(DEVICE)
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return self.mse(self.vgg(input), self.vgg(target))

def main():
    gen = GeneratorModel(num_blocks=16).to(DEVICE)
    disc = DiscriminatorModel().to(DEVICE)

    gen_opt = optim.Adam(gen.parameters(), lr=LR, betas=(0.9, 0.999))
    disc_opt = optim.Adam(disc.parameters(), lr=LR, betas=(0.9, 0.999))
    
    vgg_loss_fn = VGGLoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # hr_size=96 and upscale_factor=4 results in 24x24 LR inputs
    train_ds = SRDataset(
        root_dir="path/to/your/images", 
        hr_size=96, 
        upscale_factor=4
    )

    # Adjust num_workers based on your CPU cores (e.g., 8-12 for your 5700x3d)
    loader = DataLoader(
        dataset=train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True # Faster data transfer to GPU
    )
    
    # hr_transform should crop to e.g. 96x96, lr_transform should downsample to 24x24
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print(f"Training on {DEVICE}...")

    for epoch in range(EPOCHS):
        for i, (lr_img, hr_img) in enumerate(loader):
            lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)

            # --- Train Discriminator ---
            disc_opt.zero_grad()
            
            fake_img = gen(lr_img)
            
            real_res = disc(hr_img)
            fake_res = disc(fake_img.detach())
            
            loss_disc_real = bce_loss_fn(real_res, torch.ones_like(real_res))
            loss_disc_fake = bce_loss_fn(fake_res, torch.zeros_like(fake_res))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            
            loss_disc.backward()
            disc_opt.step()

            # --- Train Generator ---
            gen_opt.zero_grad()
            
            gen_fake_res = disc(fake_img)
            
            # Perceptual Loss = Content Loss (VGG) + 1e-3 * Adversarial Loss
            content_loss = vgg_loss_fn(fake_img, hr_img)
            adversarial_loss = bce_loss_fn(gen_fake_res, torch.ones_like(gen_fake_res))
            
            loss_gen = content_loss + 1e-3 * adversarial_loss
            
            loss_gen.backward()
            gen_opt.step()

        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'gen_state_dict': gen.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'gen_opt_state_dict': gen_opt.state_dict(),
            'disc_opt_state_dict': disc_opt.state_dict(),
        }, SAVE_PATH)
        
        print(f"Epoch [{epoch}/{EPOCHS}] Loss G: {loss_gen.item():.4f} Loss D: {loss_disc.item():.4f}")

if __name__ == "__main__":
    main()