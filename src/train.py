import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from discrim_model import DiscriminatorModel
from gen_model import GeneratorModel
from torchvision.models import vgg19
"""
use Nesterov?
http://torch.ch/blog/2016/02/04/resnets.html
"""


# Define VGG loss according to paper (using VGG19 feature extractor)
class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights='VGG19_Weights.DEFAULT').features[:36].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return self.mse(self.vgg(input), self.vgg(target))
    

def train(gen: GeneratorModel, disc: DiscriminatorModel, loader: DataLoader, epochs=10, device=torch.device("cuda")):
    gen.to(device)
    disc.to(device)
    
    vgg_loss_fn = VGGLoss().to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()

    # Paper recommends Adam with lr=1e-4
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-4)
    disc_optimizer = optim.Adam(disc.parameters(), lr=1e-4)

    gen_adversarial_alpha = 1e-3

    for epoch in range(epochs):
        for batch_idx, (lr_images, hr_images) in enumerate(loader):
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            batch_size = lr_images.size(0)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            disc_optimizer.zero_grad()

            # Real images
            real_preds = disc(hr_images)
            real_labels = torch.ones_like(real_preds)
            loss_real = bce_loss_fn(real_preds, real_labels)

            # Fake images
            fake_images = gen(lr_images)
            fake_preds = disc(fake_images.detach())
            fake_labels = torch.zeros_like(fake_preds)
            loss_fake = bce_loss_fn(fake_preds, fake_labels)

            loss_disc = loss_real + loss_fake
            loss_disc.backward()
            disc_optimizer.step()

            # -----------------
            #  Train Generator
            # -----------------
            gen_optimizer.zero_grad()

            # Adversarial Loss (maximize log D(G(LR)))
            gen_fake_preds = disc(fake_images)
            gen_adv_loss = bce_loss_fn(gen_fake_preds, torch.ones_like(gen_fake_preds))

            # Content Loss (VGG)
            content_loss = vgg_loss_fn(fake_images, hr_images)

            # Total Perceptual Loss
            total_gen_loss = content_loss + (gen_adversarial_alpha * gen_adv_loss)

            total_gen_loss.backward()
            gen_optimizer.step()