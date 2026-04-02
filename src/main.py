import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from gen_model import GeneratorModel
from discrim_model import DiscriminatorModel
from dataset import SRDataset
from tqdm import tqdm
import os

# --- Performance Optimization ---
torch.backends.cudnn.benchmark = True

# --- Configuration ---
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# Full VRAM Utilization: Using batch size 8 with 2 accumulation steps
# effectively uses 16GB VRAM for 816x816 images
BATCH_SIZE = 8  
GRAD_ACCUM_STEPS = 2  
LR_G = 1e-4
LR_D = 1e-4
EPOCHS = 1000
ADV_WEIGHT = 1e-3
SAVE_PATH = "model/srgan_checkpoint.pth"

# Memory format for TensorCore optimization
MEM_FORMAT = torch.channels_last if "cuda" in str(DEVICE) else torch.contiguous_format

def main():
    # Initialize and convert models to channels_last for speed
    gen = GeneratorModel(num_blocks=16).to(DEVICE, memory_format=MEM_FORMAT)
    gen.use_checkpoint = True
    disc = DiscriminatorModel().to(DEVICE, memory_format=MEM_FORMAT)

    # Foreach Adam for reduced kernel launch overhead and GradScaler compatibility
    gen_opt = optim.Adam(gen.parameters(), lr=LR_G, betas=(0.9, 0.999), foreach=True)
    disc_opt = optim.Adam(disc.parameters(), lr=LR_D, betas=(0.9, 0.999), foreach=True)

    mse_loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda" if "cuda" in str(DEVICE) else "cpu")

    train_ds = SRDataset(root_dir="./data/flickr2k", hr_size=816, upscale_factor=2)
    
    # Increase num_workers for 5700x3d
    loader = DataLoader(
        dataset=train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
    )

    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
        gen.load_state_dict(checkpoint["gen_state_dict"])
        disc.load_state_dict(checkpoint["disc_state_dict"])
        gen_opt.load_state_dict(checkpoint["gen_opt_state_dict"])
        disc_opt.load_state_dict(checkpoint["disc_opt_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0

    print(f"Training on {DEVICE}...")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Total batches per epoch: {len(loader)}\n")

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        gen_loss_total = 0
        disc_loss_total = 0
        real_acc_total = 0
        fake_acc_total = 0

        pbar = tqdm(loader, desc=f"E:{epoch + 1}/{EPOCHS}", unit="b")
        
        gen_opt.zero_grad()
        disc_opt.zero_grad()

        for i, (lr_img, hr_img) in enumerate(pbar):
            # Move tensors and convert to NHWC format for TensorCores
            lr_img = lr_img.to(DEVICE, memory_format=MEM_FORMAT, non_blocking=True)
            hr_img = hr_img.to(DEVICE, memory_format=MEM_FORMAT, non_blocking=True)

            # --- Train Discriminator ---
            with torch.amp.autocast("cuda" if "cuda" in str(DEVICE) else "cpu"):
                fake_img = gen(lr_img)
                real_res = disc(hr_img)
                fake_res = disc(fake_img.detach())

                loss_disc_real = F.softplus(-real_res).mean()
                loss_disc_fake = F.softplus(fake_res).mean()
                loss_disc = (loss_disc_real + loss_disc_fake) / (2 * GRAD_ACCUM_STEPS)

            scaler.scale(loss_disc).backward()

            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(disc_opt)
                torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
                scaler.step(disc_opt)
                disc_opt.zero_grad()

            # --- Train Generator ---
            with torch.amp.autocast("cuda" if "cuda" in str(DEVICE) else "cpu"):
                gen_fake_res = disc(fake_img)
                content_loss = mse_loss_fn(fake_img, hr_img)
                adversarial_loss = F.softplus(-gen_fake_res).mean()
                loss_gen = (content_loss + ADV_WEIGHT * adversarial_loss) / GRAD_ACCUM_STEPS

            scaler.scale(loss_gen).backward()

            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(gen_opt)
                torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
                scaler.step(gen_opt)
                gen_opt.zero_grad()
                scaler.update()

            gen_loss_total += loss_gen.item() * GRAD_ACCUM_STEPS
            disc_loss_total += loss_disc.item() * GRAD_ACCUM_STEPS
            
            # Accuracy metrics
            real_acc = (real_res > 0).float().mean().item()
            fake_acc = (fake_res < 0).float().mean().item()
            real_acc_total += real_acc
            fake_acc_total += fake_acc

            pbar.set_postfix(
                {
                    "μG": f"{gen_loss_total / (i + 1):.4f}",
                    "μD": f"{disc_loss_total / (i + 1):.4f}",
                    "AccR": f"{real_acc_total / (i + 1):.2f}",
                    "AccF": f"{fake_acc_total / (i + 1):.2f}",
                }
            )

        torch.save(
            {
                "epoch": epoch,
                "gen_state_dict": gen.state_dict(),
                "disc_state_dict": disc.state_dict(),
                "gen_opt_state_dict": gen_opt.state_dict(),
                "disc_opt_state_dict": disc_opt.state_dict(),
            },
            SAVE_PATH,
        )

        print(f"{'=' * 60}")
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Complete - μG: {gen_loss_total/len(loader):.4f} | μD: {disc_loss_total/len(loader):.4f}")
        print(f"Checkpoint saved to {SAVE_PATH}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
