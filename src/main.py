import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from gen_model import GeneratorModel
from discrim_model import DiscriminatorModel
from dataset import SRDataset
from tqdm import tqdm
import os

# --- Configuration ---
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
BATCH_SIZE = 16
LR_G = 1e-5  # Generator learning rate
LR_D = 1e-5  # Discriminator learning rate
EPOCHS = 1000
ADV_WEIGHT = 1e-3  # Adversarial loss weight (can increase to 1e-2 if D still dominates)
SAVE_PATH = "model/srgan_checkpoint.pth"


def main():
    gen = GeneratorModel(num_blocks=16).to(DEVICE)
    disc = DiscriminatorModel().to(DEVICE)

    gen_opt = optim.Adam(gen.parameters(), lr=LR_G, betas=(0.9, 0.999))
    disc_opt = optim.Adam(disc.parameters(), lr=LR_D, betas=(0.9, 0.999))

    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    train_ds = SRDataset(root_dir="./data/flickr2k", hr_size=816, upscale_factor=2)

    # Adjust num_workers based on CPU cores (e.g., 8-16 for 5700x3d)
    loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=16,
        prefetch_factor=1,
        pin_memory=True,  # Faster data transfer to GPU
    )

    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)

        # Restore model states
        gen.load_state_dict(checkpoint["gen_state_dict"])
        disc.load_state_dict(checkpoint["disc_state_dict"])

        # Restore optimizer states
        gen_opt.load_state_dict(checkpoint["gen_opt_state_dict"])
        disc_opt.load_state_dict(checkpoint["disc_opt_state_dict"])

        # Resume from next epoch
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0

    print(f"Training on {DEVICE}...")
    print(f"Total batches per epoch: {len(loader)}\n")

    for epoch in range(start_epoch, start_epoch+EPOCHS):
        gen_loss_total = 0
        disc_loss_total = 0

        # Create progress bar for batches
        pbar = tqdm(loader, desc=f"E:{epoch + 1}/{EPOCHS}", unit="b")

        for i, (lr_img, hr_img) in enumerate(pbar):
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
            grad_norm_disc = torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=float('inf'))
            disc_opt.step()

            # --- Train Generator ---
            gen_opt.zero_grad()

            gen_fake_res = disc(fake_img)

            # Content Loss (MSE) + Adversarial Loss
            content_loss = mse_loss_fn(fake_img, hr_img)
            adversarial_loss = bce_loss_fn(gen_fake_res, torch.ones_like(gen_fake_res))

            loss_gen = content_loss + ADV_WEIGHT * adversarial_loss

            loss_gen.backward()
            grad_norm_gen = torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=float('inf'))
            gen_opt.step()

            gen_loss_total += loss_gen.item()
            disc_loss_total += loss_disc.item()

            # Update progress bar with current losses
            pbar.set_postfix(
                {
                    "∇G": f"{grad_norm_gen:.5f}",
                    "∇D": f"{grad_norm_disc:.5f}",
                    "μG": f"{gen_loss_total / (i + 1):.5f}",
                    "μD": f"{disc_loss_total / (i + 1):.5f}",
                }
            )

        # Calculate average losses for the epoch
        avg_gen_loss = gen_loss_total / len(loader)
        avg_disc_loss = disc_loss_total / len(loader)

        # Save Checkpoint
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
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] Complete - Avg G: {avg_gen_loss:.4f} | Avg D: {avg_disc_loss:.4f}"
        )
        print(f"Checkpoint saved to {SAVE_PATH}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

