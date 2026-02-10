import torch
import os
import cv2
import numpy as np
from sys import argv
from pathlib import Path
from gen_model import GeneratorModel

# Hardware-specific mapping
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT = "model/srgan_checkpoint.pth"

def load_generator(path):
    model = GeneratorModel(num_blocks=16).to(DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['gen_state_dict'])
    model.eval()
    return model

def process_img(model, input_path, output_path):
    img = cv2.imread(str(input_path))
    if img is None: return
    
    # Pre-processing: BGR to RGB, Normalize to [-1, 1], HWC to CHW
    img = img[:, :, ::-1] 
    img = (img / 127.5) - 1.0
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        upscaled = model(img_tensor)
    
    # Post-processing: CHW to HWC, Denormalize, BGR for CV2
    upscaled = upscaled.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    upscaled = ((upscaled + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    upscaled = upscaled[:, :, ::-1]
    
    cv2.imwrite(str(output_path), upscaled)

def main():
    if len(argv) < 2:
        print("Usage: python upscale.py <input_path> [output_path]")
        return

    in_path = Path(argv[1])
    out_path = Path(argv[2]) if len(argv) > 2 else Path("./output/")
    out_path.mkdir(parents=True, exist_ok=True)

    model = load_generator(CHECKPOINT)

    # Logic for directory vs file
    files = [in_path] if in_path.is_file() else list(in_path.glob('*'))
    
    for f in files:
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            save_name = out_path / f.name if in_path.is_dir() else out_path / f.name
            if out_path.is_dir() and in_path.is_file():
                save_name = out_path / f.name
                
            print(f"Upscaling {f.name}...")
            process_img(model, f, save_name)

if __name__ == "__main__":
    main()