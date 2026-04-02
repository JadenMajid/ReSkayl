import argparse
import sys
import os
from pathlib import Path
import torch
import cv2
import numpy as np
from src.gen_model import GeneratorModel

def load_generator(model_path, device):
    model = GeneratorModel(num_blocks=16)
    # Load state dict strictly first, mapping config to CPU
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['gen_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def process_img(model, input_path, output_path, device):
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"Skipping {input_path.name}: not a valid image.")
        return
    
    # Pre-processing: BGR to RGB, Normalize to [-1, 1], HWC to CHW
    img = img[:, :, ::-1] 
    img = (img / 127.5) - 1.0
    
    upscaled = None
    
    # Try running on specific device
    try:
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            upscaled = model(img_tensor)
        upscaled = upscaled.squeeze(0).cpu().numpy()
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device.type == "cuda":
            print(f"WARNING: CUDA Out of Memory on {input_path.name}. Falling back to CPU for this image.")
            torch.cuda.empty_cache()
            
            # Switch device
            fallback_device = torch.device("cpu")
            model = model.to(fallback_device)
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(fallback_device)
            
            with torch.no_grad():
                upscaled = model(img_tensor)
            upscaled = upscaled.squeeze(0).cpu().numpy()
            
            # Put model back to original device for next images
            model = model.to(device)
        else:
            raise e
            
    # Post-processing: CHW to HWC, Denormalize, BGR for CV2
    upscaled = upscaled.transpose(1, 2, 0)
    upscaled = ((upscaled + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    upscaled = upscaled[:, :, ::-1]
    
    cv2.imwrite(str(output_path), upscaled)

def main():
    parser = argparse.ArgumentParser(description="ReSkayl - SRGAN Image Upscaler")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input image file or directory")
    parser.add_argument("-o", "--output", type=str, default="./output", help="Output directory")
    parser.add_argument("-m", "--model", type=str, default="model/srgan.pth", help="Path to SRGAN model checkpoint (.pth)")
    
    args, unknown = parser.parse_known_args()
    
    in_path = Path(args.input)
    out_path = Path(args.output)
    
    if not in_path.exists():
        print(f"Error: Input path {in_path} does not exist.")
        sys.exit(1)
        
    out_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading model...")
    model = load_generator(args.model, device)
    
    # Logic for directory vs file
    files = [in_path] if in_path.is_file() else list(in_path.glob('*'))
    valid_files = 0
    
    for f in files:
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            valid_files += 1
            save_name = out_path / f.name
            
            print(f"Upscaling {f.name}...")
            process_img(model, f, save_name, device)
            
    if valid_files == 0:
        print("No valid image files (jpg, jpeg, png) found.")
    else:
        print("Upscaling complete.")

if __name__ == "__main__":
    main()
