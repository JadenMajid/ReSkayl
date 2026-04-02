from datasets import load_dataset
from PIL import Image
import os
import traceback

os.makedirs('showcase_imgs', exist_ok=True)
try:
    ds = load_dataset("yangtao9009/Flickr2K", split="train", streaming=True)
    it = iter(ds)
    for i in range(4):
        item = next(it)
        print("Keys:", item.keys())
        # Let's see what keys are there, typically 'image' or 'hr' or 'lr'
        for key in item.keys():
            if isinstance(item[key], Image.Image):
                item[key].save(f"showcase_imgs/Flickr2K_{i:04d}_{key}.png")
                print(f"Saved image from key {key}")
                break
        else:
            print("No PIL image found in item.")
except Exception as e:
    traceback.print_exc()
