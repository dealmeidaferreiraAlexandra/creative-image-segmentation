import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime

def save_visual_comparison(img, mask, pred_unet, pred_base):
    os.makedirs("results/final", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(12,4))

    plt.subplot(1,4,1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(pred_unet, cmap="gray")
    plt.title("U-Net")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.imshow(pred_base, cmap="gray")
    plt.title("Baseline")
    plt.axis("off")

    path = f"results/final/final_{timestamp}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    print(f"🔥 Saved FINAL: {path}")