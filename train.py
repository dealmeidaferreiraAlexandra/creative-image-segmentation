import os
import torch
from torch.utils.data import DataLoader, random_split
from src.dataset import PetDataset
from src.unet import UNet
from src.baseline import BaselineCNN
from src.metrics import iou_score, dice_score
import matplotlib.pyplot as plt
from datetime import datetime

# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 5
LR = 1e-3

os.makedirs("models", exist_ok=True)
os.makedirs("results/comparisons", exist_ok=True)

dataset = PetDataset()

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

def train_model(model, name):
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.BCELoss()

    print(f"\n🚀 Training {name}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds = model(imgs)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"{name} Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    # guardar modelo
    torch.save(model.state_dict(), f"models/{name}.pth")

    return model


def evaluate(model, name):
    model.eval()

    iou_total = 0
    dice_total = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds = model(imgs)

            iou_total += iou_score(preds, masks).item()
            dice_total += dice_score(preds, masks).item()

    iou = iou_total / len(val_loader)
    dice = dice_total / len(val_loader)

    print(f"\n📊 {name} Results:")
    print(f"IoU: {iou:.4f}")
    print(f"Dice: {dice:.4f}")

    return iou, dice


def save_comparison(unet, baseline):
    imgs, masks = next(iter(val_loader))
    imgs = imgs.to(DEVICE)

    with torch.no_grad():
        pred_unet = unet(imgs)[0][0].cpu()
        pred_base = baseline(imgs)[0][0].cpu()

    img = imgs[0].cpu().permute(1,2,0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(10,4))

    plt.subplot(1,4,1)
    plt.imshow(img)
    plt.title("Image")

    plt.subplot(1,4,2)
    plt.imshow(masks[0][0], cmap='gray')
    plt.title("GT")

    plt.subplot(1,4,3)
    plt.imshow(pred_unet, cmap='gray')
    plt.title("U-Net")

    plt.subplot(1,4,4)
    plt.imshow(pred_base, cmap='gray')
    plt.title("Baseline")

    path = f"results/comparisons/compare_{timestamp}.png"
    plt.savefig(path)
    plt.close()

    print(f"📸 Saved comparison: {path}")


# ==============================
# RUN
# ==============================

unet = train_model(UNet(), "unet")
baseline = train_model(BaselineCNN(), "baseline")

evaluate(unet, "U-Net")
evaluate(baseline, "Baseline")

save_comparison(unet, baseline)

from src.save_results import save_visual_comparison

imgs, masks = next(iter(val_loader))
imgs = imgs.to(DEVICE)

with torch.no_grad():
    pred_unet = unet(imgs)[0][0].cpu().numpy()
    pred_base = baseline(imgs)[0][0].cpu().numpy()

img = imgs[0].cpu().permute(1,2,0).numpy()
mask = masks[0][0].numpy()

save_visual_comparison(img, mask, pred_unet, pred_base)

print("🏁 DONE")