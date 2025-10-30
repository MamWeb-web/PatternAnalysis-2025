import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules import Unet
from dataset import test_loader, val_dataset
from torch import nn
import numpy as np
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import random

def dice_coeff(pred, target, num_classes=4):
    smooth = 1e-6
    dice_scores = []

    for i in range(1, num_classes):  # Start from 1 to exclude background (class 0)
        # Create binary masks for the current class (i)
        pred_class = (pred == i).float()  # Only keep predictions for the current class
        target_class = (target == i).float()  # Only keep targets for the current class

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()

        if union == 0:
            dice_scores.append(torch.tensor(0.0))  # Set to 0 if class absent (avoids misleading 1.0)
            continue

        dice_score = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice_score)

    if len(dice_scores) == 0:
        mean_dice = torch.tensor(0.0)
    else:
        mean_dice = torch.mean(torch.tensor(dice_scores))
    return dice_scores, mean_dice

def test(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    val_dice = [0.0] * 3
    val_mean_dice = 0.0
    with torch.no_grad():
        for i, (img, mask) in enumerate(tqdm(val_loader)):
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            pred = output.argmax(1)
            dice_scores, mean_dice = dice_coeff(pred, mask.squeeze(1), num_classes=4)
            val_dice = [val_dice[j] + dice_scores[j].item() for j in range(3)]  # Include only foreground
            val_mean_dice += mean_dice.item()

    val_dice = [d / len(val_loader) for d in val_dice]
    val_mean_dice /= len(val_loader)

    print(f"Test Dice (Class 1, 2, 3): {val_dice}, Test Mean Dice: {val_mean_dice:.4f}")

model = Unet()
model.load_state_dict(torch.load("best_model.pth"))

test(model, test_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

indices = random.sample(range(len(val_dataset)), 3)
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

for idx in indices:
    img, mask = val_dataset[idx]
    img_unsq = img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_unsq)
        pred = output.argmax(1).squeeze(0).cpu()

    mask = mask.squeeze(0).cpu()  # [H, W]

    img = img.cpu()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.permute(1, 2, 0).numpy()

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img)
    axs[0].set_title('Input Image')
    axs[0].axis('off')

    axs[1].imshow(mask, cmap='tab10', vmin=0, vmax=3)
    axs[1].set_title('Ground Truth Mask')
    axs[1].axis('off')

    axs[2].imshow(pred, cmap='tab10', vmin=0, vmax=3)
    axs[2].set_title('Predicted Mask')
    axs[2].axis('off')

    plt.show()