import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules import Unet
from dataset import train_loader, val_loader
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

def dice_coeff(pred, target, num_classes=4):
    smooth = 1e-6
    dice_scores = []

    # Loop through each foreground class (1, 2, 3)
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


def train(model, train_loader, val_loader, num_epochs=100, lr=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    weights = torch.tensor([0.1, 1.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    best_dice = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_dice = [0.0] * 3
        train_mean_dice = 0.0
        for i, (img, mask) in enumerate(tqdm(train_loader)):
            img, mask = img.to(device), mask.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, mask.squeeze(1).long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pred = output.argmax(1)  # Shape [b, h, w]
            dice_scores, mean_dice = dice_coeff(pred, mask.squeeze(1), num_classes=4)
            train_dice = [train_dice[j] + dice_scores[j].item() for j in range(3)]  # Include only foreground
            train_mean_dice += mean_dice.item()

        model.eval()
        val_loss = 0
        val_dice = [0.0] * 3
        val_mean_dice = 0.0
        with torch.no_grad():
            for i, (img, mask) in enumerate(val_loader):
                img, mask = img.to(device), mask.to(device)
                output = model(img)
                loss = criterion(output, mask.squeeze(1).long())  # Convert mask to Long tensor
                val_loss += loss.item()

                pred = output.argmax(1)
                dice_scores, mean_dice = dice_coeff(pred, mask.squeeze(1), num_classes=4)
                val_dice = [val_dice[j] + dice_scores[j].item() for j in range(3)]  # Include only foreground
                val_mean_dice += mean_dice.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_dice = [d / len(train_loader) for d in train_dice]
        val_dice = [d / len(val_loader) for d in val_dice]
        train_mean_dice /= len(train_loader)
        val_mean_dice /= len(val_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Dice (Class 1, 2, 3): {train_dice}, Train Mean Dice: {train_mean_dice:.4f}")
        print(f"Val Dice (Class 1, 2, 3): {val_dice}, Val Mean Dice: {val_mean_dice:.4f}")

        scheduler.step(val_mean_dice)

        if val_mean_dice > best_dice:
            best_dice = val_mean_dice
            torch.save(model.state_dict(), "best_model.pth")


model = Unet()
train(model, train_loader, val_loader)