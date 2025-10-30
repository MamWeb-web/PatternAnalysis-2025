
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from torchvision import transforms

class MedicalImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.img_paths = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith('.png')])
        self.mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith('.png')])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # Mapping pixel values to class labels
        mask = np.array(mask)
        mask[mask == 0] = 0
        mask[mask == 85] = 1
        mask[mask == 170] = 2
        mask[mask == 255] = 3
        mask = Image.fromarray(mask)

        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return img, mask

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

mask_transform = transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long).unsqueeze(0))

# Load data
train_dataset = MedicalImageDataset(
    img_dir="OASIS_DATA/keras_png_slices_train",
    mask_dir="OASIS_DATA/keras_png_slices_seg_train",
    transform=transform,
    mask_transform=mask_transform
)

val_dataset = MedicalImageDataset(
    img_dir="OASIS_DATA/keras_png_slices_validate",
    mask_dir="OASIS_DATA/keras_png_slices_seg_validate",
    transform=transform,
    mask_transform=mask_transform
)

test_dataset = MedicalImageDataset(
    img_dir="OASIS_DATA/keras_png_slices_test",
    mask_dir="OASIS_DATA/keras_png_slices_seg_test",
    transform=transform,
    mask_transform=mask_transform
)

# DataLoader
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)