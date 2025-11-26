import os
import numpy as np
import torch
from glob import glob
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from PIL import Image

class DatasetPair(Dataset):
    """Dataset for loading cover-stego image pairs."""

    def __init__(self, cover_dir, stego_dir, transform=None):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        # Fixed: Changed glob pattern to properly find files
        self.cover_list = [x.split(os.sep)[-1] for x in glob(os.path.join(cover_dir, '*'))]
        self.transform = transform

        assert len(self.cover_list) != 0, f"cover_dir is empty: {cover_dir}"

    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, index):
        # Labels: 0 for cover, 1 for stego
        labels = torch.tensor([0, 1], dtype=torch.long)

        cover_path = os.path.join(self.cover_dir, self.cover_list[index])
        stego_path = os.path.join(self.stego_dir, self.cover_list[index])

        cover = Image.open(cover_path)
        stego = Image.open(stego_path)

        # Convert to numpy arrays with channel dimension
        cover_array = np.array(cover)[:, :, np.newaxis]
        stego_array = np.array(stego)[:, :, np.newaxis]

        # Apply transforms
        cover_tensor = self.transform(cover_array)
        stego_tensor = self.transform(stego_array)

        # Stack images
        imgs = torch.stack([cover_tensor, stego_tensor])

        return imgs, labels


def my_collate(batch):
    """Custom collate function to flatten batch dimension."""
    imgs, targets = zip(*batch)
    return torch.cat(imgs), torch.cat(targets)


def getDataLoader(train_cover_dir, train_stego_dir, valid_cover_dir,
                  valid_stego_dir, test_cover_dir, test_stego_dir, batch_size):
    """
    Create DataLoaders for training, validation, and testing.

    Args:
        train_cover_dir (str): Path to training cover images
        train_stego_dir (str): Path to training stego images
        valid_cover_dir (str): Path to validation cover images
        valid_stego_dir (str): Path to validation stego images
        test_cover_dir (str): Path to test cover images
        test_stego_dir (str): Path to test stego images
        batch_size (int): Batch size for DataLoaders

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create datasets
    train_data = DatasetPair(train_cover_dir, train_stego_dir, transform=transform)
    valid_data = DatasetPair(valid_cover_dir, valid_stego_dir, transform=transform)
    test_data = DatasetPair(test_cover_dir, test_stego_dir, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_data,
        collate_fn=my_collate,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_data,
        collate_fn=my_collate,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )
    test_loader = DataLoader(
        test_data,
        collate_fn=my_collate,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    return train_loader, valid_loader, test_loader
