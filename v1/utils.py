import os
import numpy as np
import torch
from glob import glob
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
from datetime import datetime


class DatasetPair(Dataset):
    """Dataset for loading cover-stego image pairs."""

    def __init__(self, cover_dir, stego_dir, transform=None):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.cover_list = [x.split(os.sep)[-1] for x in glob(os.path.join(cover_dir, '*'))]
        self.transform = transform

        assert len(self.cover_list) != 0, f"cover_dir is empty: {cover_dir}"
        print(f"Loaded {len(self.cover_list)} image pairs from {cover_dir}")

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


def get_data_loaders(train_cover_dir, train_stego_dir, valid_cover_dir,
                     valid_stego_dir, test_cover_dir, test_stego_dir,
                     batch_size, num_workers=4):
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
        num_workers (int): Number of workers for data loading

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create datasets
    print("\n" + "="*70)
    print("Loading datasets...")
    print("="*70)

    train_data = DatasetPair(train_cover_dir, train_stego_dir, transform=transform)
    valid_data = DatasetPair(valid_cover_dir, valid_stego_dir, transform=transform)
    test_data = DatasetPair(test_cover_dir, test_stego_dir, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_data,
        collate_fn=my_collate,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_data,
        collate_fn=my_collate,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_data,
        collate_fn=my_collate,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train batches: {len(train_loader)} (samples: {len(train_data) * 2})")
    print(f"Valid batches: {len(valid_loader)} (samples: {len(valid_data) * 2})")
    print(f"Test batches: {len(test_loader)} (samples: {len(test_data) * 2})")
    print("="*70 + "\n")

    return train_loader, valid_loader, test_loader


def calculate_metrics(y_true, y_pred, y_probs=None):
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities (optional, for AUC calculation)

    Returns:
        dict: Dictionary containing all metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
    }

    # Calculate per-class metrics
    for label in [0, 1]:
        label_name = 'cover' if label == 0 else 'stego'
        metrics[f'precision_{label_name}'] = precision_score(
            y_true, y_pred, labels=[label], average='binary', pos_label=label, zero_division=0
        )
        metrics[f'recall_{label_name}'] = recall_score(
            y_true, y_pred, labels=[label], average='binary', pos_label=label, zero_division=0
        )
        metrics[f'f1_{label_name}'] = f1_score(
            y_true, y_pred, labels=[label], average='binary', pos_label=label, zero_division=0
        )

    # Calculate AUC if probabilities are provided
    if y_probs is not None:
        y_probs = np.array(y_probs)
        try:
            metrics['auc'] = roc_auc_score(y_true, y_probs)
        except:
            metrics['auc'] = 0.0

    return metrics


def save_checkpoint(model, optimizer, epoch, loss, save_dir, filename):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        loss: Current loss value
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    return filepath


class MetricsTracker:
    """
    Track and save training metrics across epochs.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Initialize metric storage
        self.train_metrics = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'precision_cover': [],
            'recall_cover': [],
            'f1_cover': [],
            'precision_stego': [],
            'recall_stego': [],
            'f1_stego': [],
            'learning_rate': []
        }

        self.valid_metrics = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'auc': [],
            'precision_cover': [],
            'recall_cover': [],
            'f1_cover': [],
            'precision_stego': [],
            'recall_stego': [],
            'f1_stego': []
        }

        self.best_metrics = {
            'best_val_acc': 0.0,
            'best_val_acc_epoch': 0,
            'best_val_loss': float('inf'),
            'best_val_loss_epoch': 0,
            'best_val_f1': 0.0,
            'best_val_f1_epoch': 0,
        }

        self.start_time = datetime.now()

    def update_train_metrics(self, epoch, loss, metrics, lr):
        """Update training metrics for current epoch."""
        self.train_metrics['epoch'].append(epoch)
        self.train_metrics['loss'].append(loss)
        self.train_metrics['learning_rate'].append(lr)

        for key, value in metrics.items():
            if key in self.train_metrics:
                self.train_metrics[key].append(value)

    def update_valid_metrics(self, epoch, loss, metrics):
        """Update validation metrics for current epoch."""
        self.valid_metrics['epoch'].append(epoch)
        self.valid_metrics['loss'].append(loss)

        for key, value in metrics.items():
            if key in self.valid_metrics:
                self.valid_metrics[key].append(value)

        # Update best metrics
        if metrics['accuracy'] > self.best_metrics['best_val_acc']:
            self.best_metrics['best_val_acc'] = metrics['accuracy']
            self.best_metrics['best_val_acc_epoch'] = epoch

        if loss < self.best_metrics['best_val_loss']:
            self.best_metrics['best_val_loss'] = loss
            self.best_metrics['best_val_loss_epoch'] = epoch

        if metrics['f1_score'] > self.best_metrics['best_val_f1']:
            self.best_metrics['best_val_f1'] = metrics['f1_score']
            self.best_metrics['best_val_f1_epoch'] = epoch

    def save_metrics(self):
        """Save all metrics to files."""
        # Save as JSON
        metrics_dict = {
            'train': self.train_metrics,
            'valid': self.valid_metrics,
            'best': self.best_metrics,
            'training_duration': str(datetime.now() - self.start_time)
        }

        json_path = os.path.join(self.save_dir, 'metrics.json')
        with open(json_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)

        # Save as CSV for easy plotting
        self._save_csv('train_metrics.csv', self.train_metrics)
        self._save_csv('valid_metrics.csv', self.valid_metrics)

        # Save summary
        self._save_summary()

    def _save_csv(self, filename, metrics_dict):
        """Save metrics dictionary as CSV."""
        import csv

        filepath = os.path.join(self.save_dir, filename)

        if not metrics_dict['epoch']:
            return

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(metrics_dict.keys())

            # Write data
            num_rows = len(metrics_dict['epoch'])
            for i in range(num_rows):
                row = [metrics_dict[key][i] for key in metrics_dict.keys()]
                writer.writerow(row)

    def _save_summary(self):
        """Save training summary."""
        summary_path = os.path.join(self.save_dir, 'training_summary.txt')

        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("="*70 + "\n\n")

            f.write(f"Training Duration: {datetime.now() - self.start_time}\n\n")

            f.write("Best Validation Metrics:\n")
            f.write(f"  Best Accuracy:  {self.best_metrics['best_val_acc']:.4f} (Epoch {self.best_metrics['best_val_acc_epoch']})\n")
            f.write(f"  Best Loss:      {self.best_metrics['best_val_loss']:.4f} (Epoch {self.best_metrics['best_val_loss_epoch']})\n")
            f.write(f"  Best F1 Score:  {self.best_metrics['best_val_f1']:.4f} (Epoch {self.best_metrics['best_val_f1_epoch']})\n\n")

            if self.valid_metrics['epoch']:
                last_epoch = self.valid_metrics['epoch'][-1]
                idx = -1

                f.write(f"Final Validation Metrics (Epoch {last_epoch}):\n")
                f.write(f"  Accuracy:           {self.valid_metrics['accuracy'][idx]:.4f}\n")
                f.write(f"  Precision:          {self.valid_metrics['precision'][idx]:.4f}\n")
                f.write(f"  Recall:             {self.valid_metrics['recall'][idx]:.4f}\n")
                f.write(f"  F1 Score:           {self.valid_metrics['f1_score'][idx]:.4f}\n")
                f.write(f"  AUC:                {self.valid_metrics['auc'][idx]:.4f}\n\n")

                f.write(f"  Cover Metrics:\n")
                f.write(f"    Precision:        {self.valid_metrics['precision_cover'][idx]:.4f}\n")
                f.write(f"    Recall:           {self.valid_metrics['recall_cover'][idx]:.4f}\n")
                f.write(f"    F1 Score:         {self.valid_metrics['f1_cover'][idx]:.4f}\n\n")

                f.write(f"  Stego Metrics:\n")
                f.write(f"    Precision:        {self.valid_metrics['precision_stego'][idx]:.4f}\n")
                f.write(f"    Recall:           {self.valid_metrics['recall_stego'][idx]:.4f}\n")
                f.write(f"    F1 Score:         {self.valid_metrics['f1_stego'][idx]:.4f}\n")

            f.write("\n" + "="*70 + "\n")

    def print_epoch_summary(self, epoch, train_loss, train_metrics, val_loss=None, val_metrics=None):
        """Print formatted summary for current epoch."""
        print(f"\n{'*'*70}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"{'*'*70}")

        print(f"\nTrain Metrics:")
        print(f"  Loss:      {train_loss:.6f}")
        print(f"  Accuracy:  {train_metrics['accuracy']:.4f} ({train_metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {train_metrics['precision']:.4f}")
        print(f"  Recall:    {train_metrics['recall']:.4f}")
        print(f"  F1 Score:  {train_metrics['f1_score']:.4f}")

        if val_loss is not None and val_metrics is not None:
            print(f"\nValidation Metrics:")
            print(f"  Loss:      {val_loss:.6f}")
            print(f"  Accuracy:  {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
            print(f"  Precision: {val_metrics['precision']:.4f}")
            print(f"  Recall:    {val_metrics['recall']:.4f}")
            print(f"  F1 Score:  {val_metrics['f1_score']:.4f}")
            if 'auc' in val_metrics:
                print(f"  AUC:       {val_metrics['auc']:.4f}")

            print(f"\n  Per-Class Metrics:")
            print(f"    Cover  - Prec: {val_metrics['precision_cover']:.4f}, Rec: {val_metrics['recall_cover']:.4f}, F1: {val_metrics['f1_cover']:.4f}")
            print(f"    Stego  - Prec: {val_metrics['precision_stego']:.4f}, Rec: {val_metrics['recall_stego']:.4f}, F1: {val_metrics['f1_stego']:.4f}")

        print(f"{'*'*70}\n")
