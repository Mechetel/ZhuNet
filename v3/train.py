import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import os
from model import AttentionZhuNet
from utils import get_data_loaders, calculate_metrics, save_checkpoint, MetricsTracker


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, print_freq=50):
    """
    Train the model for one epoch.

    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        epoch: Current epoch number
        print_freq: Frequency of printing progress

    Returns:
        tuple: (epoch_loss, metrics_dict, all_labels, all_preds)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    print(f"\n{'='*70}")
    print(f"Epoch: {epoch}")
    print(f"{'='*70}")

    pbar = tqdm(train_loader, desc='Training')

    for batch_idx, (data, labels) in enumerate(pbar):
        data, labels = data.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        pred = outputs.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

        running_loss += loss.item()

        # Store predictions and labels for metrics
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())

        # Update progress bar
        batch_acc = 100.0 * pred.eq(labels).sum().item() / labels.size(0)
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{batch_acc:.2f}%'
        })

        # Print detailed progress
        if (batch_idx + 1) % print_freq == 0:
            print(f'  Batch [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.6f}, '
                  f'Accuracy: {batch_acc:.2f}%')

    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader)
    metrics = calculate_metrics(all_labels, all_preds)

    return epoch_loss, metrics, all_labels, all_preds


def validate(model, val_loader, criterion, device):
    """
    Validate the model.

    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run on

    Returns:
        tuple: (val_loss, metrics_dict, all_labels, all_preds, all_probs)
    """
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc='Validation'):
            data, labels = data.to(device), labels.to(device)

            output = model(data)
            val_loss += criterion(output, labels).item()

            # Get predictions and probabilities
            probs = torch.softmax(output, dim=1)[:, 1]  # Probability of stego class
            pred = output.argmax(dim=1)

            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    val_loss /= len(val_loader)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)

    return val_loss, metrics, all_labels, all_preds, all_probs


def train(args):
    """
    Main training function.

    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, valid_loader, _ = get_data_loaders(
        args.train_cover,
        args.train_stego,
        args.val_cover,
        args.val_stego,
        args.test_cover,
        args.test_stego,
        args.batch_size,
        args.num_workers
    )

    # Initialize model
    model = AttentionZhuNet().to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Loss function and optimizer - EXACT SAME AS ORIGINAL
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(),  # Only optimize layer2 parameters, same as original
        lr=args.lr,
        momentum=0.95,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler - same as original
    schedule_lr = args.schedule_lr

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(args.save_dir)

    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')

    print("\n" + "="*70)
    print("Starting training...")
    print(f"Total epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Learning rate schedule: {schedule_lr}")
    print("="*70 + "\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # Learning rate scheduling - EXACT SAME AS ORIGINAL
        # if epoch in schedule_lr:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] /= 5
        #     print(f"Learning rate adjusted to: {param_group['lr']}")

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Train
        train_loss, train_metrics, _, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.print_freq
        )

        # Update training metrics
        metrics_tracker.update_train_metrics(epoch, train_loss, train_metrics, current_lr)

        # Validate every validation_freq epochs
        if epoch % args.validation_freq == 0:
            val_loss, val_metrics, _, _, _ = validate(model, valid_loader, criterion, device)

            # Update validation metrics
            metrics_tracker.update_valid_metrics(epoch, val_loss, val_metrics)

            # Print epoch summary
            metrics_tracker.print_epoch_summary(epoch, train_loss, train_metrics, val_loss, val_metrics)

            # Save best model based on validation accuracy
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                filepath = save_checkpoint(
                    model, optimizer, epoch, val_loss, args.save_dir,
                    f"best_model_acc_{val_metrics['accuracy']:.4f}_epoch_{epoch}.pth"
                )
                print(f"🎉 New best accuracy! Model saved to {filepath}")

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                filepath = save_checkpoint(
                    model, optimizer, epoch, val_loss, args.save_dir,
                    f"best_model_loss_{val_loss:.4f}_epoch_{epoch}.pth"
                )
                print(f"✓ New best loss! Model saved to {filepath}")
        else:
            # Print training summary only
            metrics_tracker.print_epoch_summary(epoch, train_loss, train_metrics)

        # Save checkpoint at save frequency
        if epoch % args.save_freq == 0:
            filepath = save_checkpoint(
                model, optimizer, epoch, train_loss, args.save_dir,
                f"checkpoint_epoch_{epoch}.pth"
            )
            print(f"Checkpoint saved to {filepath}")

        # Calculate and print epoch time
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch time: {epoch_time:.2f} seconds\n")

        # Save metrics after each epoch
        metrics_tracker.save_metrics()

    # Save final model
    final_filepath = save_checkpoint(
        model, optimizer, args.epochs, train_loss, args.save_dir,
        "final_model.pth"
    )
    print(f"\nFinal model saved to {final_filepath}")

    # Save final metrics
    metrics_tracker.save_metrics()

    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total training time: {time.time() - metrics_tracker.start_time.timestamp():.2f} seconds")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Steganalysis Classifier (AttentionZhuNet)')

    # Machine type
    parser.add_argument(
        "--machine",
        type=str,
        default="server",
        choices=["local", "server"],
        help="Machine type: 'local' or 'server' (default: server)"
    )

    args, remaining = parser.parse_known_args()

    # Set paths based on machine type
    if args.machine == "local":
        default_train_cover = r'/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/cover/train'
        default_train_stego = r'/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/stego/S-UNIWARD/0.4bpp/train'
        default_val_cover = r'/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/cover/val'
        default_val_stego = r'/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/stego/S-UNIWARD/0.4bpp/val'
        default_test_cover = r'/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/cover/test'
        default_test_stego = r'/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/stego/S-UNIWARD/0.4bpp/test'
    else:  # server
        default_train_cover = r'/root/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/cover/train'
        default_train_stego = r'/root/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/stego/S-UNIWARD/0.4bpp/train'
        default_val_cover = r'/root/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/cover/val'
        default_val_stego = r'/root/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/stego/S-UNIWARD/0.4bpp/val'
        default_test_cover = r'/root/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/cover/test'
        default_test_stego = r'/root/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/stego/S-UNIWARD/0.4bpp/test'

    # Data paths
    parser.add_argument("--train_cover", default=default_train_cover, help="Path to training cover images")
    parser.add_argument("--train_stego", default=default_train_stego, help="Path to training stego images")
    parser.add_argument("--val_cover", default=default_val_cover, help="Path to validation cover images")
    parser.add_argument("--val_stego", default=default_val_stego, help="Path to validation stego images")
    parser.add_argument("--test_cover", default=default_test_cover, help="Path to test cover images")
    parser.add_argument("--test_stego", default=default_test_stego, help="Path to test stego images")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--schedule_lr", type=int, nargs='+', default=[20, 35, 50, 65],
                       help="Epochs at which to decay learning rate by factor of 5")

    # Saving and logging
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_freq", type=int, default=5, help="Frequency of saving checkpoints (epochs)")
    parser.add_argument("--validation_freq", type=int, default=1, help="Frequency of validation (epochs)")
    parser.add_argument("--print_freq", type=int, default=50, help="Frequency of printing progress (batches)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")

    args = parser.parse_args()

    train(args)
