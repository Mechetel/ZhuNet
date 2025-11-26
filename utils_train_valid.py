import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class Trainer:
    """
    Trainer class for model training and validation.

    Args:
        model: Neural network model
        lr (float): Learning rate
        lr_decay (float): Learning rate decay factor
        token (str): Experiment identifier
        token1 (str): Window name identifier
        weight_decay (float): Weight decay for optimizer
        save_dir (str): Directory to save model checkpoints
        optimizer: Custom optimizer (optional)
        save_freq (int): Frequency of saving checkpoints (epochs)
        cur_epoch (int): Starting epoch number
        print_freq (int): Frequency of printing training progress
        shedule_lr (list): Epochs at which to decay learning rate
    """

    def __init__(self, model, lr, lr_decay, token, token1, weight_decay, save_dir,
                 optimizer=None, save_freq=1, cur_epoch=0, print_freq=150,
                 shedule_lr=None):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_f = nn.CrossEntropyLoss().to(self.device)
        self.shedule_lr = shedule_lr if shedule_lr else []

        self.lr_decay = lr_decay

        if optimizer is None:
            self.optimizer = optim.SGD(
                self.model.layer2.parameters(),
                lr=lr,
                momentum=0.95,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer

        # Create save directory
        self.save_dir = os.path.join("PATH", save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        self.save_freq = save_freq
        self.print_freq = print_freq
        self.cur_epoch = cur_epoch

        # Tracking metrics
        self.train_loss = []
        self.train_acc = []
        self.val_acc = []
        self.val_loss = []
        self.best_acc = 0

    def train(self, train_loader):
        """
        Train the model for one epoch.

        Args:
            train_loader: DataLoader for training data
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        self.cur_epoch += 1

        print(f"\n{'='*60}")
        print(f"Epoch: {self.cur_epoch}")
        print(f"{'='*60}")

        # Learning rate scheduling
        if self.cur_epoch in self.shedule_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] /= 5
            print(f"Learning rate adjusted to: {param_group['lr']}")

        # Check if train_loader is empty
        if len(train_loader) == 0:
            print("Warning: train_loader is empty!")
            return

        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)

            # Calculate loss
            loss = self.loss_f(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Calculate accuracy
            pred = outputs.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

            running_loss += loss.item()

            # Print progress
            if (batch_idx + 1) % self.print_freq == 0:
                batch_acc = 100.0 * pred.eq(labels).sum().item() / labels.size(0)
                print(f'  Batch [{batch_idx + 1}/{len(train_loader)}] '
                      f'Loss: {loss.item():.6f}, '
                      f'Accuracy: {batch_acc:.2f}%')

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        self.train_loss.append(epoch_loss)
        self.train_acc.append(epoch_acc)

        print(f"\n{'*'*60}")
        print(f"TRAIN - Epoch {self.cur_epoch} Summary:")
        print(f"  Average Loss: {epoch_loss:.6f}")
        print(f"  Accuracy: {correct}/{total} ({epoch_acc:.2f}%)")
        print(f"{'*'*60}")

        # Save checkpoint
        if self.cur_epoch % self.save_freq == 0:
            checkpoint_path = os.path.join(
                self.save_dir,
                f"epoch_{self.cur_epoch}.pth"
            )
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")

    def valid(self, valid_loader):
        """
        Validate the model.

        Args:
            valid_loader: DataLoader for validation data
        """
        self.model.eval()

        valid_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in valid_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                output = self.model(data)
                valid_loss += self.loss_f(output, labels).item()

                pred = output.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

        # Calculate metrics
        valid_loss /= len(valid_loader)
        cur_acc = 100.0 * correct / total

        print(f"\n{'*'*60}")
        print(f"VALIDATION - Epoch {self.cur_epoch} Summary:")
        print(f"  Average Loss: {valid_loss:.6f}")
        print(f"  Accuracy: {correct}/{total} ({cur_acc:.2f}%)")
        print(f"{'*'*60}")

        self.val_acc.append(cur_acc)
        self.val_loss.append(valid_loss)

        # Save best model
        if cur_acc > self.best_acc:
            self.best_acc = cur_acc
            best_model_path = os.path.join(
                self.save_dir,
                f"best_model_epoch_{self.cur_epoch}_acc_{cur_acc:.2f}.pth"
            )
            torch.save(self.model.state_dict(), best_model_path)
            print(f"🎉 New best accuracy! Model saved to {best_model_path}")

    def test(self, test_loader):
        """
        Test the model.

        Args:
            test_loader: DataLoader for test data
        """
        self.model.eval()

        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                output = self.model(data)
                test_loss += self.loss_f(output, labels).item()

                pred = output.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

        # Calculate metrics
        test_loss /= len(test_loader)
        accuracy = 100.0 * correct / total

        print(f"\n{'='*60}")
        print(f"TEST RESULTS:")
        print(f"  Average Loss: {test_loss:.4f}")
        print(f"  Accuracy: {correct}/{total} ({accuracy:.2f}%)")
        print(f"{'='*60}\n")

    def save_metrics(self):
        """Save training and validation metrics to text files."""
        # Save training loss
        with open(os.path.join(self.save_dir, "train_loss.txt"), 'w') as f:
            f.write(','.join(map(str, self.train_loss)))

        # Save training accuracy
        with open(os.path.join(self.save_dir, "train_acc.txt"), 'w') as f:
            f.write(','.join(map(str, self.train_acc)))

        # Save validation accuracy
        with open(os.path.join(self.save_dir, "val_acc.txt"), 'w') as f:
            f.write(','.join(map(str, self.val_acc)))

        # Save validation loss
        with open(os.path.join(self.save_dir, "val_loss.txt"), 'w') as f:
            f.write(','.join(map(str, self.val_loss)))

        print("Metrics saved successfully")

    def print_training_summary(self):
        """Print a summary of all training epochs."""
        print(f"\n{'='*80}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"{'Epoch':<8} {'Train Loss':<15} {'Train Acc':<15} {'Val Loss':<15} {'Val Acc':<15}")
        print(f"{'-'*80}")

        for i in range(len(self.train_loss)):
            train_loss_str = f"{self.train_loss[i]:.6f}"
            train_acc_str = f"{self.train_acc[i]:.2f}%" if i < len(self.train_acc) else "N/A"
            val_loss_str = f"{self.val_loss[i]:.6f}" if i < len(self.val_loss) else "N/A"
            val_acc_str = f"{self.val_acc[i]:.2f}%" if i < len(self.val_acc) else "N/A"

            print(f"{i+1:<8} {train_loss_str:<15} {train_acc_str:<15} {val_loss_str:<15} {val_acc_str:<15}")

        print(f"{'='*80}")
        print(f"Best Validation Accuracy: {self.best_acc:.2f}%")
        print(f"{'='*80}\n")
