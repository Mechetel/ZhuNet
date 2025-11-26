import time
import torch
from utils_for_dataset import getDataLoader
from utils_train_valid import Trainer
from ZhuNet import Zhu_Net


# Dataset paths
train_cover = r'/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/cover/train'
train_stego = r'/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/stego/S-UNIWARD/0.4bpp/train'

test_cover = r'/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/cover/test'
test_stego = r'/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/stego/S-UNIWARD/0.4bpp/test'

val_cover = r'/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/cover/val'
val_stego = r'/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/stego/S-UNIWARD/0.4bpp/val'

# Create data loaders
print("Loading datasets...")
train_loader, valid_loader, test_loader = getDataLoader(
    train_cover,
    train_stego,
    test_cover,
    test_stego,
    val_cover,
    val_stego,
    batch_size=50
)

print(f"Train batches: {len(train_loader)}")
print(f"Valid batches: {len(valid_loader)}")
print(f"Test batches: {len(test_loader)}")

# Initialize model
net = Zhu_Net()

# Initialize trainer
trainer = Trainer(
    model=net,
    lr=0.001,
    cur_epoch=0,
    lr_decay=0.95,
    weight_decay=0.0,
    shedule_lr=[20, 35, 50, 65],
    token='ZhuNet_S-UNIWARD_0.4bpp',
    token1='experiment_1',
    save_dir="saved_models",
    print_freq=50
)

# Training loop
num_epochs = 150
print(f"\nStarting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    start_time = time.time()

    # Train
    trainer.train(train_loader=train_loader)

    # Validate every 5 epochs
    if (epoch + 1) % 5 == 0:
        trainer.valid(valid_loader=valid_loader)

    epoch_time = time.time() - start_time
    print(f"Epoch time: {epoch_time:.2f} seconds\n")

# Save final model
final_model_path = 'final_model.pth'
torch.save(net.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

# Save training metrics
trainer.save_metrics()

# Test final model
print("\nTesting final model...")
trainer.test(test_loader=test_loader)
