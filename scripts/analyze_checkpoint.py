
import torch
import matplotlib.pyplot as plt

# Load checkpoint
checkpoint_path = "UNet_IoUBCELoss_augmented.pth"  # Ensure this path is correct
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# View contents of the checkpoint
print("Keys in checkpoint:", checkpoint.keys())

# View metadata
print(f"Best Validation IoU: {checkpoint['iou']}")
print(f"Best Epoch: {checkpoint['epoch']}")

# Plot training and validation loss curves
train_losses = checkpoint['train_losses']
val_losses = checkpoint['val_losses']

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
