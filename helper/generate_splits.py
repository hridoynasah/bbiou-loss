
import os
import random

# Set your image directory
image_dir = r"E:\BBIoULoss\BBIoULoss_Updated_V7_Liver\kvasir-seg-main\data\Updated_Liver\images"

# Get all image filenames
all_images = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
all_images = sorted(all_images)  # Optional: for consistent order

# Remove extensions (because your train.txt should only contain filenames without .jpg/.png)
image_ids = [os.path.splitext(f)[0] for f in all_images]

# Shuffle and split (80% train, 20% val)
random.shuffle(image_ids)
split_index = int(0.8 * len(image_ids))
train_ids = image_ids[:split_index]
val_ids = image_ids[split_index:]

# Create output folder
split_folder = r"E:\BBIoULoss\BBIoULoss_Updated_V7_Liver\kvasir-seg-main\train-val-split"
os.makedirs(split_folder, exist_ok=True)

# Write to files
with open(os.path.join(split_folder, "train.txt"), "w") as f:
    for id in train_ids:
        f.write(id + "\n")

with open(os.path.join(split_folder, "val.txt"), "w") as f:
    for id in val_ids:
        f.write(id + "\n")

print(f"Done! {len(train_ids)} for training and {len(val_ids)} for validation.")
