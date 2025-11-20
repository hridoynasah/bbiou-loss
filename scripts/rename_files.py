import os
import shutil

# Paths
images_dir = r"E:\BBIoULoss\Datasets\Pancreas-CT\output\images"
masks_dir  = r"E:\BBIoULoss\Datasets\Pancreas-CT\output\nii_masks"

# Get sorted lists of files
image_files = sorted(os.listdir(images_dir))
mask_files  = sorted(os.listdir(masks_dir))

if len(image_files) != len(mask_files):
    print(f" Warning: {len(image_files)} images but {len(mask_files)} masks")
    
# Rename masks one by one
for img_file, mask_file in zip(image_files, mask_files):
    img_name, img_ext = os.path.splitext(img_file)
    mask_name, mask_ext = os.path.splitext(mask_file)

    new_mask_name = img_name + mask_ext  # mask keeps its extension but matches image name
    old_path = os.path.join(masks_dir, mask_file)
    new_path = os.path.join(masks_dir, new_mask_name)

    shutil.move(old_path, new_path)  # rename (safe way)
    print(f" {mask_file} → {new_mask_name}")

print(" Done! All masks renamed to match images.")
