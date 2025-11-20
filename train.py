#%%writefile /kaggle/working/BBIoULoss_Updated_V7_Liver/kvasir-seg-main/train_models.py
# --------------------------------------------------------------------------------
# built-in imports
# --------------------------------------------------------------------------------
import os
import sys
import copy
import time
import random
import argparse


# --------------------------------------------------------------------------------
# standard imports
# --------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
# working with images
#import cv2
import imageio as iio
# torch
#from sympy import python
import torch
import torch.nn as nn
import torch.nn.functional as F
# torchvision
import torchvision
import torchvision.transforms as transforms
# torchmetrics
import torchmetrics # for metrics like IoU, accuracy, etc.
# torchsummary
import torchsummary #shows model summary like Keras.
# interactive progress bar
from tqdm import notebook
# debugging
import ipdb


# --------------------------------------------------------------------------------
# custom imports
# --------------------------------------------------------------------------------


# losses
from utils.metrics import (
    iou_pytorch_eval, IoULoss, BiouLoss, BCEWithLogitsLoss, IoUBCELoss, ConservativeBBIoULoss
)


# dataset
from utils.dataset import myDataSet
# transforms
from utils.transforms import SIZE, resize_transform, train_transforms, test_transforms
# models
from models.unet import UNet


# --------------------------------------------------------------------------------
# train and validation loop functions
# --------------------------------------------------------------------------------


def train_eval_one_epoch(model, optimizer, criterion, dataloader, epoch, device, settings, train_mode, clip_grad_norm=None):
    if train_mode == True:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_iou = 0
    nan_count = 0
    
    for i, (imgs, masks) in enumerate(dataloader):
        batch_size = imgs.shape[0]
       
        imgs, masks = imgs.to(device), masks.to(device)
        
        # Forward Pass
        if train_mode:
            prediction = model(imgs)
        else:
            with torch.no_grad():
                prediction = model(imgs)

        if train_mode:
            loss = criterion(prediction, masks)
            
            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nNaN/Inf loss detected at epoch {epoch}, batch {i}")
                print(f"Loss value: {loss.item() if not torch.isnan(loss) else 'NaN'}")
                nan_count += 1
                if nan_count > 3:
                    print("Too many NaN losses, stopping epoch")
                    break
                continue
            
            optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping if specified
            if clip_grad_norm is not None and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            optimizer.step()
        else:
            loss = criterion(prediction, masks)
            if torch.isnan(loss) or torch.isinf(loss):
                continue

        batch_loss = loss.item()
        total_loss += batch_loss

        # FIXED: Use reduction="sum" for proper accumulation
        batch_iou = iou_pytorch_eval(prediction, masks, reduction="sum")
        total_iou += batch_iou

        print(f"\r Epoch: {epoch} of {settings['num_epochs']-1}, "
        f"Iter.: {i+1} of {len(dataloader)}, "
        f"Avg Batch Loss: {batch_loss / batch_size:.6f}, "
        f"Avg Batch IoU : {batch_iou / batch_size:.6f}", end="")

    print()
    # FIXED: Correct averaging
    avg_loss = total_loss / len(dataloader.dataset)
    avg_iou = total_iou / len(dataloader.dataset)

    prefix = "Train" if train_mode else "Valid"
    print(f"\r Epoch: {epoch} of {settings['num_epochs']-1}, {prefix} Avg Epoch Loss: {avg_loss:.6f}", end="")
    print(f"\r Epoch: {epoch} of {settings['num_epochs']-1}, {prefix} Avg Epoch IoU : {avg_iou:.6f}", end="\n")
   
    return avg_loss, avg_iou
 
# --------------------------------------------------------------------------------
# Cosine LR Scheduler per stage
# --------------------------------------------------------------------------------
def get_cosine_scheduler(optimizer, num_epochs_in_stage, min_lr=1e-6):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_in_stage, eta_min=min_lr)

# --------------------------------------------------------------------------------
# check settings(This is like the Pre-flight checklist for model training pipeline.)
# --------------------------------------------------------------------------------
# It protects training code from crashing with weird errors later and makes sure data paths, model choices, and options are all correct.


def check_settings(original_settings):
    settings = copy.deepcopy(original_settings)


    # check if settings are correct
    assert isinstance(settings["gpu_index"], int)
    assert settings["gpu_index"] >= 0
    assert isinstance(settings["num_cpu_workers_for_dataloader"], int)
    assert settings["num_cpu_workers_for_dataloader"] > 0
    assert isinstance(settings["batch_size"], int)
    assert settings["batch_size"] > 0


    assert os.path.isdir(settings["images_dir_path"])
    assert os.path.isdir(settings["masks_dir_path"])
    assert os.path.isfile(settings["train_ids_txt"])
    assert os.path.isfile(settings["valid_ids_txt"])


    assert settings["model_architecture"] in ["UNet", "UNet_attention"]
    assert settings["loss_function"] in ["IoULoss", "BCEWithLogitsLoss", "IoUBCELoss", "BiouLoss" , "BBIoULoss", "ConservativeBBIoULoss", "Progressive"]
    assert isinstance(settings["training_augmentation"], bool)
    assert settings["model_name"] is not None
    assert settings["model_name"] != ""




# --------------------------------------------------------------------------------
# main function(to use them in command prompt)
# --------------------------------------------------------------------------------


# python train_models.py --loss_function="IoULoss" --training_augmentation=0
# python train_models.py --loss_function="BCEWithLogitsLoss" --training_augmentation=0
# python train_models.py --loss_function="IoUBCELoss" --training_augmentation=0
# python train_models.py --loss_function="IoULoss" --training_augmentation=1
# python train_models.py --loss_function="BCEWithLogitsLoss" --training_augmentation=1
# python train_models.py --loss_function="IoUBCELoss" --training_augmentation=1

def get_safe_criterion(epoch, device_str):
    """Progressive loss with BiouLoss"""
    
    if epoch < 100:
        #return BCEWithLogitsLoss(reduction="sum").to(device_str)
        return ConservativeBBIoULoss(reduction="sum").to(device_str)
    # else:
    #     # USE THE IMPROVED VERSION
    #     return ConservativeBBIoULoss(reduction="sum").to(device_str)
        
def main():
    parser = argparse.ArgumentParser(description='Train and validate a segmentation model')
    parser.add_argument('--gpu_index', default=0, type=int)
    parser.add_argument('--num_cpu_workers_for_dataloader', default=min(2, os.cpu_count()), type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--model_architecture', default='UNet', type=str)
    parser.add_argument('--training_augmentation', default=1, type=int)
    parser.add_argument('--loss_function', default='Progressive', type=str)
    
    # CRITICAL: Extended training with more patience
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--patience', default=45, type=int)
    parser.add_argument('--lr', default=1.2e-4, type=float)  # Slightly higher
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    args = parser.parse_args()

    SETTINGS = {
        "gpu_index": args.gpu_index,
        "num_cpu_workers_for_dataloader": args.num_cpu_workers_for_dataloader,
        "batch_size": args.batch_size,
        "model_architecture": args.model_architecture,
        "loss_function": args.loss_function,
        "training_augmentation": bool(args.training_augmentation),
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "image_channels": 1,
        "mask_channels": 1,
        "images_dir_path": r"E:\E Drive\BBIoULoss_Updated_V7_Liver\kvasir-seg-main\data\Updated_Liver\images",
        "masks_dir_path": r"E:\E Drive\BBIoULoss_Updated_V7_Liver\kvasir-seg-main\data\Updated_Liver\masks",
        "train_ids_txt": r"E:\E Drive\BBIoULoss_Updated_V7_Liver\kvasir-seg-main\train-val-split\train.txt",
        "valid_ids_txt": r"E:\E Drive\BBIoULoss_Updated_V7_Liver\kvasir-seg-main\train-val-split\val.txt",
    }
    
    postfix = "augmented" if SETTINGS["training_augmentation"] else "baseline"
    SETTINGS["model_name"] = f"{SETTINGS['model_architecture']}_{SETTINGS['loss_function']}_{postfix}"
    check_settings(SETTINGS)

    save_dir = r"E:\E Drive\BBIoULoss_Updated_V7_Liver\kvasir-seg-main\checkpoints"
    os.makedirs(save_dir, exist_ok=True)
   
    # Seeds
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True

    # Device
    device_type_str = "cuda" if torch.cuda.is_available() else "cpu"
    print("device_type_str:", device_type_str)
    device_str = f"{device_type_str}:{SETTINGS['gpu_index']}" if device_type_str == "cuda" else device_type_str
    print("     device_str:", device_str)

    # Model
    if SETTINGS["model_architecture"] == "UNet":
        model = UNet(channel_in=SETTINGS["image_channels"], channel_out=SETTINGS["mask_channels"])
    else:
        raise NotImplementedError
    model = model.to(device_str)
   
    # CRITICAL: Use AdamW instead of Adam for better generalization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SETTINGS["learning_rate"],
        weight_decay=SETTINGS["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8
    )
   
    use_progressive_loss = SETTINGS["loss_function"] == "Progressive"
    
    if not use_progressive_loss:
        if SETTINGS["loss_function"] == "IoULoss":
            criterion = IoULoss(reduction="sum").to(device_str)
        elif SETTINGS["loss_function"] == "BCEWithLogitsLoss":
            criterion = BCEWithLogitsLoss(reduction="sum").to(device_str)
        elif SETTINGS["loss_function"] == "IoUBCELoss":
            criterion = IoUBCELoss(reduction="sum").to(device_str)
        elif SETTINGS["loss_function"] == "BiouLoss":
            criterion = BiouLoss(reduction="sum").to(device_str)
        elif SETTINGS["loss_function"] == "ConservativeBBIoULoss":
            criterion = ConservativeBBIoULoss(reduction="sum").to(device_str)
        else:
            raise NotImplementedError

    # Augmentation
    if SETTINGS["training_augmentation"]:
        model_train_transform = train_transforms
    else:
        model_train_transform = test_transforms

    # Load data
    with open(SETTINGS["train_ids_txt"], 'r') as f:
        ids_train = [l.strip()+'.png' for l in f]
    with open(SETTINGS["valid_ids_txt"], 'r') as f:
        ids_val = [l.strip()+'.png' for l in f]
    
    custom_dataset_train = myDataSet(ids_train, SETTINGS["images_dir_path"], SETTINGS["masks_dir_path"], transforms=model_train_transform)
    custom_dataset_valid = myDataSet(ids_val, SETTINGS["images_dir_path"], SETTINGS["masks_dir_path"], transforms=test_transforms)
    
    print(f"My custom train-dataset has {len(custom_dataset_train)} elements")
    print(f"My custom valid-dataset has {len(custom_dataset_valid)} elements")
    
    dataloader_train = torch.utils.data.DataLoader(
        custom_dataset_train, batch_size=SETTINGS["batch_size"], 
        num_workers=SETTINGS["num_cpu_workers_for_dataloader"],
        shuffle=True, drop_last=False
    )
    dataloader_valid = torch.utils.data.DataLoader(
        custom_dataset_valid, batch_size=SETTINGS["batch_size"],
        num_workers=SETTINGS["num_cpu_workers_for_dataloader"],
        shuffle=False, drop_last=False
    )
    
    print(f"My custom train-dataloader has {len(dataloader_train)} batches, batch_size={dataloader_train.batch_size}")
    print(f"My custom valid-dataloader has {len(dataloader_valid)} batches, batch_size={dataloader_valid.batch_size}")

    train_losses = []
    valid_losses = []
    best_iou = 0
    best_epoch = -1
    state = {}
    
    # CRITICAL FIX: Single smooth cosine decay - NO RESTARTS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=SETTINGS["num_epochs"],
        eta_min=3e-6 
    )

    for epoch in range(SETTINGS["num_epochs"]):
        
        # Dynamic criterion selection for progressive training
        if use_progressive_loss:
            criterion = get_safe_criterion(epoch, device_str)
            
            # Print stage info
            if epoch == 0:
                print("\nProgressive Training:")
                print("Stage 1: BCE Loss (Epochs 0-49)")
            elif epoch == 50:
                print("\nStage 2: BCE + IoU Loss (Epochs 50-79)")  
            elif epoch == 65:
                print("\nStage 3: BCE + IoU + Small BiIoU (Epochs 80+)")
            
            # Reduce LR when adding BiIoU
            # if epoch == 50:
            #     current_lr = optimizer.param_groups[0]['lr']
            #     new_lr = max(current_lr * 0.75, 1.2e-5) 
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = new_lr
            #     print(f"Reduced learning rate to: {optimizer.param_groups[0]['lr']:.2e}")
        
        # # Set gradient clipping values
        # clip_grad_norm = 5.0 if (use_progressive_loss and epoch < 70) else 3.0
        
                # Apply gradient clipping only in Stage 3 (epoch >= 70)
        if use_progressive_loss and epoch >= 0:
            clip_grad_norm = 4.0   # you can tune this value
        else:
            clip_grad_norm = None  # no clipping

        epoch_avg_train_loss, epoch_avg_train_iou = train_eval_one_epoch(
            model, optimizer, criterion, dataloader_train, epoch, 
            device=device_str, settings=SETTINGS, train_mode=True, 
            clip_grad_norm=clip_grad_norm)
        
        epoch_avg_valid_loss, epoch_avg_valid_iou = train_eval_one_epoch(
            model, optimizer, criterion, dataloader_valid, epoch, 
            device=device_str, settings=SETTINGS, train_mode=False, 
            clip_grad_norm=None)
        scheduler.step()
        print(f"\nEpoch {epoch}/{SETTINGS['num_epochs']-1}")
        print(f"Train Loss: {epoch_avg_train_loss:.6f} | Train IoU: {epoch_avg_train_iou:.6f}")
        print(f"Valid Loss: {epoch_avg_valid_loss:.6f} | Valid IoU: {epoch_avg_valid_iou:.6f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
       
        train_losses.append(epoch_avg_train_loss)
        valid_losses.append(epoch_avg_valid_loss)

        # Save best model
        if epoch_avg_valid_iou > best_iou:
            best_iou = epoch_avg_valid_iou
            best_epoch = epoch
            print('Saving best model...')
            
            state['net'] = model.state_dict()
            state['iou'] = best_iou
            state['epoch'] = epoch
            state['train_losses'] = train_losses
            state['valid_losses'] = valid_losses
               
            torch.save(state, f'{save_dir}/{SETTINGS["model_name"]}.pth')
       
        elif best_epoch + SETTINGS['patience'] < epoch:
            print(f"\nEarly stopping. No improvement for {SETTINGS['patience']} epochs.\n")
            break

    # Final save and checks
    try:
        state = torch.load(f'{save_dir}/{SETTINGS["model_name"]}.pth')
        state['train_losses'] = train_losses
        state['valid_losses'] = valid_losses
        torch.save(state, f'{save_dir}/{SETTINGS["model_name"]}.pth')
        print(f'Best epoch: {best_epoch}, Best IoU: {best_iou:.6f}')
        
        model.load_state_dict(state['net'])
        print('Model loaded successfully')
        print(f'Final Validation IoU: {best_iou:.6f}')
    except Exception as e:
        print(f"Error loading final model: {e}")

if __name__ == "__main__":
    main()