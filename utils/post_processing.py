#import cv2
import numpy as np
import torch

def boundary_refinement(prediction_tensor, kernel_size=3, gaussian_kernel=3):
    """
    Refines segmentation boundaries using morphological operations
    
    Args:
        prediction_tensor: torch.Tensor of shape (B, 1, H, W) with values in [0, 1]
        kernel_size: Size of morphological kernel (default: 3)
        gaussian_kernel: Size of Gaussian smoothing kernel (default: 3)
    
    Returns:
        torch.Tensor: Refined predictions with same shape as input
    """
    device = prediction_tensor.device
    batch_size = prediction_tensor.shape[0]
    refined_batch = []
    
    # Process each image in batch
    for b in range(batch_size):
        # Convert to numpy and binarize
        pred_np = prediction_tensor[b, 0].cpu().numpy()
        pred_binary = (pred_np > 0.5).astype(np.uint8) * 255
        
        # Create morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Step 1: Close small holes inside mask
        closed = cv2.morphologyEx(pred_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Step 2: Remove small noise outside mask  
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Step 3: Smooth boundaries with Gaussian blur
        if gaussian_kernel > 0:
            smoothed = cv2.GaussianBlur(opened.astype(np.float32), (gaussian_kernel, gaussian_kernel), 0)
            # Re-binarize after smoothing
            refined = (smoothed > 127).astype(np.float32) / 255.0
        else:
            refined = opened.astype(np.float32) / 255.0
        
        refined_batch.append(refined)
    
    # Convert back to tensor
    refined_tensor = torch.from_numpy(np.stack(refined_batch)).unsqueeze(1).float().to(device)
    return refined_tensor