#%%writefile /kaggle/working/BBIoULoss_Updated_V7_Liver/kvasir-seg-main/utils/metrics.py
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
#import cv2
import numpy as np

def reduce_metric(metric, reduction='mean'):
    """
    If "sum" or "mean" Reduces a metric tensor in the 0th dimention (batch_size)
    Otherwise returns the metric tensor as is.
    """
    if reduction == 'mean':
        return metric.mean(0)
    elif reduction == 'sum':
        return metric.sum(0)
    elif reduction == 'none':
        return metric
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

def iou_pytorch_eval(outputs: torch.Tensor, labels: torch.Tensor, reduction='mean'):
    outputs = torch.sigmoid(outputs)
    outputs = outputs > 0.5
    outputs = outputs.squeeze(1).byte()
    labels = labels.squeeze(1).byte()

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return reduce_metric(iou, reduction)


def iou_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor, reduction='mean'):
    # intersection = tp
    # union = tp + fp + fn
    # iou = tp / (tp + fp + fn) = intersection / union

    # BATCH x H x W
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5
    labels = labels > 0.5

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    return reduce_metric(iou, reduction)


def dice_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor, reduction='mean'):
    # intersection = tp
    # union = tp + fp + fn
    # dice = 2 * tp / (2 * tp + fp + fn) = 2*intersection / (intersection + union)

    # BATCH x H x W
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5
    labels = labels > 0.5

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    dice = (2*intersection + SMOOTH) / (intersection + union + SMOOTH)  # We smooth our devision to avoid 0/0

    return reduce_metric(dice, reduction)


def precision_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor, reduction='mean'):
    # intersection = tp
    # tpfp = tp + fp
    # precision = tp / (tp + fp) = intersection / tpfp

    # BATCH x H x W
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5
    labels = labels > 0.5

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    tpfp = (labels).float().sum((1, 2))                    # Will be zero if both are 0
    precision = (intersection + SMOOTH) / (tpfp + SMOOTH)  # We smooth our devision to avoid 0/0

    return reduce_metric(precision, reduction)


def recall_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor, reduction='mean'):
    # intersection = tp
    # tpfn = tp + fn
    # recall = tp / (tp + fn) = intersection / tpfn

    # BATCH x H x W
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5
    labels = labels > 0.5

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    tpfn = (outputs).float().sum((1, 2))                   # Will be zero if both are 0
    recall = (intersection + SMOOTH) / (tpfn + SMOOTH)     # We smooth our devision to avoid 0/0

    return reduce_metric(recall, reduction)


def fbeta_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor, beta:float, reduction='mean'):
    # intersection = tp
    #
    # tpfp = tp + fp
    # precision = tp / (tp + fp) = intersection / tpfp
    #
    # tpfn = tp + fn
    # recall = tp / (tp + fn) = intersection / tpfn
    #
    # fbeta = (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
    # https://www.quora.com/What-is-the-F2-score-in-machine-learning

    # BATCH x H x W
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5
    labels = labels > 0.5

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0

    tpfn = (outputs).float().sum((1, 2))                   # Will be zero if both are 0
    recall = (intersection + SMOOTH) / (tpfn + SMOOTH)     # We smooth our devision to avoid 0/0

    tpfp = (labels).float().sum((1, 2))                    # Will be zero if both are 0
    precision = (intersection + SMOOTH) / (tpfp + SMOOTH)  # We smooth our devision to avoid 0/0

    f_beta = (1 + beta ** 2) * (precision * recall) / ((beta **2 * precision) + recall)

    return reduce_metric(f_beta, reduction)


def accuracy_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor):

    # BATCH x H x W
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5
    labels = labels > 0.5

    acc = (outputs == labels).float().mean((1, 2))

    return reduce_metric(acc, reduction='mean')

# if we care about both classes
def binary_both_classes_iou_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor):
    # intersection = tp
    # union = tp + fp + fn
    # iou = tp / (tp + fp + fn) = intersection / union

    # BATCH x H x W, need because we process images sequentially in a for-loop
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)
    SMOOTH = 1e-8

    # thresholding since that's how we will make predictions on new imputs (class 0)
    outputs0 = outputs < 0.5
    labels0 = labels < 0.5
    intersection = (outputs0 & labels0).float().sum((1, 2))  # Will be zero if Truth=1 or Prediction=1
    union = (outputs0 | labels0).float().sum((1, 2))         # Will be zero if both are 1
    iou0 = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    # thresholding since that's how we will make predictions on new imputs (class 1)
    outputs1 = outputs > 0.5
    labels1 = labels > 0.5
    intersection = (outputs1 & labels1).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs1 | labels1).float().sum((1, 2))         # Will be zero if both are 0
    iou1 = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    # average iou of both classes - can add unequal weights if we care more about one class
    weighted_iou = (iou0 + iou1) / 2
    return reduce_metric(weighted_iou, reduction='mean')



class IoULoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(IoULoss, self).__init__()
        self.reduction = reduction

    def forward(self, preds, targets, smooth=1e-6):
        if preds.shape != targets.shape:
            targets = F.interpolate(targets, size=preds.shape[2:], mode="nearest")

        preds = torch.sigmoid(preds).view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
        iou = (intersection + smooth) / (union + smooth)
        loss = 1 - iou

        return loss.mean() if self.reduction == "mean" else loss.sum()

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BCEWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='none')

    def leave_only_batch_and_flatten(self, inputs, targets):
        inputs = inputs.view(inputs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)
        return inputs, targets

    def forward(self, inputs, targets):
        inputs, targets = self.leave_only_batch_and_flatten(inputs, targets)
        BCE_loss = self.BCEWithLogitsLoss(inputs, targets)
        BCE_loss = BCE_loss.mean(1)
        return reduce_metric(BCE_loss, reduction=self.reduction)

class BiouLoss(nn.Module):
    """
    Fixed BiIoU Loss with proper value ranges.
    
    Key fixes:
    1. Returns PER-SAMPLE AVERAGE (not sum) for consistency with other losses
    2. Binarize masks before Sobel for cleaner edges
    3. Adaptive thresholding handles varying lesion sizes
    4. Values now correctly in [0, 1] range
    """
    
    def __init__(self, reduction="mean", dilation_ratio=0.02, eps=1e-6, 
                 edge_threshold=0.05, use_adaptive_threshold=True):
        super(BiouLoss, self).__init__()
        
        self.reduction = reduction
        self.dilation_ratio = dilation_ratio
        self.eps = eps
        self.edge_threshold = edge_threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        
        # Standard Sobel kernels
        sobel_x = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        sobel_y = torch.tensor([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
    
    def get_edges(self, mask):
        """
        Edge detection with adaptive thresholding.
        """
        # Binarize first for cleaner edges
        binary_mask = (mask > 0.5).float()
        
        # Apply Sobel filters
        edge_x = F.conv2d(binary_mask, self.sobel_x, padding=1)
        edge_y = F.conv2d(binary_mask, self.sobel_y, padding=1)
        
        # Gradient magnitude
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + self.eps)
        
        if self.use_adaptive_threshold:
            # Adaptive threshold: use percentile per image
            B = edge_magnitude.shape[0]
            binary_edges = torch.zeros_like(edge_magnitude)
            
            for i in range(B):
                edge_flat = edge_magnitude[i].flatten()
                
                if edge_flat.max() > self.eps:
                    # edge_threshold as percentile (0.05 = keep top 95%)
                    percentile = 1.0 - self.edge_threshold
                    threshold = torch.quantile(edge_flat, percentile)
                    binary_edges[i] = (edge_magnitude[i] > threshold).float()
        else:
            # Fixed threshold
            binary_edges = (edge_magnitude > self.edge_threshold).float()
        
        return binary_edges
    
    def dilate(self, edge, kernel_size):
        """Morphological dilation using max pooling."""
        return F.max_pool2d(edge, kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, inputs, targets):
        """
        Compute BiIoU loss with proper reduction.
        
        CRITICAL: Returns mean of per-sample losses for "sum" reduction mode
        to match behavior of other losses in your codebase.
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(inputs)
        
        # Resize targets if needed
        if targets.shape != probs.shape:
            targets = F.interpolate(targets, size=probs.shape[2:], mode="nearest")
        
        B, _, H, W = probs.shape
        
        # Compute dilation kernel size
        kernel_size = max(3, int(max(H, W) * self.dilation_ratio))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Get edges
        P_edge = self.get_edges(probs)
        G_edge = self.get_edges(targets)
        
        # Check if edges were detected
        P_edge_count = P_edge.sum()
        G_edge_count = G_edge.sum()
        
        if P_edge_count < 1.0 or G_edge_count < 1.0:
            # No edges detected - return zero loss with gradient
            if self.reduction == "sum":
                return torch.tensor(0.0, device=inputs.device, requires_grad=True)
            else:
                return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Dilate edges
        P_dil = self.dilate(P_edge, kernel_size)
        G_dil = self.dilate(G_edge, kernel_size)
        
        # Compute boundary intersection (symmetric)
        intersection = (G_dil * P_edge).sum(dim=(1,2,3)) + (P_dil * G_edge).sum(dim=(1,2,3))
        
        # Compute union
        union = G_edge.sum(dim=(1,2,3)) + P_edge.sum(dim=(1,2,3)) + self.eps
        
        # Boundary IoU per sample (values in [0, 1])
        biou = intersection / union
        
        # Loss is 1 - BiIoU (values in [0, 1])
        loss = 1.0 - biou
        
        # CRITICAL FIX: For "sum" reduction, return MEAN (not sum)
        # This matches the behavior of your other losses which divide by dataset size
        if self.reduction == "sum":
            # Return mean of per-sample losses
            # Your training loop will sum these across batches and divide by dataset size
            return loss.mean()
        elif self.reduction == "mean":
            return loss.mean()
        else:  # "none"
            return loss



class IoUBCELoss(nn.Module):
    def __init__(self, reduction="mean", bce_weight=0.5, iou_weight=0.5):
        super(IoUBCELoss, self).__init__()
        self.reduction = reduction
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.bce = BCEWithLogitsLoss(reduction=reduction)
        self.iou = IoULoss(reduction=reduction)

    def forward(self, preds, targets):
        if preds.shape != targets.shape:
            targets = F.interpolate(targets, size=preds.shape[2:], mode="nearest")

        bce_loss = self.bce(preds, targets)
        iou_loss = self.iou(preds, targets)
        
        total_loss = self.bce_weight * bce_loss + self.iou_weight * iou_loss
        return total_loss


class ConservativeBBIoULoss(nn.Module):
    
    def __init__(self, reduction='mean', edge_threshold=0.05, dilation_ratio=0.02):
        super(ConservativeBBIoULoss, self).__init__()
        self.reduction = reduction
        
        self.bce_loss = BCEWithLogitsLoss(reduction=reduction)
        self.iou_loss = IoULoss(reduction=reduction)
        self.biou_loss = BiouLoss(
            reduction=reduction,
            dilation_ratio=dilation_ratio,
            edge_threshold=edge_threshold,
            use_adaptive_threshold=True
        )

    def forward(self, inputs, targets):

        if inputs.shape != targets.shape:
            targets = F.interpolate(targets, size=inputs.shape[2:], mode="nearest")

        bce_val = self.bce_loss(inputs, targets)
        iou_val = self.iou_loss(inputs, targets)
        biou_val = self.biou_loss(inputs, targets)
        
        # Keep your original weights
        #total_loss = 0.10 * bce_val +  ((0.50*iou_val + 0.40* biou_val)/2)
        #total_loss =  bce_val +  ((0.70*iou_val + 0.30* biou_val)/2)
        total_loss =  bce_val +  ((0.60*iou_val + 0.40* biou_val)/2)
        
        return total_loss

