# BBIoU Loss Function

BBIoU (Boundary-Balanced Intersection over Union) Loss — a boundary-aware IoU-style loss designed for image segmentation tasks. This repository contains a compact implementation that can be used as a drop-in loss for training segmentation models.

Features
- Boundary-aware: gives greater emphasis to boundary pixels to improve segmentation contour accuracy.
- IoU-based: optimizes overlap (Intersection over Union) between prediction and ground truth.
- Easy to integrate: usable as a standard PyTorch loss function (see Usage).

Contents
- Implementation: the loss implementation file(s) are in the repository root (look for `bbiou_loss.py` or similarly named modules).
- Data splits: example train/validation splits are under `data/splits/` (see `data/splits/README.md`).

Installation

1. Clone this repository:

```
git clone https://github.com/hridoynasah/bbiou-loss.git
cd bbiou-loss
```

2. Install dependencies (example for PyTorch-based usage):

```
pip install -r requirements.txt
# or
pip install torch torchvision
```

Usage

Use the implementation provided in this repository as a loss during model training. Example (PyTorch-style pseudocode):

```
from bbiou_loss import BBIoULoss

criterion = BBIoULoss()
# model outputs should be logits or probabilities depending on the implementation
outputs = model(inputs)          # shape: (N, C, H, W) or (N, 1, H, W)
loss = criterion(outputs, targets)  # targets: binary mask or integer class mask
loss.backward()
```

Notes
- The exact function signature and expected tensor shapes depend on the implementation file in this repo. If you open the implementation file, it will show whether the loss expects logits or probabilities and the required tensor shapes.
- For a recommended train/validation split, see `data/splits/README.md` which includes links to the original split used for fair comparison.

Contributing

Contributions, bug reports, and suggestions are welcome. If you'd like to improve the README or implementation, please open an issue or a pull request and include tests or reproducible examples where possible.

License

If this repository does not include a LICENSE file, add one or contact the maintainer for licensing details.

Contact

Maintainer: hridoynasah

---

This README was updated to include installation and usage guidance, and to point users to the data splits. Please review the loss implementation file(s) to confirm the exact API and adjust the usage snippet accordingly.
