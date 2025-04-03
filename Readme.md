# PyTorch Lightning Segmentation Model

A PyTorch Lightning implementation of a semantic segmentation model using U-Net architecture with ResNet34 encoder. This project demonstrates how to build a robust segmentation pipeline with proper data handling, augmentation, and monitoring.

## Features

- 🚀 **PyTorch Lightning** framework for clean, organized deep learning code
- 🖼️ **Image Segmentation** using U-Net with pretrained ResNet34 encoder
- 📊 **TensorBoard** logging for metrics, learning rates, and prediction visualization
- 🔄 **Data Augmentation** with random flips, rotations, and color adjustments
- 📈 **Advanced Training Features** including learning rate scheduling and model checkpointing
- 📏 **IoU Metrics** for evaluating segmentation quality

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/pytorch-lightning-segmentation.git
cd pytorch-lightning-segmentation

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision pytorch-lightning segmentation-models-pytorch torchmetrics matplotlib tensorboard
```

## Dataset Structure

The code expects your dataset to be organized in the following structure:

```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks/
    ├── image1_mask.png
    ├── image2_mask.png
    └── ...
```

- Images should be in JPG format
- Masks should be in PNG format with the naming convention `<image_name>_mask.png`
- Masks should contain pixel values corresponding to class indices (e.g., 0 for background, 1 for foreground)

## Usage

### Training

To train the segmentation model:

```python
python train.py
```

You can modify the following parameters in the `main()` function:

- `batch_size`: Number of samples per batch
- `img_size`: Image resolution for training
- `num_workers`: Number of data loading workers
- `encoder_name`: Encoder backbone for U-Net (e.g., "resnet34", "resnet50")
- `learning_rate`: Initial learning rate
- `max_epochs`: Maximum number of training epochs

### Monitoring Training

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir lightning_logs
```

This will provide:
- Loss and IoU metrics for training and validation
- Learning rate changes over time
- Visual comparison of original images, ground truth masks, and predictions

### Using Your Trained Model

```python
import torch
import pytorch_lightning as pl
from segmentation_model import SegmentationModel

# Load the trained model
model = SegmentationModel.load_from_checkpoint("final_segmentation_model.ckpt")
model.eval()

# Make predictions
image = torch.randn(1, 3, 256, 256)  # Example input
with torch.no_grad():
    output = model(image)
    prediction = torch.argmax(output, dim=1)
```

## Model Architecture

The model uses Segmentation Models PyTorch (SMP) implementation of U-Net with a ResNet34 encoder pretrained on ImageNet. The architecture consists of:

- Encoder: ResNet34 for feature extraction
- Decoder: U-Net decoder for upsampling and feature merging
- Output: Segmentation mask with class probabilities

## Customization

### Modifying the Model

To use a different encoder or architecture:

```python
# In the SegmentationModel class initialization
self.model = smp.Unet(  # or smp.DeepLabV3Plus, smp.FPN, etc.
    encoder_name="resnet50",  # Options: resnet18, resnet50, efficientnet-b0, etc.
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes
)
```

### Multi-class Segmentation

To adapt the model for more than two classes:

1. Update `num_classes` in `SegmentationModel`
2. Ensure your mask files contain the correct class indices (0, 1, 2, ...)
3. Adjust visualization code if needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If this code helps with your research, please cite:

```
@software{pytorch_lightning_segmentation,
  author = {Your Name},
  title = {PyTorch Lightning Segmentation Model},
  year = {2025},
  url = {https://github.com/yourusername/pytorch-lightning-segmentation}
}
```

## Acknowledgments

- [PyTorch Lightning](https://lightning.ai/)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [TorchMetrics](https://torchmetrics.readthedocs.io/)