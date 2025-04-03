import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from torchmetrics import JaccardIndex
import matplotlib.pyplot as plt
import random


# Dataset with transform maps
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = []
        self.mask_map = {f.replace('_mask.png', ''): f for f in os.listdir(mask_dir) if f.endswith('_mask.png')}
        
        all_files = sorted(os.listdir(image_dir))
        for img in all_files:
            if not img.endswith('.jpg'):
                print(f"Skipping {img} - not a .jpg file")
                continue
            img_path = os.path.join(self.image_dir, img)
            if not os.path.isfile(img_path):
                print(f"Skipping {img} - not a file")
                continue
                
            base_name = img.split('.rf')[0] if '.rf' in img else img.split('.')[0]
            for mask_base in self.mask_map.keys():
                if base_name in mask_base:
                    self.images.append(img)
                    break
            else:
                print(f"Skipping {img} - no matching mask found")
        
        print(f"Dataset size: {len(self.images)} images with masks")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        base_name = img_name.split('.rf')[0] if '.rf' in img_name else img_name.split('.')[0]
        mask_name = next((m for m in self.mask_map.values() if base_name in m), None)
        if not mask_name:
            raise FileNotFoundError(f"No mask for {img_name}")
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Apply transforms (both to image and mask)
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask


# Custom transform that applies the same transform to both image and mask
class SegmentationTransform:
    def __init__(self, img_size=256, augment=False):
        self.img_size = img_size
        self.augment = augment
    
    def __call__(self, img, mask):
        # Resize
        img = TF.resize(img, (self.img_size, self.img_size))
        mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)
        
        if self.augment:
            # Random horizontal flipping
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            
            # Random vertical flipping
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            
            # Random rotation
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)
                
            # Random brightness and contrast (image only)
            if random.random() > 0.5:
                img = TF.adjust_brightness(img, brightness_factor=random.uniform(0.8, 1.2))
                img = TF.adjust_contrast(img, contrast_factor=random.uniform(0.8, 1.2))
        
        # Convert to tensor
        img = TF.to_tensor(img)
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        
        # Normalize image
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return img, mask


# PyTorch Lightning DataModule
class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, image_dir, mask_dir, batch_size=4, img_size=256, num_workers=4):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.train_transform = SegmentationTransform(img_size=img_size, augment=True)
        self.val_transform = SegmentationTransform(img_size=img_size, augment=False)

    def setup(self, stage=None):
        # Create dataset
        full_dataset = SegmentationDataset(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            transform=self.train_transform
        )
        
        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        # Override dataset transforms after splitting
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Create separate datasets with appropriate transforms
        train_full = SegmentationDataset(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            transform=self.train_transform
        )
        
        val_full = SegmentationDataset(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            transform=self.val_transform
        )
        
        # Use the same indices from the random split
        self.train_dataset = torch.utils.data.Subset(train_full, self.train_dataset.indices)
        self.val_dataset = torch.utils.data.Subset(val_full, self.val_dataset.indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers
        )


# Lightning Module
class SegmentationModel(pl.LightningModule):
    def __init__(self, encoder_name="resnet34", num_classes=2, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.iou_metric = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        
        # Calculate IoU
        preds = torch.argmax(outputs, dim=1)
        iou = self.iou_metric(preds, masks)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_iou', iou, prog_bar=True)
        
        # Log images to TensorBoard periodically
        if batch_idx % 50 == 0:
            self._log_images(images, masks, preds, "train")
            
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        
        # Calculate IoU
        preds = torch.argmax(outputs, dim=1)
        iou = self.iou_metric(preds, masks)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_iou', iou, prog_bar=True)
        
        # Log images to TensorBoard
        if batch_idx == 0:
            self._log_images(images, masks, preds, "val")
            
        return {"val_loss": loss, "val_iou": iou}
    
    def _log_images(self, images, masks, preds, stage):
        # Log a subset of images to TensorBoard
        img_idx = np.random.choice(images.size(0))
        
        # Denormalize image for visualization
        img = images[img_idx].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        mask = masks[img_idx].cpu().numpy()
        pred = preds[img_idx].cpu().numpy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img)
        axes[0].set_title("Image")
        axes[0].axis("off")
        
        axes[1].imshow(mask, cmap="viridis")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        
        axes[2].imshow(pred, cmap="viridis")
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        
        # Log figure to TensorBoard
        self.logger.experiment.add_figure(f"{stage}_predictions", fig, self.global_step)
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }


# Main training script
def main():
    # Data module
    data_module = SegmentationDataModule(
        image_dir="dataset/images/",
        mask_dir="dataset/masks/",
        batch_size=8,
        img_size=256,
        num_workers=4
    )
    
    # Model
    model = SegmentationModel(
        encoder_name="resnet34", 
        num_classes=2,
        learning_rate=0.001
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Logger
    logger = TensorBoardLogger("lightning_logs", name="segmentation")
    
    # Trainer
    trainer = pl.Trainer(
        accelerator="auto",  # Uses GPU if available, otherwise CPU
        devices=1,
        max_epochs=20,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Save final model
    trainer.save_checkpoint("final_segmentation_model.ckpt")
    print("Training completed!")


if __name__ == "__main__":
    main()