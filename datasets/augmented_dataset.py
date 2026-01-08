"""
Dataset wrapper que aplica augmentation a un dataset existente.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class AugmentedDatasetWrapper(Dataset):
    """
    Wrapper que aplica data augmentation a un dataset existente.
    
    Útil para aplicar augmentation solo al subset de entrenamiento
    sin modificar el dataset base.
    """
    
    def __init__(self, base_dataset, augmentation_transform=None):
        """
        Args:
            base_dataset: Dataset base (ej: NormalizedDataset o EMNIST)
            augmentation_transform: Transform de augmentation a aplicar
        """
        self.base_dataset = base_dataset
        self.augmentation = augmentation_transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Obtener imagen y label del dataset base
        image, label = self.base_dataset[idx]
        
        # Si hay augmentation, aplicarlo
        if self.augmentation:
            # Convertir tensor a PIL Image
            if isinstance(image, torch.Tensor):
                # Desnormalizar si está normalizado
                image_np = image.squeeze().numpy()
                
                # Escalar a 0-255
                image_np = ((image_np + 1) * 127.5).astype(np.uint8)
                
                # Convertir a PIL
                pil_image = Image.fromarray(image_np, mode='L')
                
                # Aplicar augmentation
                augmented = self.augmentation(pil_image)
                
                # Convertir de vuelta a tensor normalizado
                if isinstance(augmented, Image.Image):
                    augmented = transforms.ToTensor()(augmented)
                
                # Re-normalizar a [-1, 1]
                augmented = (augmented - 0.5) / 0.5
                
                return augmented, label
        
        return image, label


def get_augmentation_transform():
    """
    Transformaciones de data augmentation.
    
    Returns:
        Transform de torchvision con augmentations.
    """
    return transforms.Compose([
        # Rotación leve
        transforms.RandomRotation(
            degrees=10,
            fill=255
        ),
        
        # Transformación afín
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            fill=255
        ),
        
        # Perspectiva (50% probabilidad)
        transforms.RandomPerspective(
            distortion_scale=0.2,
            p=0.5,
            fill=255
        ),
        
        # Variación de brillo
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.15
        ),
    ])