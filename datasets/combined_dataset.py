"""
Dataset combinado que unifica múltiples datasets de caracteres.

Este módulo permite combinar el dataset manuscrito custom con EMNIST u otros datasets.
"""

import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets, transforms
from typing import Optional, List

from models.config import OCRConfig
from datasets.handwritten_dataset import HandwrittenDataset


class CombinedCharacterDataset:
    """
    Clase helper para crear datasets combinados de caracteres.
    
    Facilita la combinación de múltiples fuentes de datos (custom, EMNIST, etc.)
    y proporciona DataLoaders listos para entrenar.
    """
    
    def __init__(self, config: OCRConfig):
        """
        Inicializa el constructor de dataset combinado.
        
        Args:
            config: Configuración del modelo.
        """
        self.config = config
        self.train_dataset = None
        self.test_dataset = None
    
    def create_emnist_transform(self) -> transforms.Compose:
        """
        Crea las transformaciones específicas para EMNIST.
        
        EMNIST requiere rotación y flip para corregir la orientación.
        
        Returns:
            Composición de transformaciones para EMNIST.
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.rot90(x, k=-1, dims=[1, 2])),
            transforms.Lambda(lambda x: 1 - x),  # Invertir colores
            transforms.Lambda(lambda x: x.flip(-1)),  # Flip horizontal
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def create_custom_transform(self) -> transforms.Compose:
        """
        Crea las transformaciones para el dataset custom.
        
        Returns:
            Composición de transformaciones para dataset custom.
        """
        return transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def load_emnist_dataset(self, root: str = './data', split: str = 'byclass'):
        """
        Carga el dataset EMNIST.
        
        Args:
            root: Directorio donde se encuentra o descargará EMNIST.
            split: Split de EMNIST a usar ('byclass', 'bymerge', etc.).
        """
        emnist_transform = self.create_emnist_transform()
        
        self.emnist_train = datasets.EMNIST(
            root=root,
            split=split,
            train=True,
            download=True,
            transform=emnist_transform
        )
        
        self.emnist_test = datasets.EMNIST(
            root=root,
            split=split,
            train=False,
            download=True,
            transform=emnist_transform
        )
        
        print(f"\nEMNIST Dataset Cargado:")
        print(f"  Train: {len(self.emnist_train)} imágenes")
        print(f"  Test: {len(self.emnist_test)} imágenes")
    
    def load_custom_dataset(
        self,
        root_dir: str,
        train_split: float = 0.8
    ):
        """
        Carga el dataset manuscrito custom y lo divide en train/test.
        
        Args:
            root_dir: Directorio raíz del dataset custom.
            train_split: Proporción de datos para entrenamiento (0.0 a 1.0).
        """
        custom_transform = self.create_custom_transform()
        
        # Cargar dataset completo
        full_dataset = HandwrittenDataset(
            root_dir=root_dir,
            transform=custom_transform,
            config=self.config
        )
        
        # Dividir en train/test
        train_size = int(train_split * len(full_dataset))
        test_size = len(full_dataset) - train_size
        
        self.custom_train, self.custom_test = torch.utils.data.random_split(
            full_dataset,
            [train_size, test_size]
        )
        
        print(f"\nDataset Custom Cargado:")
        print(f"  Train: {train_size} imágenes")
        print(f"  Test: {test_size} imágenes")
        
        full_dataset.print_statistics()
    
    def combine_datasets(
        self,
        include_emnist: bool = True,
        include_custom: bool = True
    ):
        """
        Combina los datasets cargados.
        
        Args:
            include_emnist: Si True, incluye EMNIST en la combinación.
            include_custom: Si True, incluye dataset custom en la combinación.
        """
        train_datasets = []
        test_datasets = []
        
        if include_emnist:
            if not hasattr(self, 'emnist_train'):
                raise ValueError("EMNIST no ha sido cargado. Llama a load_emnist_dataset() primero.")
            train_datasets.append(self.emnist_train)
            test_datasets.append(self.emnist_test)
        
        if include_custom:
            if not hasattr(self, 'custom_train'):
                raise ValueError("Dataset custom no ha sido cargado. Llama a load_custom_dataset() primero.")
            train_datasets.append(self.custom_train)
            test_datasets.append(self.custom_test)
        
        if not train_datasets:
            raise ValueError("No se han especificado datasets para combinar.")
        
        # Combinar usando ConcatDataset
        self.train_dataset = ConcatDataset(train_datasets)
        self.test_dataset = ConcatDataset(test_datasets)
        
        print(f"\nDatasets Combinados:")
        print(f"  Train total: {len(self.train_dataset)} imágenes")
        print(f"  Test total: {len(self.test_dataset)} imágenes")
    
    def get_dataloaders(
        self,
        batch_size: int = 128,
        shuffle_train: bool = True,
        num_workers: int = 0
    ):
        """
        Crea DataLoaders para los datasets combinados.
        
        Args:
            batch_size: Tamaño del batch.
            shuffle_train: Si True, baraja los datos de entrenamiento.
            num_workers: Número de workers para carga de datos.
        
        Returns:
            Tupla (train_loader, test_loader).
        """
        if self.train_dataset is None or self.test_dataset is None:
            raise ValueError("Los datasets no han sido combinados. Llama a combine_datasets() primero.")
        
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers
        )
        
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, test_loader


def create_handwritten_only_dataloaders(
    custom_root: str,
    config: OCRConfig,
    batch_size: int = 128,
    train_split: float = 0.8
):
    """
    Función helper para crear DataLoaders solo con dataset manuscrito custom.
    
    Args:
        custom_root: Directorio raíz del dataset custom.
        config: Configuración del modelo.
        batch_size: Tamaño del batch.
        train_split: Proporción de train/test.
    
    Returns:
        Tupla (train_loader, test_loader).
    """
    combined = CombinedCharacterDataset(config)
    combined.load_custom_dataset(custom_root, train_split)
    combined.combine_datasets(include_emnist=False, include_custom=True)
    return combined.get_dataloaders(batch_size=batch_size)


def create_combined_dataloaders(
    custom_root: str,
    config: OCRConfig,
    emnist_root: str = './data',
    batch_size: int = 128,
    train_split: float = 0.8
):
    """
    Función helper para crear DataLoaders con custom + EMNIST.
    
    Args:
        custom_root: Directorio raíz del dataset custom.
        config: Configuración del modelo.
        emnist_root: Directorio para EMNIST.
        batch_size: Tamaño del batch.
        train_split: Proporción de train/test para dataset custom.
    
    Returns:
        Tupla (train_loader, test_loader).
    """
    combined = CombinedCharacterDataset(config)
    combined.load_custom_dataset(custom_root, train_split)
    combined.load_emnist_dataset(emnist_root)
    combined.combine_datasets(include_emnist=True, include_custom=True)
    return combined.get_dataloaders(batch_size=batch_size)