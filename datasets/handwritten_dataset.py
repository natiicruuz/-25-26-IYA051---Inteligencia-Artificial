"""
Dataset de caracteres manuscritos custom.

Este módulo define el dataset para cargar caracteres manuscritos organizados
en la estructura: mayúsculas/, minúsculas/, números/ con subcarpetas por carácter.
"""

import os
from typing import Tuple, Callable, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset

from models.config import OCRConfig


class HandwrittenDataset(Dataset):
    """
    Dataset para caracteres manuscritos del dataset custom.
    
    Estructura esperada:
        root/
        ├── mayúsculas/
        │   ├── A/
        │   │   ├── A_Nombre_Apellido.png
        │   └── ...
        ├── minúsculas/
        │   ├── a/
        │   │   ├── a_Nombre_Apellido.png
        │   └── ...
        └── números/
            ├── 0/
            │   ├── 0_Nombre_Apellido.png
            └── ...
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        config: Optional[OCRConfig] = None
    ):
        """
        Inicializa el dataset.
        
        Args:
            root_dir: Directorio raíz del dataset.
            transform: Transformaciones a aplicar a las imágenes.
            config: Configuración con el mapeo de caracteres.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.config = config or OCRConfig(num_classes=64)
        self.char_mapping = self.config.get_character_mapping()
        
        # Cargar todas las rutas de imágenes y sus etiquetas
        self.image_paths = []
        self.labels = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Carga las rutas de todas las imágenes y sus etiquetas."""
        
        # Definir subdirectorios (aceptar con y sin tildes)
        subdirs = {}
        
        # Números (con y sin tilde)
        for variant in ['números', 'numeros']:
            path = os.path.join(self.root_dir, variant)
            if os.path.exists(path):
                subdirs['numeros'] = path
                break
        
        # Mayúsculas (con y sin tilde)
        for variant in ['mayúsculas', 'mayusculas']:
            path = os.path.join(self.root_dir, variant)
            if os.path.exists(path):
                subdirs['mayusculas'] = path
                break
        
        # Minúsculas (con y sin tilde)
        for variant in ['minúsculas', 'minusculas']:
            path = os.path.join(self.root_dir, variant)
            if os.path.exists(path):
                subdirs['minusculas'] = path
                break
        
        for subdir_name, subdir_path in subdirs.items():
            if not os.path.exists(subdir_path):
                print(f"Advertencia: No se encontró el directorio {subdir_path}")
                continue
            
            # Recorrer cada subcarpeta de carácter
            for char_folder in os.listdir(subdir_path):
                char_folder_path = os.path.join(subdir_path, char_folder)
                
                if not os.path.isdir(char_folder_path):
                    continue
                
                # El nombre de la carpeta es el carácter
                character = char_folder
                
                # Verificar que el carácter esté en el mapeo
                if character not in self.char_mapping:
                    print(f"Advertencia: Carácter '{character}' no está en el mapeo")
                    continue
                
                # Obtener el índice del carácter
                label = self.char_mapping.index(character)
                
                # Cargar todas las imágenes de esta carpeta
                for img_file in os.listdir(char_folder_path):
                    if img_file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(char_folder_path, img_file)
                        self.image_paths.append(img_path)
                        self.labels.append(label)
        
        print(f"\nDataset Manuscrito Cargado:")
        print(f"  Total de imágenes: {len(self.image_paths)}")
        print(f"  Número de clases: {len(set(self.labels))}")
    
    def __len__(self) -> int:
        """Retorna el número total de muestras."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Obtiene una muestra del dataset.
        
        Args:
            idx: Índice de la muestra.
        
        Returns:
            Tupla (imagen_tensor, etiqueta).
        """
        # Cargar imagen
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convertir a escala de grises
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        
        # Obtener etiqueta
        label = self.labels[idx]
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        """
        Calcula la distribución de clases en el dataset.
        
        Returns:
            Diccionario con carácter: cantidad de muestras.
        """
        from collections import Counter
        
        label_counts = Counter(self.labels)
        char_distribution = {}
        
        for label, count in label_counts.items():
            char = self.char_mapping[label]
            char_distribution[char] = count
        
        return char_distribution
    
    def print_statistics(self):
        """Imprime estadísticas del dataset."""
        distribution = self.get_class_distribution()
        
        print(f"\n{'='*60}")
        print(f"Estadísticas del Dataset Manuscrito")
        print(f"{'='*60}")
        print(f"Total de imágenes: {len(self)}")
        print(f"Número de clases: {len(distribution)}")
        print(f"\nDistribución por tipo:")
        
        # Números
        nums = {k: v for k, v in distribution.items() if k.isdigit()}
        print(f"  Números (0-9): {sum(nums.values())} imágenes")
        
        # Mayúsculas
        upper = {k: v for k, v in distribution.items() if k.isupper() and k.isalpha()}
        print(f"  Mayúsculas: {sum(upper.values())} imágenes")
        
        # Minúsculas
        lower = {k: v for k, v in distribution.items() if k.islower()}
        print(f"  Minúsculas: {sum(lower.values())} imágenes")
        
        print(f"{'='*60}\n")