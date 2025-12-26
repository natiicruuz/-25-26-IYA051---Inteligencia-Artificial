"""
Dataset para caracteres manuscritos con estructura normalizada simple.

Este módulo carga datasets con estructura plana donde cada carpeta
representa directamente un carácter (sin subcarpetas mayúsculas/minúsculas/números).

Estructura esperada:
    normalized/
    ├── 0/
    │   ├── imagen1.png
    ├── A/
    ├── a/
    └── ...
"""

import os
from typing import Tuple, Callable, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets

from models.config import OCRConfig


class NormalizedDataset(Dataset):
    """
    Dataset para caracteres manuscritos con estructura plana normalizada.
    
    Esta clase es un wrapper sobre ImageFolder que maneja automáticamente
    el mapeo de carpetas a índices de caracteres.
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        config: Optional[OCRConfig] = None
    ):
        """
        Inicializa el dataset normalizado.
        
        Args:
            root_dir: Directorio raíz con subcarpetas por carácter.
            transform: Transformaciones a aplicar a las imágenes.
            config: Configuración con el mapeo de caracteres.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.config = config or OCRConfig(num_classes=64)
        self.char_mapping = self.config.get_character_mapping()
        
        # Usar ImageFolder de PyTorch (más eficiente)
        self.image_folder = datasets.ImageFolder(
            root=root_dir,
            transform=transform
        )
        
        # Crear mapeo de carpetas a índices correctos
        self.folder_to_char = {}
        self.char_to_index = {}
        
        for folder_name, folder_idx in self.image_folder.class_to_idx.items():
            # folder_name es el nombre de la carpeta (ej: 'A', 'a', '0')
            if folder_name in self.char_mapping:
                correct_idx = self.char_mapping.index(folder_name)
                self.folder_to_char[folder_idx] = correct_idx
                self.char_to_index[folder_name] = correct_idx
            else:
                print(f"⚠️  Advertencia: Carácter '{folder_name}' no está en el mapeo")
        
        print(f"\n✓ Dataset Normalizado Cargado:")
        print(f"  Total de imágenes: {len(self.image_folder)}")
        print(f"  Carpetas encontradas: {len(self.image_folder.classes)}")
        print(f"  Caracteres mapeados: {len(self.folder_to_char)}")
    
    def __len__(self) -> int:
        """Retorna el número total de muestras."""
        return len(self.image_folder)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Obtiene una muestra del dataset.
        
        Args:
            idx: Índice de la muestra.
        
        Returns:
            Tupla (imagen_tensor, etiqueta_correcta).
        """
        image, folder_idx = self.image_folder[idx]
        
        # Mapear el índice de carpeta al índice correcto del carácter
        correct_label = self.folder_to_char.get(folder_idx, folder_idx)
        
        return image, correct_label
    
    def get_class_distribution(self) -> dict:
        """
        Calcula la distribución de clases en el dataset.
        
        Returns:
            Diccionario con carácter: cantidad de muestras.
        """
        from collections import Counter
        
        # Contar imágenes por carpeta
        folder_counts = Counter()
        for _, folder_idx in self.image_folder.imgs:
            folder_counts[folder_idx] += 1
        
        # Mapear a caracteres
        char_distribution = {}
        for folder_idx, count in folder_counts.items():
            if folder_idx in self.folder_to_char:
                correct_idx = self.folder_to_char[folder_idx]
                char = self.char_mapping[correct_idx]
                char_distribution[char] = count
        
        return char_distribution
    
    def print_statistics(self):
        """Imprime estadísticas del dataset."""
        distribution = self.get_class_distribution()
        
        print(f"\n{'='*60}")
        print(f"Estadísticas del Dataset Normalizado")
        print(f"{'='*60}")
        print(f"Total de imágenes: {len(self)}")
        print(f"Número de clases: {len(distribution)}")
        print(f"\nDistribución por tipo:")
        
        # Números
        nums = {k: v for k, v in distribution.items() if k.isdigit()}
        if nums:
            print(f"  Números (0-9): {sum(nums.values())} imágenes")
        
        # Mayúsculas
        upper = {k: v for k, v in distribution.items() if k.isupper() and k.isalpha()}
        if upper:
            print(f"  Mayúsculas: {sum(upper.values())} imágenes")
        
        # Minúsculas
        lower = {k: v for k, v in distribution.items() if k.islower()}
        if lower:
            print(f"  Minúsculas: {sum(lower.values())} imágenes")
        
        # Mostrar clases con menos imágenes (posibles problemas)
        print(f"\nClases con menos de 10 imágenes:")
        low_count = {k: v for k, v in distribution.items() if v < 10}
        if low_count:
            for char, count in sorted(low_count.items()):
                print(f"  '{char}': {count} imágenes ⚠️")
        else:
            print(f"  ✓ Todas las clases tienen >= 10 imágenes")
        
        print(f"{'='*60}\n")


def verify_normalized_dataset(root_dir: str):
    """
    Función helper para verificar el dataset normalizado antes de entrenar.
    
    Args:
        root_dir: Directorio raíz del dataset normalizado.
    """
    from torchvision import transforms
    
    print("\n" + "="*70)
    print(" VERIFICACIÓN DE DATASET NORMALIZADO")
    print("="*70)
    
    if not os.path.exists(root_dir):
        print(f"\n❌ ERROR: No se encontró el directorio: {root_dir}")
        return False
    
    print(f"\nDirectorio: {root_dir}")
    
    # Verificar que haya subcarpetas
    subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    
    print(f"✓ Carpetas encontradas: {len(subfolders)}")
    
    if len(subfolders) == 0:
        print("❌ ERROR: No hay carpetas de caracteres en el directorio")
        return False
    
    # Mostrar algunas carpetas
    print(f"\nPrimeras 10 carpetas: {sorted(subfolders)[:10]}")
    
    # Intentar cargar el dataset
    try:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        config = OCRConfig(num_classes=64)
        dataset = NormalizedDataset(
            root_dir=root_dir,
            transform=transform,
            config=config
        )
        
        dataset.print_statistics()
        
        # Probar cargar una muestra
        image, label = dataset[0]
        char = config.index_to_char(label)
        
        print("✓ Muestra de prueba cargada:")
        print(f"  Forma: {image.shape}")
        print(f"  Carácter: '{char}'")
        
        print("\n" + "="*70)
        print(" ✅ DATASET NORMALIZADO LISTO PARA ENTRENAR")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return False