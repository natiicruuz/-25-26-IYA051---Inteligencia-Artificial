"""
Script de verificación del dataset.

Verifica que tu dataset esté correctamente organizado antes de entrenar.
"""

import os
import sys
import argparse

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.handwritten_dataset import HandwrittenDataset
from models.config import OCRConfig
from torchvision import transforms


def verify_dataset(dataset_path: str):
    """
    Verifica la estructura y contenido del dataset.
    
    Args:
        dataset_path: Ruta al directorio raíz del dataset.
    """
    print("\n" + "="*70)
    print(" VERIFICACIÓN DE DATASET")
    print("="*70 + "\n")
    
    print(f"Ruta del dataset: {dataset_path}")
    
    # Verificar que existe el directorio
    if not os.path.exists(dataset_path):
        print(f"\n❌ ERROR: No se encontró el directorio: {dataset_path}")
        sys.exit(1)
    
    print("✓ Directorio encontrado")
    
    # Verificar subdirectorios
    print("\n" + "-"*70)
    print("Verificando subdirectorios...")
    print("-"*70)
    
    subdirs_found = []
    
    # Buscar variantes de nombres
    for name_variant in [('numeros', 'números'), ('mayusculas', 'mayúsculas'), ('minusculas', 'minúsculas')]:
        for variant in name_variant:
            subdir_path = os.path.join(dataset_path, variant)
            if os.path.exists(subdir_path):
                subdirs_found.append((name_variant[0], variant, subdir_path))
                print(f"✓ Encontrado: '{variant}/'")
                break
        else:
            print(f"⚠️  No encontrado: {name_variant[0]}/ (o variantes)")
    
    if len(subdirs_found) == 0:
        print("\n❌ ERROR: No se encontró ningún subdirectorio válido")
        print("\nEstructura esperada:")
        print("  dataset/")
        print("  ├── numeros/ (o números/)")
        print("  ├── mayusculas/ (o mayúsculas/)")
        print("  └── minusculas/ (o minúsculas/)")
        sys.exit(1)
    
    # Contar imágenes en cada subdirectorio
    print("\n" + "-"*70)
    print("Contando imágenes...")
    print("-"*70)
    
    total_images = 0
    total_classes = 0
    
    for category, variant_name, subdir_path in subdirs_found:
        char_folders = [f for f in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, f))]
        
        images_in_category = 0
        
        for char_folder in char_folders:
            char_path = os.path.join(subdir_path, char_folder)
            images = [f for f in os.listdir(char_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            images_in_category += len(images)
            total_classes += 1
        
        total_images += images_in_category
        
        print(f"  {variant_name}/")
        print(f"    Clases: {len(char_folders)}")
        print(f"    Imágenes: {images_in_category}")
    
    print("\n" + "-"*70)
    print(f"TOTAL: {total_images} imágenes en {total_classes} clases")
    print("-"*70)
    
    # Intentar cargar el dataset
    print("\n" + "-"*70)
    print("Probando carga del dataset...")
    print("-"*70)
    
    try:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        config = OCRConfig(num_classes=64)
        dataset = HandwrittenDataset(
            root_dir=dataset_path,
            transform=transform,
            config=config
        )
        
        print(f"✓ Dataset cargado correctamente")
        print(f"✓ Total de muestras: {len(dataset)}")
        
        # Mostrar estadísticas
        dataset.print_statistics()
        
        # Intentar cargar una muestra
        print("\n" + "-"*70)
        print("Probando carga de una muestra...")
        print("-"*70)
        
        image, label = dataset[0]
        char = config.index_to_char(label)
        
        print(f"✓ Muestra cargada correctamente")
        print(f"  Forma del tensor: {image.shape}")
        print(f"  Etiqueta (índice): {label}")
        print(f"  Carácter: '{char}'")
        
        print("\n" + "="*70)
        print(" ✅ VERIFICACIÓN EXITOSA - DATASET LISTO PARA ENTRENAR")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR al cargar el dataset:")
        print(f"   {str(e)}")
        print("\nVerifica que la estructura sea correcta.")
        return False


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description='Verificar estructura y contenido del dataset'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Ruta al directorio raíz del dataset'
    )
    
    args = parser.parse_args()
    
    success = verify_dataset(args.dataset)
    
    if success:
        print("Puedes proceder a entrenar el modelo con:")
        print(f'\npython training\\train_handwritten.py --dataset "{args.dataset}" --epochs 90\n')
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()