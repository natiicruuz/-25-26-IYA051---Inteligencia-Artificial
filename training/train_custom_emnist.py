"""
Script de entrenamiento: Custom normalizado + EMNIST.

Este script combina tu dataset normalizado con EMNIST para obtener:
- Tu dataset: datos reales manuscritos
- EMNIST: volumen de datos para generalización

Precisión esperada: 85-92%
"""

import os
import sys
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from models.config import OCRConfig
from models.cnn import OCRCNN
from models.ocr_model import OCRModelWrapper
from datasets.normalized_dataset import NormalizedDataset
from utils.viz import plot_training_loss


def create_normalized_transform():
    """Transformaciones para dataset normalizado (SIN Normalize extra)."""
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])


def create_emnist_transform():
    """Transformaciones para EMNIST (con correcciones de orientación)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.rot90(x, k=-1, dims=[1, 2])),
        transforms.Lambda(lambda x: 1 - x),
        transforms.Lambda(lambda x: x.flip(-1)),
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
    ])


def train_custom_emnist(
    custom_path: str,
    emnist_root: str = './data/emnist',
    epochs: int = 15,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    save_dir: str = './models/weights',
    custom_weight: float = 2.0
):
    """
    Entrena modelo con custom + EMNIST.
    
    Args:
        custom_path: Ruta al dataset normalizado custom
        emnist_root: Ruta para EMNIST
        epochs: Número de épocas
        batch_size: Tamaño del batch
        learning_rate: Tasa de aprendizaje
        save_dir: Directorio para guardar
        custom_weight: Peso para dataset custom (2.0 = cuenta doble)
    """
    print("="*70)
    print(" ENTRENAMIENTO FINAL: CUSTOM + EMNIST")
    print("="*70)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Configuración (62 clases - compatible con EMNIST)
    print("\n[1/6] Configurando...")
    config = OCRConfig(
        input_shape=(1, 28, 28),
        num_classes=62,  # EMNIST tiene 62 clases
        learning_rate=learning_rate
    )
    print(config)
    
    # Cargar datasets
    print("\n[2/6] Cargando datasets...")
    
    # Dataset custom normalizado
    custom_transform = create_normalized_transform()
    custom_dataset = NormalizedDataset(
        root_dir=custom_path,
        transform=custom_transform,
        config=config
    )
    
    # EMNIST
    print("\nDescargando/cargando EMNIST...")
    from torchvision import datasets as torch_datasets
    
    emnist_transform = create_emnist_transform()
    
    emnist_train = torch_datasets.EMNIST(
        root=emnist_root,
        split='byclass',
        train=True,
        download=True,
        transform=emnist_transform
    )
    
    emnist_test = torch_datasets.EMNIST(
        root=emnist_root,
        split='byclass',
        train=False,
        download=True,
        transform=emnist_transform
    )
    
    print(f"\n✓ Custom: {len(custom_dataset)} imágenes")
    print(f"✓ EMNIST train: {len(emnist_train)} imágenes")
    print(f"✓ EMNIST test: {len(emnist_test)} imágenes")
    
    # Dividir custom en train/test
    from torch.utils.data import random_split
    custom_train_size = int(0.8 * len(custom_dataset))
    custom_test_size = len(custom_dataset) - custom_train_size
    
    custom_train, custom_test = random_split(
        custom_dataset,
        [custom_train_size, custom_test_size]
    )
    
    # Combinar datasets
    # Repetir custom para darle más peso
    train_datasets = [emnist_train]
    test_datasets = [emnist_test]
    
    for _ in range(int(custom_weight)):
        train_datasets.append(custom_train)
        test_datasets.append(custom_test)
    
    combined_train = ConcatDataset(train_datasets)
    combined_test = ConcatDataset(test_datasets)
    
    print(f"\n✓ Train combinado: {len(combined_train)} imágenes")
    print(f"✓ Test combinado: {len(combined_test)} imágenes")
    
    # DataLoaders
    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(combined_test, batch_size=batch_size, shuffle=False)
    
    # Crear modelo
    print("\n[3/6] Creando modelo...")
    model = OCRCNN(config)
    print(f"✓ Dispositivo: {model.device}")
    print(f"✓ Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Entrenar
    print(f"\n[4/6] Entrenando {epochs} épocas...")
    print("-"*70)
    
    train_losses, val_losses = model.train_model(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        verbose=True
    )
    
    # Evaluar
    print("\n[5/6] Evaluando...")
    model.evaluate_model(test_loader, verbose=True)
    
    # Guardar
    print("\n[6/6] Guardando...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"final_custom_emnist_{timestamp}.pth")
    model.save_checkpoint(model_path)
    
    # Visualizaciones
    plot_path = os.path.join(save_dir, f"training_final_{timestamp}.png")
    plot_training_loss(train_losses, val_losses, save_path=plot_path)
    
    wrapper = OCRModelWrapper(model, custom_transform, config)
    viz_path = os.path.join(save_dir, f"predictions_final_{timestamp}.png")
    wrapper.visualize_predictions(test_loader, num_samples=9, save_path=viz_path)
    
    print("\n" + "="*70)
    print(" ✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"Modelo: {model_path}")
    print(f"Gráfica: {plot_path}")
    print("="*70 + "\n")
    
    return model_path


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar con custom + EMNIST'
    )
    
    parser.add_argument(
        '--custom',
        type=str,
        default='./data/normalized',
        help='Ruta al dataset custom normalizado'
    )
    
    parser.add_argument(
        '--emnist',
        type=str,
        default='./data/emnist',
        help='Ruta para EMNIST'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='Número de épocas (default: 15)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.custom):
        print(f"❌ ERROR: No existe {args.custom}")
        print("Ejecuta primero: python normalize_dataset.py")
        sys.exit(1)
    
    train_custom_emnist(
        custom_path=args.custom,
        emnist_root=args.emnist,
        epochs=args.epochs
    )


if __name__ == '__main__':
    main()