"""
Script de entrenamiento: Custom + EMNIST + Early Stopping + Data Augmentation.
"""

import os
import sys
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms

from models.config import OCRConfig
from models.cnn import OCRCNN
from models.ocr_model import OCRModelWrapper
from datasets.normalized_dataset import NormalizedDataset
from datasets.augmented_dataset import AugmentedDatasetWrapper, get_augmentation_transform
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
    custom_weight: float = 2.0,
    early_stopping: bool = True,
    patience: int = 5,
    use_augmentation: bool = True
):
    """
    Entrena modelo con custom + EMNIST + Early Stopping + Data Augmentation.
    
    Args:
        custom_path: Ruta al dataset normalizado custom
        emnist_root: Ruta para EMNIST
        epochs: Número máximo de épocas
        batch_size: Tamaño del batch
        learning_rate: Tasa de aprendizaje
        save_dir: Directorio para guardar
        custom_weight: Peso para dataset custom (2.0 = cuenta doble)
        early_stopping: Si True, activa early stopping
        patience: Épocas a esperar sin mejora antes de parar
        use_augmentation: Si True, aplica data augmentation
    """
    print("="*70)
    if use_augmentation:
        print(" ENTRENAMIENTO: CUSTOM + EMNIST + EARLY STOPPING + AUGMENTATION")
    else:
        print(" ENTRENAMIENTO: CUSTOM + EMNIST + EARLY STOPPING")
    print("="*70)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Configuración
    print("\n[1/6] Configurando...")
    config = OCRConfig(
        input_shape=(1, 28, 28),
        num_classes=62,
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
    
    # Dividir custom en train/test
    from torch.utils.data import random_split
    custom_train_size = int(0.8 * len(custom_dataset))
    custom_test_size = len(custom_dataset) - custom_train_size
    
    custom_train, custom_test = random_split(
        custom_dataset,
        [custom_train_size, custom_test_size]
    )
    
    # ⭐ APLICAR AUGMENTATION SOLO A TRAIN
    if use_augmentation:
        print("\n✅ Data Augmentation activado para entrenamiento")
        augmentation = get_augmentation_transform()
        custom_train = AugmentedDatasetWrapper(custom_train, augmentation)
        print("  → Rotación: ±10°")
        print("  → Escalado: 0.9x - 1.1x")
        print("  → Traslación: ±10%")
        print("  → Perspectiva: 20% distorsión (50% prob)")
        print("  → Brillo/contraste: ±20%/±15%")
    
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
    
    # ⭐ APLICAR AUGMENTATION A EMNIST TRAIN
    if use_augmentation:
        emnist_train = AugmentedDatasetWrapper(emnist_train, augmentation)
    
    print(f"\n✓ Custom: {len(custom_dataset)} imágenes")
    print(f"  → Train: {len(custom_train)} (augmentation: {use_augmentation})")
    print(f"  → Test: {len(custom_test)} (sin augmentation)")
    print(f"✓ EMNIST train: {len(emnist_train)} (augmentation: {use_augmentation})")
    print(f"✓ EMNIST test: {len(emnist_test)} (sin augmentation)")
    
    # Combinar datasets
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
    print(f"\n[4/6] Entrenando (máximo {epochs} épocas)...")
    print("-"*70)
    
    train_losses, val_losses, early_stop_info = model.train_model(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        verbose=True,
        early_stopping=early_stopping,
        patience=patience,
        checkpoint_dir='./checkpoints'
    )
    
    # Resumen early stopping
    if early_stop_info:
        print(f"\n{'='*70}")
        print(f"📊 RESUMEN DE EARLY STOPPING")
        print(f"{'='*70}")
        
        if early_stop_info.get('stopped'):
            print(f"✅ Entrenamiento detenido automáticamente")
            print(f"  → Mejor val_loss: {early_stop_info['best_loss']:.4f}")
            print(f"  → Mejor época: {early_stop_info['best_epoch']}")
        else:
            print(f"⚠️  Completó todas las épocas sin early stopping")
            print(f"  → Val_loss final: {val_losses[-1]:.4f}")
        
        print(f"{'='*70}\n")
    
    # Evaluar
    print("\n[5/6] Evaluando modelo final...")
    model.evaluate_model(test_loader, verbose=True)
    
    # Guardar
    print("\n[6/6] Guardando resultados...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aug_suffix = "_augmented" if use_augmentation else ""
    model_path = os.path.join(save_dir, f"final_custom_emnist{aug_suffix}_{timestamp}.pth")
    model.save_checkpoint(model_path)
    
    # Visualizaciones
    plot_path = os.path.join(save_dir, f"training_final{aug_suffix}_{timestamp}.png")
    plot_training_loss(train_losses, val_losses, save_path=plot_path)
    
    wrapper = OCRModelWrapper(model, custom_transform, config)
    viz_path = os.path.join(save_dir, f"predictions_final{aug_suffix}_{timestamp}.png")
    wrapper.visualize_predictions(test_loader, num_samples=9, save_path=viz_path)
    
    # Resumen final
    print("\n" + "="*70)
    print(" ✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"📁 Modelo guardado: {model_path}")
    print(f"📊 Gráfica: {plot_path}")
    print(f"🖼️  Visualización: {viz_path}")
    
    if early_stop_info and early_stop_info.get('stopped'):
        print(f"\n🎯 Early Stopping:")
        print(f"  → Mejor época: {early_stop_info['best_epoch']}/{epochs}")
        print(f"  → Val loss óptimo: {early_stop_info['best_loss']:.4f}")
    
    if use_augmentation:
        print(f"\n🔄 Data Augmentation:")
        print(f"  → Aplicado solo en entrenamiento")
        print(f"  → Transformaciones: Rotación, Escalado, Traslación, Perspectiva, Brillo")
    
    print("="*70 + "\n")
    
    return model_path


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar con custom + EMNIST + Early Stopping + Augmentation'
    )
    
    parser.add_argument('--custom', type=str, default='./data/normalized')
    parser.add_argument('--emnist', type=str, default='./data/emnist')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--no-early-stopping', action='store_true')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Desactivar data augmentation')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.custom):
        print(f"❌ ERROR: No existe {args.custom}")
        sys.exit(1)
    
    train_custom_emnist(
        custom_path=args.custom,
        emnist_root=args.emnist,
        epochs=args.epochs,
        early_stopping=not args.no_early_stopping,
        patience=args.patience,
        use_augmentation=not args.no_augmentation
    )


if __name__ == '__main__':
    main()