"""
Script de entrenamiento MEJORADO para dataset normalizado.

Este script incluye técnicas anti-sobreajuste:
- Early stopping
- Learning rate scheduling
- Data augmentation
- Menor número de épocas
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from models.config import OCRConfig
from models.cnn import OCRCNN
from models.ocr_model import OCRModelWrapper
from datasets.normalized_dataset import NormalizedDataset
from utils.viz import plot_training_loss


def create_augmented_transforms():
    """
    Crea transformaciones con DATA AUGMENTATION para evitar sobreajuste.
    """
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        # Data Augmentation
        transforms.RandomRotation(degrees=10),  # Rotar ±10 grados
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Trasladar un poco
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def create_test_transforms():
    """
    Transformaciones SIN augmentation para validación/test.
    """
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def train_with_early_stopping(
    dataset_path: str,
    epochs: int = 50,  # Reducido de 90 a 50
    batch_size: int = 128,
    learning_rate: float = 0.001,
    save_dir: str = './models/weights',
    train_split: float = 0.8,
    patience: int = 10  # Parar si no mejora en 10 épocas
):
    """
    Entrena el modelo con dataset normalizado usando early stopping.
    
    Args:
        dataset_path: Ruta al dataset normalizado.
        epochs: Número máximo de épocas.
        batch_size: Tamaño del batch.
        learning_rate: Tasa de aprendizaje.
        save_dir: Directorio donde guardar.
        train_split: Proporción train/test.
        patience: Épocas sin mejora antes de parar.
    """
    print("="*70)
    print(" ENTRENAMIENTO MEJORADO - DATASET NORMALIZADO")
    print("="*70)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Configuración
    print("\n[1/6] Configurando modelo...")
    config = OCRConfig(
        input_shape=(1, 28, 28),
        num_classes=64,
        learning_rate=learning_rate
    )
    print(config)
    
    # Cargar dataset con AUGMENTATION
    print("\n[2/6] Cargando dataset normalizado...")
    
    # Transformaciones diferentes para train y test
    train_transform = create_augmented_transforms()
    test_transform = create_test_transforms()
    
    # Cargar dataset completo primero (sin transform)
    full_dataset = NormalizedDataset(
        root_dir=dataset_path,
        transform=None,  # Lo aplicaremos después
        config=config
    )
    
    # Dividir en train/test
    train_size = int(train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_indices, test_indices = random_split(
        range(len(full_dataset)),
        [train_size, test_size]
    )
    
    # Crear datasets con transformaciones apropiadas
    train_dataset = NormalizedDataset(dataset_path, train_transform, config)
    test_dataset = NormalizedDataset(dataset_path, test_transform, config)
    
    # Aplicar los índices
    train_dataset.image_folder.samples = [train_dataset.image_folder.samples[i] for i in train_indices.indices]
    test_dataset.image_folder.samples = [test_dataset.image_folder.samples[i] for i in test_indices.indices]
    
    print(f"✓ Train: {len(train_dataset.image_folder.samples)} imágenes (con augmentation)")
    print(f"✓ Test: {len(test_dataset.image_folder.samples)} imágenes (sin augmentation)")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Crear modelo
    print("\n[3/6] Creando modelo CNN...")
    model = OCRCNN(config)
    print(f"✓ Modelo en: {model.device}")
    print(f"✓ Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Early Stopping setup
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(save_dir, f"best_normalized_{timestamp}.pth")
    
    print(f"\n[4/6] Iniciando entrenamiento con Early Stopping...")
    print(f"  Épocas máximas: {epochs}")
    print(f"  Patience: {patience} épocas")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Data Augmentation: ✓ Activado")
    print(f"  Modelo se guardará en: {best_model_path}")
    print("-"*70)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Entrenar una época
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(model.device)
            labels = labels.to(model.device)
            
            outputs = model(images)
            loss = model.criterion(outputs, labels)
            
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(avg_train_loss)
        
        # Validar
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(model.device)
                labels = labels.to(model.device)
                
                outputs = model(images)
                loss = model.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(test_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        
        # Imprimir progreso
        print(f"Época [{epoch+1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            
            # Sobrescribir el mejor modelo (siempre el mismo archivo)
            model.save_checkpoint(best_model_path)
            print(f"  ✓ Mejor modelo actualizado (val_loss: {best_val_loss:.4f}, época {epoch+1})")
        else:
            epochs_without_improvement += 1
            print(f"  ⚠️  Sin mejora ({epochs_without_improvement}/{patience})")
            
            if epochs_without_improvement >= patience:
                print(f"\n⏹️  Early Stopping: No mejora en {patience} épocas")
                print(f"  Mejor val_loss: {best_val_loss:.4f}")
                break
    
    # Evaluación final
    print("\n[5/6] Evaluando mejor modelo...")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=model.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Cargado mejor modelo desde: {best_model_path}")
    
    model.evaluate_model(test_loader, verbose=True)
    
    # Guardar gráficas
    print("\n[6/6] Generando visualizaciones...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(save_dir, f"training_normalized_{timestamp}.png")
    plot_training_loss(train_losses, val_losses, save_path=plot_path)
    
    # Wrapper y visualización
    test_transform_final = create_test_transforms()
    wrapper = OCRModelWrapper(model, test_transform_final, config)
    
    viz_path = os.path.join(save_dir, f"predictions_normalized_{timestamp}.png")
    wrapper.visualize_predictions(test_loader, num_samples=9, save_path=viz_path)
    
    print("\n" + "="*70)
    print(" ✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"Mejor modelo: {best_model_path}")
    print(f"Mejor val_loss: {best_val_loss:.4f}")
    print(f"Épocas entrenadas: {epoch+1}/{epochs}")
    print(f"Gráfica: {plot_path}")
    print("="*70 + "\n")
    
    return model, wrapper


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar modelo con dataset normalizado (anti-sobreajuste)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Ruta al directorio del dataset normalizado'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Número máximo de épocas (default: 50)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Épocas sin mejora antes de parar (default: 10)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Tamaño del batch (default: 128)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Tasa de aprendizaje (default: 0.001)'
    )
    
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./models/weights',
        help='Directorio donde guardar (default: ./models/weights)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"❌ ERROR: No se encontró el dataset: {args.dataset}")
        sys.exit(1)
    
    train_with_early_stopping(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        patience=args.patience
    )


if __name__ == '__main__':
    main()