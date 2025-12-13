"""
Script de entrenamiento para el modelo con dataset manuscrito custom.

Este script entrena un modelo CNN usando solo el dataset de caracteres
manuscritos custom (mayúsculas, minúsculas y números).
"""

import os
import sys
import argparse
from datetime import datetime

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import OCRConfig
from models.cnn import OCRCNN
from models.ocr_model import OCRModelWrapper
from datasets.combined_dataset import create_handwritten_only_dataloaders
from utils.viz import plot_training_loss
from torchvision import transforms


def create_transforms():
    """
    Crea las transformaciones para el dataset manuscrito custom.
    
    Returns:
        Transformaciones de PyTorch.
    """
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def train_handwritten_model(
    dataset_path: str,
    epochs: int = 90,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    save_dir: str = './models/weights',
    train_split: float = 0.8
):
    """
    Entrena el modelo con el dataset manuscrito custom.
    
    Args:
        dataset_path: Ruta al directorio raíz del dataset custom.
        epochs: Número de épocas de entrenamiento.
        batch_size: Tamaño del batch.
        learning_rate: Tasa de aprendizaje.
        save_dir: Directorio donde guardar el modelo entrenado.
        train_split: Proporción train/test.
    """
    print("="*70)
    print(" ENTRENAMIENTO DE MODELO OCR - DATASET MANUSCRITO CUSTOM")
    print("="*70)
    
    # Crear directorio de guardado si no existe
    os.makedirs(save_dir, exist_ok=True)
    
    # Configuración del modelo (64 clases: incluye Ñ y ñ)
    print("\n[1/5] Configurando modelo...")
    config = OCRConfig(
        input_shape=(1, 28, 28),
        num_classes=64,
        learning_rate=learning_rate
    )
    print(config)
    
    # Cargar dataset
    print("\n[2/5] Cargando dataset manuscrito custom...")
    train_loader, test_loader = create_handwritten_only_dataloaders(
        custom_root=dataset_path,
        config=config,
        batch_size=batch_size,
        train_split=train_split
    )
    
    # Crear modelo
    print("\n[3/5] Creando modelo CNN...")
    model = OCRCNN(config)
    print(f"Modelo creado en dispositivo: {model.device}")
    print(f"Número de parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Entrenar
    print(f"\n[4/5] Iniciando entrenamiento por {epochs} épocas...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("-"*70)
    
    train_losses, val_losses = model.train_model(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        verbose=True
    )
    
    # Evaluar
    print("\n[5/5] Evaluando modelo final...")
    model.evaluate_model(test_loader, verbose=True)
    
    # Guardar modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"handwritten_custom_ep{epochs}_{timestamp}.pth"
    model_path = os.path.join(save_dir, model_filename)
    model.save_checkpoint(model_path)
    
    # Visualizar pérdidas
    print("\nGenerando gráfica de pérdidas...")
    plot_path = os.path.join(save_dir, f"training_loss_{timestamp}.png")
    plot_training_loss(train_losses, val_losses, save_path=plot_path)
    
    # Crear wrapper y visualizar predicciones
    print("\nCreando visualización de predicciones...")
    transform = create_transforms()
    wrapper = OCRModelWrapper(model, transform, config)
    
    viz_path = os.path.join(save_dir, f"predictions_{timestamp}.png")
    wrapper.visualize_predictions(test_loader, num_samples=9, save_path=viz_path)
    
    print("\n" + "="*70)
    print(" ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"Modelo guardado en: {model_path}")
    print(f"Gráfica guardada en: {plot_path}")
    print(f"Predicciones guardadas en: {viz_path}")
    print("="*70 + "\n")
    
    return model, wrapper


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description='Entrenar modelo OCR con dataset manuscrito custom'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Ruta al directorio raíz del dataset (contiene mayúsculas/, minúsculas/, números/)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=90,
        help='Número de épocas de entrenamiento (default: 90)'
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
        help='Directorio donde guardar el modelo (default: ./models/weights)'
    )
    
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Proporción de datos para entrenamiento (default: 0.8)'
    )
    
    args = parser.parse_args()
    
    # Verificar que existe el dataset
    if not os.path.exists(args.dataset):
        print(f"Error: No se encontró el directorio del dataset: {args.dataset}")
        sys.exit(1)
    
    # Entrenar modelo
    train_handwritten_model(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        train_split=args.train_split
    )


if __name__ == '__main__':
    main()