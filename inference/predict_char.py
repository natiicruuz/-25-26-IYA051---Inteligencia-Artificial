"""
Script para predecir un solo carácter desde una imagen.

Este script carga un modelo entrenado y predice el carácter contenido
en una imagen de un solo carácter.
"""

import os
import sys
import argparse

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import transforms
from models.config import OCRConfig
from models.cnn import OCRCNN
from models.ocr_model import OCRModelWrapper


def create_transforms():
    """Crea las transformaciones para preprocesar imágenes."""
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def predict_character(
    image_path: str,
    model_path: str,
    show_image: bool = True
) -> str:
    """
    Predice el carácter en una imagen.
    
    Args:
        image_path: Ruta a la imagen del carácter.
        model_path: Ruta al modelo entrenado (.pth).
        show_image: Si True, muestra la imagen con la predicción.
    
    Returns:
        Carácter predicho.
    """
    print(f"\n{'='*60}")
    print(f"PREDICCIÓN DE CARÁCTER")
    print(f"{'='*60}")
    
    # Verificar que existen los archivos
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
    
    print(f"\nImagen: {image_path}")
    print(f"Modelo: {model_path}")
    
    # Cargar checkpoint
    print("\nCargando modelo...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Crear configuración y modelo
    config = checkpoint['config']
    model = OCRCNN(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Modelo cargado correctamente")
    print(f"Dispositivo: {device}")
    print(f"Número de clases: {config.num_classes}")
    
    # Crear wrapper
    transform = create_transforms()
    wrapper = OCRModelWrapper(model, transform, config)
    
    # Predecir
    print("\nRealizando predicción...")
    predicted_char = wrapper.predict_single_char(image_path, show_image=show_image)
    
    print(f"\n{'='*60}")
    print(f"RESULTADO: '{predicted_char}'")
    print(f"{'='*60}\n")
    
    return predicted_char


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description='Predecir un carácter desde una imagen'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Ruta a la imagen del carácter'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Ruta al modelo entrenado (.pth)'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='No mostrar la imagen (solo imprimir resultado)'
    )
    
    args = parser.parse_args()
    
    # Predecir
    predicted = predict_character(
        image_path=args.image,
        model_path=args.model,
        show_image=not args.no_show
    )
    
    return predicted


if __name__ == '__main__':
    main()