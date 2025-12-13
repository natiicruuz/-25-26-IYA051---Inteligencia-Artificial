"""
Script para predecir una palabra completa desde una imagen.

Este script segmenta una imagen de palabra en letras individuales
y predice cada letra para reconstruir la palabra.
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
from segmentation.letter_segmentation import LetterSegmenter
import cv2


def create_transforms():
    """Crea las transformaciones para preprocesar imágenes."""
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def predict_word(
    image_path: str,
    model_path: str,
    show_visualization: bool = True
) -> str:
    """
    Predice la palabra en una imagen.
    
    Args:
        image_path: Ruta a la imagen de la palabra.
        model_path: Ruta al modelo entrenado (.pth).
        show_visualization: Si True, muestra la segmentación y predicciones.
    
    Returns:
        Palabra predicha (string).
    """
    print(f"\n{'='*60}")
    print(f"PREDICCIÓN DE PALABRA")
    print(f"{'='*60}")
    
    # Verificar que existen los archivos
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
    
    print(f"\nImagen: {image_path}")
    print(f"Modelo: {model_path}")
    
    # Cargar modelo
    print("\nCargando modelo...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    model = OCRCNN(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Modelo cargado correctamente")
    
    # Crear wrapper y segmentador
    transform = create_transforms()
    wrapper = OCRModelWrapper(model, transform, config)
    segmenter = LetterSegmenter()
    
    # Cargar imagen
    print("\nSegmentando palabra en letras...")
    word_image = cv2.imread(image_path)
    
    if word_image is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    
    # Segmentar letras
    letter_images = segmenter.segment_word(word_image)
    
    print(f"Letras detectadas: {len(letter_images)}")
    
    if len(letter_images) == 0:
        print("⚠️  No se detectaron letras en la imagen")
        return ""
    
    # Predecir letras
    print("\nRealizando predicciones...")
    predictions = wrapper.predict_letter_sequence(
        letter_images,
        show_visualization=show_visualization,
        word_index=0
    )
    
    # Reconstruir palabra
    predicted_word = ''.join(predictions)
    
    print(f"\n{'='*60}")
    print(f"RESULTADO: '{predicted_word}'")
    print(f"{'='*60}\n")
    
    return predicted_word


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description='Predecir una palabra desde una imagen'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Ruta a la imagen de la palabra'
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
        help='No mostrar visualización (solo imprimir resultado)'
    )
    
    args = parser.parse_args()
    
    # Predecir
    predicted = predict_word(
        image_path=args.image,
        model_path=args.model,
        show_visualization=not args.no_show
    )
    
    return predicted


if __name__ == '__main__':
    main()