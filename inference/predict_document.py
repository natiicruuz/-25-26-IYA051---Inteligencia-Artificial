"""
Script para predecir un documento completo (múltiples palabras).

Este script procesa una imagen de documento, detecta palabras, segmenta letras
y reconstruye el texto completo del documento.
"""

import os
import sys
import argparse
from typing import List

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import transforms
from models.config import OCRConfig
from models.cnn import OCRCNN
from models.ocr_model import OCRModelWrapper
from segmentation.word_segmentation import WordSegmenter
from segmentation.letter_segmentation import LetterSegmenter


def create_transforms():
    """Crea las transformaciones para preprocesar imágenes."""
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def predict_document(
    image_path: str,
    model_path: str,
    output_path: str = None,
    show_visualization: bool = False
) -> str:
    """
    Predice el texto completo de un documento.
    
    Args:
        image_path: Ruta a la imagen del documento.
        model_path: Ruta al modelo entrenado (.pth).
        output_path: Ruta opcional donde guardar el texto predicho.
        show_visualization: Si True, muestra visualizaciones de la segmentación.
    
    Returns:
        Texto completo predicho.
    """
    print(f"\n{'='*70}")
    print(f"PREDICCIÓN DE DOCUMENTO COMPLETO")
    print(f"{'='*70}")
    
    # Verificar archivos
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
    
    print(f"\nImagen: {image_path}")
    print(f"Modelo: {model_path}")
    
    # Cargar modelo
    print("\n[1/4] Cargando modelo...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    model = OCRCNN(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Modelo cargado correctamente en {device}")
    
    # Crear componentes
    transform = create_transforms()
    wrapper = OCRModelWrapper(model, transform, config)
    word_segmenter = WordSegmenter()
    letter_segmenter = LetterSegmenter()
    
    # Segmentar documento en palabras
    print("\n[2/4] Segmentando documento en palabras...")
    word_images = word_segmenter.detect_words(image_path)
    
    print(f"✓ Palabras detectadas: {len(word_images)}")
    
    if len(word_images) == 0:
        print("⚠️  No se detectaron palabras en el documento")
        return ""
    
    # Segmentar palabras en letras
    print("\n[3/4] Segmentando palabras en letras...")
    letters_by_word = letter_segmenter.segment_multiple_words(word_images)
    
    total_letters = sum(len(letters) for letters in letters_by_word)
    print(f"✓ Total de letras detectadas: {total_letters}")
    
    # Predecir cada palabra
    print("\n[4/4] Realizando predicciones...")
    predicted_words = []
    
    for word_idx, letter_images in enumerate(letters_by_word):
        if len(letter_images) == 0:
            continue
        
        # Predecir letras de la palabra
        predictions = wrapper.predict_letter_sequence(
            letter_images,
            show_visualization=show_visualization,
            word_index=word_idx
        )
        
        word = ''.join(predictions)
        predicted_words.append(word)
        
        print(f"  Palabra {word_idx + 1}/{len(letters_by_word)}: '{word}'")
    
    # Reconstruir texto completo
    full_text = ' '.join(predicted_words)
    
    # Guardar resultado si se especifica
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"\n✓ Texto guardado en: {output_path}")
    
    # Mostrar resultado
    print(f"\n{'='*70}")
    print(f"RESULTADO FINAL:")
    print(f"{'='*70}")
    print(f"\n{full_text}\n")
    print(f"{'='*70}")
    print(f"Palabras detectadas: {len(predicted_words)}")
    print(f"Caracteres totales: {len(full_text.replace(' ', ''))}")
    print(f"{'='*70}\n")
    
    return full_text


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description='Predecir texto completo de un documento'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Ruta a la imagen del documento'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Ruta al modelo entrenado (.pth)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Ruta opcional donde guardar el texto predicho (.txt)'
    )
    
    parser.add_argument(
        '--show-viz',
        action='store_true',
        help='Mostrar visualizaciones de segmentación para cada palabra'
    )
    
    args = parser.parse_args()
    
    # Si no se especifica output, usar el mismo nombre que la imagen pero con .txt
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        args.output = f"{base_name}_predicted.txt"
    
    # Predecir documento
    predicted = predict_document(
        image_path=args.image,
        model_path=args.model,
        output_path=args.output,
        show_visualization=args.show_viz
    )
    
    return predicted


if __name__ == '__main__':
    main()