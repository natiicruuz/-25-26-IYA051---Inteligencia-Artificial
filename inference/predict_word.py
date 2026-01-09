"""
Script para predecir palabras completas desde una imagen.

Versión estable sin recursión.
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
import numpy as np
from torchvision import transforms
from models.config import OCRConfig
from models.cnn import OCRCNN
from models.ocr_model import OCRModelWrapper
from segmentation.letter_segmentation import LetterSegmenter


def create_transforms():
    """Transformaciones para preprocesar imágenes."""
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])


def preprocess_word_image(image_path: str) -> np.ndarray:
    """
    Preprocesa imagen de palabra.
    
    Args:
        image_path: Ruta a la imagen.
    
    Returns:
        Imagen preprocesada.
    """
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"No se pudo cargar: {image_path}")
    
    # Escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binarizar
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Limpiar ruido pequeño
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Convertir a BGR
    processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    return processed


def predict_word(
    image_path: str,
    model_path: str,
    show_visualization: bool = True,
    debug_segmentation: bool = False,
    valley_threshold: float = 0.10
) -> str:
    """
    Predice la palabra en una imagen.
    
    Args:
        image_path: Ruta a la imagen de la palabra.
        model_path: Ruta al modelo entrenado (.pth).
        show_visualization: Si True, muestra predicciones.
        debug_segmentation: Si True, muestra proceso de segmentación.
        valley_threshold: Umbral para detectar valles (0-1, menor = más sensible).
    
    Returns:
        Palabra predicha (string).
    """
    print(f"\n{'='*70}")
    print(f" PREDICCIÓN DE PALABRA COMPLETA")
    print(f"{'='*70}")
    
    # Verificar archivos
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ No se encontró: {image_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ No se encontró: {model_path}")
    
    print(f"\n📁 Archivos:")
    print(f"  Imagen: {image_path}")
    print(f"  Modelo: {model_path}")
    
    # Cargar modelo
    print("\n[1/4] Cargando modelo...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        model = OCRCNN(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"  ✓ Modelo cargado")
        print(f"  → Dispositivo: {device}")
        print(f"  → Clases: {config.num_classes}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        sys.exit(1)
    
    # Crear wrapper y segmentador
    transform = create_transforms()
    wrapper = OCRModelWrapper(model, transform, config)
    
    segmenter = LetterSegmenter(
        min_width=10,
        min_height=15,
        margin=2,
        valley_threshold=valley_threshold,
        min_valley_width=2,
        max_aspect_ratio=3.0, 
        split_wide_threshold=2.5 
    )
    
    # Preprocesar imagen
    print("\n[2/4] Preprocesando imagen...")
    word_image = preprocess_word_image(image_path)
    h, w = word_image.shape[:2]
    print(f"  ✓ Procesada: {w}x{h} px")
    
    # Segmentar letras
    print(f"\n[3/4] Segmentando (valley_threshold={valley_threshold:.2f})...")
    try:
        letter_images = segmenter.segment_word(word_image, debug=debug_segmentation)
        
        print(f"  ✓ Letras detectadas: {len(letter_images)}")
        
        if len(letter_images) == 0:
            print("\n  ⚠️  No se detectaron letras")
            print(f"  Prueba ajustando el umbral:")
            print(f"    python inference/predict_word.py --image {image_path} --model {model_path} --threshold 0.05")
            return ""
        
        for i, letter in enumerate(letter_images):
            h, w = letter.shape[:2]
            print(f"    Letra {i+1}: {w}x{h} px")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Predecir letras
    print("\n[4/4] Realizando predicciones...")
    try:
        predictions = wrapper.predict_letter_sequence(
            letter_images,
            show_visualization=show_visualization,
            word_index=0
        )
        
        print(f"  ✓ Predicciones:")
        for i, char in enumerate(predictions):
            print(f"    Letra {i+1}: '{char}'")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        sys.exit(1)
    
    # Resultado
    predicted_word = ''.join(predictions)
    corrected_word = fix_spanish_words(predicted_word)

    print(f"Predicción: {predicted_word}")
    if corrected_word != predicted_word:
        print(f"Corregido: {corrected_word}")
        
    print(f"\n{'='*70}")
    print(f"✅ RESULTADO: '{predicted_word}'")
    print(f"{'='*70}\n")
    
    return predicted_word
    
def fix_spanish_words(word):
    """
    Post-procesamiento simple para Ñ/ñ.
    
    Corrige patrones comunes donde N → Ñ.
    """
    # Diccionario de palabras comunes
    corrections = {
        'espana': 'españa',
        'ESPaRa': 'España',
        'Espana': 'España',
        'ESPArA': 'ESPAÑA',
        'marana': 'mañana',
        'Marana': 'Mañana',
        'niro': 'niño',
        'Niro': 'Niño',
        'aro': 'año',
        'Aro': 'Año',
        'seror': 'señor',
        'Seror': 'Señor',
    }
    
    # Buscar coincidencia exacta
    if word in corrections:
        return corrections[word]
    
    # Patrón común: 'n' + vocal después → podría ser 'ñ' 
    return word




def main():
    parser = argparse.ArgumentParser(
        description='Predecir palabras completas',
        epilog="""
    Ejemplos:

    # Básico
    python inference/predict_word.py --image word.png --model modelo.pth
    
    # Con debugging
    python inference/predict_word.py --image word.png --model modelo.pth --debug
    
    # Ajustar sensibilidad (menor = más sensible)
    python inference/predict_word.py --image word.png --model modelo.pth --threshold 0.05
            """
    )
    
    parser.add_argument('--image', type=str, required=True, help='Ruta a la imagen')
    parser.add_argument('--model', type=str, required=True, help='Ruta al modelo (.pth)')
    parser.add_argument('--no-show', action='store_true', help='No mostrar visualización')
    parser.add_argument('--debug', action='store_true', help='Mostrar proceso de segmentación')
    parser.add_argument('--threshold', type=float, default=0.10, 
                        help='Umbral de valle (default: 0.10, menor = más sensible)')
    
    args = parser.parse_args()
    
    predict_word(
        image_path=args.image,
        model_path=args.model,
        show_visualization=not args.no_show,
        debug_segmentation=args.debug,
        valley_threshold=args.threshold
    )


if __name__ == '__main__':
    main()