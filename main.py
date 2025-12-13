"""
Script principal del sistema OCR.

Este script proporciona una interfaz de línea de comandos para usar todas
las funcionalidades del sistema OCR: predicción, entrenamiento y extracción.
"""

import os
import sys
import argparse
from datetime import datetime

# Añadir el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torchvision import transforms

from models.config import OCRConfig
from models.cnn import OCRCNN
from models.ocr_model import OCRModelWrapper
from segmentation.word_segmentation import WordSegmenter
from segmentation.letter_segmentation import LetterSegmenter
from extras.detect_tables import TableDetector
from extras.detect_figures import FigureDetector


def create_transforms():
    """Crea las transformaciones estándar para el modelo."""
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def load_model(model_path: str):
    """
    Carga un modelo entrenado.
    
    Args:
        model_path: Ruta al checkpoint del modelo.
    
    Returns:
        Tupla (modelo, configuración, wrapper).
    """
    print(f"Cargando modelo desde: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    model = OCRCNN(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = create_transforms()
    wrapper = OCRModelWrapper(model, transform, config)
    
    print(f"✓ Modelo cargado correctamente")
    print(f"  Dispositivo: {device}")
    print(f"  Clases: {config.num_classes}")
    
    return model, config, wrapper


def ocr_document_full(
    image_path: str,
    model_path: str,
    output_dir: str = './ocr_output',
    extract_tables: bool = False,
    extract_figures: bool = False
):
    """
    Pipeline completo de OCR para un documento.
    
    Args:
        image_path: Ruta al documento.
        model_path: Ruta al modelo entrenado.
        output_dir: Directorio de salida.
        extract_tables: Si True, extrae tablas.
        extract_figures: Si True, extrae figuras.
    """
    print(f"\n{'='*70}")
    print(f" SISTEMA OCR - PROCESAMIENTO COMPLETO DE DOCUMENTO")
    print(f"{'='*70}\n")
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar modelo
    _, config, wrapper = load_model(model_path)
    
    # Segmentadores
    word_segmenter = WordSegmenter()
    letter_segmenter = LetterSegmenter()
    
    # 1. OCR - Extraer texto
    print("\n[1/3] Extrayendo texto del documento...")
    
    word_images = word_segmenter.detect_words(image_path)
    print(f"  Palabras detectadas: {len(word_images)}")
    
    letters_by_word = letter_segmenter.segment_multiple_words(word_images)
    
    predicted_words = []
    for word_idx, letter_images in enumerate(letters_by_word):
        if len(letter_images) == 0:
            continue
        
        predictions = wrapper.predict_letter_sequence(
            letter_images,
            show_visualization=False,
            word_index=word_idx
        )
        
        word = ''.join(predictions)
        predicted_words.append(word)
    
    full_text = ' '.join(predicted_words)
    
    # Guardar texto
    text_path = os.path.join(output_dir, 'extracted_text.txt')
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"  ✓ Texto extraído: {len(predicted_words)} palabras")
    print(f"  ✓ Guardado en: {text_path}")
    
    # 2. Extraer tablas
    if extract_tables:
        print("\n[2/3] Extrayendo tablas...")
        table_detector = TableDetector()
        tables_dir = os.path.join(output_dir, 'tables')
        table_paths = table_detector.save_tables(image_path, tables_dir)
        print(f"  ✓ Tablas extraídas: {len(table_paths)}")
    else:
        print("\n[2/3] Extracción de tablas omitida (--extract-tables para activar)")
    
    # 3. Extraer figuras
    if extract_figures:
        print("\n[3/3] Extrayendo figuras...")
        figure_detector = FigureDetector()
        figures_dir = os.path.join(output_dir, 'figures')
        figure_paths = figure_detector.save_figures(image_path, figures_dir)
        print(f"  ✓ Figuras extraídas: {len(figure_paths)}")
    else:
        print("\n[3/3] Extracción de figuras omitida (--extract-figures para activar)")
    
    # Resumen
    print(f"\n{'='*70}")
    print(f" PROCESAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"Documento: {os.path.basename(image_path)}")
    print(f"Texto extraído: {len(full_text)} caracteres")
    print(f"Directorio de salida: {output_dir}")
    print(f"{'='*70}\n")


def main():
    """Función principal con argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Sistema OCR - Reconocimiento de Caracteres Manuscritos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  
  # OCR básico (solo texto)
  python main.py --image documento.png --model models/weights/modelo.pth
  
  # OCR completo (texto + tablas + figuras)
  python main.py --image documento.png --model models/weights/modelo.pth \\
                 --extract-tables --extract-figures
  
  # Especificar directorio de salida
  python main.py --image documento.png --model models/weights/modelo.pth \\
                 --output ./resultados
        """
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Ruta a la imagen del documento a procesar'
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
        default='./ocr_output',
        help='Directorio donde guardar los resultados (default: ./ocr_output)'
    )
    
    parser.add_argument(
        '--extract-tables',
        action='store_true',
        help='Extraer tablas del documento'
    )
    
    parser.add_argument(
        '--extract-figures',
        action='store_true',
        help='Extraer figuras/imágenes del documento'
    )
    
    args = parser.parse_args()
    
    # Verificar que existen los archivos
    if not os.path.exists(args.image):
        print(f"Error: No se encontró la imagen: {args.image}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: No se encontró el modelo: {args.model}")
        sys.exit(1)
    
    # Procesar documento
    ocr_document_full(
        image_path=args.image,
        model_path=args.model,
        output_dir=args.output,
        extract_tables=args.extract_tables,
        extract_figures=args.extract_figures
    )


if __name__ == '__main__':
    main()