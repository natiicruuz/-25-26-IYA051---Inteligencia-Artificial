"""
Script de predicción para caracteres normalizados - VERSIÓN FINAL.

Compatible con:
- Dataset normalizado (fondo negro, letra blanca)
- Modelos entrenados con custom + EMNIST
- PyTorch 2.6+
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import transforms
from models.config import OCRConfig
from models.cnn import OCRCNN
from models.ocr_model import OCRModelWrapper


def create_transforms_for_normalized():
    """
    Transformaciones para imágenes YA NORMALIZADAS.
    
    CRÍTICO: NO aplicar Normalize() porque las imágenes del dataset
    normalizado ya tienen fondo negro (0) y letra blanca (255).
    """
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        # SIN Normalize((0.5,), (0.5,)) - ya está normalizado
    ])


def predict_character(
    image_path: str,
    model_path: str,
    show_image: bool = True
) -> str:
    """
    Predice el carácter en una imagen normalizada.
    
    Args:
        image_path: Ruta a la imagen del carácter
        model_path: Ruta al modelo entrenado (.pth)
        show_image: Si True, muestra la imagen
    
    Returns:
        Carácter predicho
    """
    print(f"\n{'='*70}")
    print(f" PREDICCIÓN DE CARÁCTER")
    print(f"{'='*70}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
    
    print(f"\nImagen: {image_path}")
    print(f"Modelo: {model_path}")
    
    # Cargar modelo
    print("\nCargando modelo...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        print("Verifica que el archivo .pth sea válido")
        sys.exit(1)
    
    config = checkpoint['config']
    model = OCRCNN(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Modelo cargado correctamente")
    print(f"  Dispositivo: {device}")
    print(f"  Clases: {config.num_classes}")
    
    # Crear transformaciones SIN normalización extra
    transform = create_transforms_for_normalized()
    wrapper = OCRModelWrapper(model, transform, config)
    
    # Predecir
    print("\nRealizando predicción...")
    try:
        predicted_char = wrapper.predict_single_char(image_path, show_image=show_image)
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f" RESULTADO: '{predicted_char}'")
    print(f"{'='*70}\n")
    
    return predicted_char


def main():
    parser = argparse.ArgumentParser(
        description='Predecir carácter desde imagen normalizada',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Predicción básica
  python inference/predict_char.py --image test.png --model models/weights/final_custom_emnist_*.pth
  
  # Sin mostrar imagen
  python inference/predict_char.py --image test.png --model models/weights/modelo.pth --no-show
        """
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
        help='No mostrar la imagen (solo resultado)'
    )
    
    args = parser.parse_args()
    
    predicted = predict_character(
        image_path=args.image,
        model_path=args.model,
        show_image=not args.no_show
    )
    
    return predicted


if __name__ == '__main__':
    main()