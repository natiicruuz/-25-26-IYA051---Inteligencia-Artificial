"""
Script helper para encontrar el mejor modelo y hacer predicciones.

Encuentra automáticamente el modelo más reciente y permite probar rápidamente.
"""

import os
import sys
import glob
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.predict_char import predict_character
from inference.predict_word import predict_word
from inference.predict_document import predict_document


def find_latest_model(weights_dir='./models/weights', pattern='best_normalized*.pth'):
    """
    Encuentra el modelo más reciente en el directorio de weights.
    
    Args:
        weights_dir: Directorio donde buscar modelos.
        pattern: Patrón de nombre de archivo.
    
    Returns:
        Ruta al modelo más reciente o None.
    """
    search_path = os.path.join(weights_dir, pattern)
    models = glob.glob(search_path)
    
    if not models:
        return None
    
    # Ordenar por fecha de modificación (más reciente primero)
    models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return models[0]


def list_available_models(weights_dir='./models/weights'):
    """
    Lista todos los modelos disponibles.
    
    Args:
        weights_dir: Directorio donde buscar.
    """
    print("\n" + "="*70)
    print(" MODELOS DISPONIBLES")
    print("="*70 + "\n")
    
    all_models = glob.glob(os.path.join(weights_dir, '*.pth'))
    
    if not all_models:
        print("⚠️  No se encontraron modelos en", weights_dir)
        return
    
    # Agrupar por tipo
    normalized = [m for m in all_models if 'normalized' in os.path.basename(m)]
    custom = [m for m in all_models if 'custom' in os.path.basename(m)]
    combined = [m for m in all_models if 'combined' in os.path.basename(m)]
    
    if normalized:
        print(f"📦 Modelos Normalizados ({len(normalized)}):")
        for model in sorted(normalized, key=lambda x: os.path.getmtime(x), reverse=True)[:5]:
            size_mb = os.path.getsize(model) / (1024*1024)
            print(f"  • {os.path.basename(model)}")
            print(f"    └─ {size_mb:.2f} MB")
        if len(normalized) > 5:
            print(f"  ... y {len(normalized)-5} más")
    
    if custom:
        print(f"\n📦 Modelos Custom ({len(custom)}):")
        for model in sorted(custom, key=lambda x: os.path.getmtime(x), reverse=True)[:3]:
            size_mb = os.path.getsize(model) / (1024*1024)
            print(f"  • {os.path.basename(model)}")
            print(f"    └─ {size_mb:.2f} MB")
    
    if combined:
        print(f"\n📦 Modelos Combinados ({len(combined)}):")
        for model in sorted(combined, key=lambda x: os.path.getmtime(x), reverse=True)[:3]:
            size_mb = os.path.getsize(model) / (1024*1024)
            print(f"  • {os.path.basename(model)}")
            print(f"    └─ {size_mb:.2f} MB")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Helper para usar modelos entrenados fácilmente',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:

  # Listar modelos disponibles
  python quick_predict.py --list
  
  # Predecir con el modelo más reciente (automático)
  python quick_predict.py --char imagen.png
  
  # Predecir una palabra
  python quick_predict.py --word palabra.png
  
  # Predecir documento completo
  python quick_predict.py --document doc.png --output resultado.txt
  
  # Usar un modelo específico
  python quick_predict.py --char imagen.png --model "models/weights/modelo.pth"
        """
    )
    
    parser.add_argument('--list', action='store_true', help='Listar modelos disponibles')
    parser.add_argument('--char', type=str, help='Predecir un carácter')
    parser.add_argument('--word', type=str, help='Predecir una palabra')
    parser.add_argument('--document', type=str, help='Predecir documento completo')
    parser.add_argument('--model', type=str, help='Ruta al modelo (opcional, usa el más reciente)')
    parser.add_argument('--output', type=str, help='Archivo de salida para documento')
    parser.add_argument('--no-show', action='store_true', help='No mostrar visualizaciones')
    
    args = parser.parse_args()
    
    # Listar modelos
    if args.list:
        list_available_models()
        return
    
    # Verificar que se especificó al menos una acción
    if not (args.char or args.word or args.document):
        parser.print_help()
        return
    
    # Encontrar modelo
    if args.model:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"❌ ERROR: Modelo no encontrado: {model_path}")
            sys.exit(1)
    else:
        print("🔍 Buscando modelo más reciente...")
        model_path = find_latest_model()
        
        if not model_path:
            print("❌ ERROR: No se encontraron modelos")
            print("\nBusca en: ./models/weights/")
            print("\nO entrena un modelo primero:")
            print("  python training\\train_normalized.py --dataset \"ruta\"")
            sys.exit(1)
        
        print(f"✓ Usando: {os.path.basename(model_path)}")
    
    # Ejecutar predicción
    try:
        if args.char:
            predict_character(args.char, model_path, show_image=not args.no_show)
        
        elif args.word:
            predict_word(args.word, model_path, show_visualization=not args.no_show)
        
        elif args.document:
            output = args.output or 'output.txt'
            predict_document(args.document, model_path, output_path=output, show_visualization=not args.no_show)
    
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()