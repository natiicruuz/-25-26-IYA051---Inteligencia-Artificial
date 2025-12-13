"""
Script para verificar modelos entrenados.

Prueba que los modelos se carguen correctamente y puedan hacer predicciones.
"""

import os
import sys
import glob
import torch
from datetime import datetime

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cnn import OCRCNN
from models.config import OCRConfig


def format_size(bytes_size):
    """Formatea el tamaño de archivo en MB."""
    return f"{bytes_size / (1024*1024):.2f} MB"


def test_model(model_path):
    """
    Prueba un modelo entrenado.
    
    Args:
        model_path: Ruta al archivo .pth del modelo.
    
    Returns:
        dict con información del modelo.
    """
    print(f"\n{'─'*70}")
    print(f"Probando: {os.path.basename(model_path)}")
    print(f"{'─'*70}")
    
    # Información del archivo
    file_size = os.path.getsize(model_path)
    modified_time = datetime.fromtimestamp(os.path.getmtime(model_path))
    
    print(f"  Tamaño: {format_size(file_size)}")
    print(f"  Última modificación: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Intentar cargar el modelo
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Verificar contenido del checkpoint
        if 'model_state_dict' not in checkpoint:
            print("  ❌ ERROR: Checkpoint no contiene 'model_state_dict'")
            print("  ⚠️  El archivo puede estar corrupto o incompleto")
            return None
        
        if 'config' not in checkpoint:
            print("  ❌ ERROR: Checkpoint no contiene 'config'")
            return None
        
        # Cargar configuración
        config = checkpoint['config']
        
        print(f"\n  ✓ Checkpoint cargado correctamente")
        print(f"  ├─ Dispositivo: {device}")
        print(f"  ├─ Número de clases: {config.num_classes}")
        print(f"  ├─ Input shape: {config.input_shape}")
        print(f"  └─ Learning rate: {config.learning_rate}")
        
        # Intentar crear y cargar el modelo
        model = OCRCNN(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Contar parámetros
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n  ✓ Modelo cargado en memoria")
        print(f"  ├─ Parámetros totales: {total_params:,}")
        print(f"  └─ Parámetros entrenables: {trainable_params:,}")
        
        # Hacer una predicción de prueba
        dummy_input = torch.randn(1, 1, 28, 28).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"\n  ✓ Predicción de prueba exitosa")
        print(f"  ├─ Shape de salida: {output.shape}")
        print(f"  └─ Clase predicha: {torch.argmax(output).item()}")
        
        # Verificar si hay optimizador guardado
        if 'optimizer_state_dict' in checkpoint:
            print(f"\n  ✓ Optimizer state guardado (puede reanudar entrenamiento)")
        
        print(f"\n  {'═'*68}")
        print(f"  ✅ MODELO VÁLIDO Y FUNCIONAL")
        print(f"  {'═'*68}")
        
        return {
            'path': model_path,
            'size': file_size,
            'modified': modified_time,
            'num_classes': config.num_classes,
            'total_params': total_params,
            'status': 'OK'
        }
        
    except Exception as e:
        print(f"\n  ❌ ERROR al cargar el modelo:")
        print(f"  {str(e)}")
        print(f"\n  ⚠️  Este modelo puede estar corrupto o incompleto")
        
        return {
            'path': model_path,
            'size': file_size,
            'modified': modified_time,
            'status': 'ERROR',
            'error': str(e)
        }


def main():
    """Función principal."""
    print("\n" + "="*70)
    print(" VERIFICACIÓN DE MODELOS ENTRENADOS")
    print("="*70)
    
    # Buscar todos los modelos .pth en weights/
    weights_dir = os.path.join(os.path.dirname(__file__), 'models', 'weights')
    
    if not os.path.exists(weights_dir):
        print(f"\n❌ ERROR: No se encontró el directorio: {weights_dir}")
        print("\nAsegúrate de estar en el directorio raíz del proyecto.")
        sys.exit(1)
    
    model_files = glob.glob(os.path.join(weights_dir, '*.pth'))
    
    if not model_files:
        print(f"\n⚠️  No se encontraron modelos (.pth) en: {weights_dir}")
        print("\n¿Ya entrenaste algún modelo?")
        print("\nPara entrenar:")
        print("  python training\\train_handwritten.py --dataset \"C:\\ruta\\dataset\" --epochs 90")
        sys.exit(0)
    
    print(f"\nEncontrados {len(model_files)} modelo(s) en {weights_dir}")
    
    # Probar cada modelo
    results = []
    for model_file in sorted(model_files):
        result = test_model(model_file)
        if result:
            results.append(result)
    
    # Resumen final
    print("\n" + "="*70)
    print(" RESUMEN")
    print("="*70)
    
    ok_models = [r for r in results if r['status'] == 'OK']
    error_models = [r for r in results if r['status'] == 'ERROR']
    
    print(f"\n✓ Modelos válidos: {len(ok_models)}/{len(results)}")
    
    if ok_models:
        print(f"\nModelos listos para usar:")
        for model in ok_models:
            print(f"  • {os.path.basename(model['path'])}")
            print(f"    └─ {model['num_classes']} clases, {format_size(model['size'])}")
    
    if error_models:
        print(f"\n⚠️  Modelos con errores: {len(error_models)}")
        for model in error_models:
            print(f"  • {os.path.basename(model['path'])}")
            print(f"    └─ Error: {model.get('error', 'Desconocido')}")
    
    # Buscar imágenes de gráficas
    print(f"\n" + "─"*70)
    print("Buscando gráficas de entrenamiento...")
    print("─"*70)
    
    graph_files = glob.glob(os.path.join(weights_dir, '*.png'))
    
    if graph_files:
        print(f"\nEncontradas {len(graph_files)} gráfica(s):")
        for graph in sorted(graph_files):
            print(f"  • {os.path.basename(graph)}")
        print(f"\nAbre estas imágenes para ver cómo fue el entrenamiento.")
    else:
        print(f"\n⚠️  No se encontraron gráficas (.png)")
    
    print("\n" + "="*70)
    
    if ok_models:
        print(" ✅ TODO LISTO - Tienes modelos entrenados y funcionales")
        print("="*70)
        print("\nPRÓXIMOS PASOS:")
        print("\n1. Ver las gráficas para verificar que el entrenamiento fue bueno")
        print("\n2. Probar el modelo en una imagen:")
        print(f"   python inference\\predict_char.py --image imagen.png --model \"{ok_models[0]['path']}\"")
        print("\n3. O usar el pipeline completo:")
        print(f"   python main.py --image documento.png --model \"{ok_models[0]['path']}\"")
    else:
        print(" ⚠️  NO HAY MODELOS VÁLIDOS")
        print("="*70)
        print("\nEs posible que el entrenamiento se haya interrumpido.")
        print("Deberás volver a entrenar el modelo.")
    
    print()


if __name__ == '__main__':
    main()