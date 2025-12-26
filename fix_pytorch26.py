"""
Script de actualización automática para compatibilidad con PyTorch 2.6.

Este script actualiza todos los archivos que usan torch.load() para que
funcionen con PyTorch 2.6 añadiendo weights_only=False.
"""

import os
import re


def update_file(filepath, description):
    """
    Actualiza un archivo reemplazando torch.load() sin weights_only.
    
    Args:
        filepath: Ruta al archivo a actualizar.
        description: Descripción del archivo.
    
    Returns:
        True si se actualizó, False si no se encontró o ya estaba actualizado.
    """
    if not os.path.exists(filepath):
        print(f"  ⚠️  No encontrado: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar torch.load sin weights_only=False
    pattern = r'torch\.load\s*\(\s*([^,)]+)\s*,\s*map_location\s*=\s*([^)]+)\s*\)'
    
    # Verificar si ya tiene weights_only=False
    if 'weights_only=False' in content:
        print(f"  ✓ Ya actualizado: {description}")
        return False
    
    # Reemplazar con weights_only=False
    def replacement(match):
        arg1 = match.group(1)
        arg2 = match.group(2)
        return f'torch.load({arg1}, map_location={arg2}, weights_only=False)'
    
    new_content, count = re.subn(pattern, replacement, content)
    
    if count > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  ✓ Actualizado: {description} ({count} cambio(s))")
        return True
    else:
        print(f"  - Sin cambios: {description}")
        return False


def main():
    """Función principal de actualización."""
    print("\n" + "="*70)
    print(" ACTUALIZACIÓN AUTOMÁTICA PARA PYTORCH 2.6")
    print("="*70 + "\n")
    
    print("Este script actualiza todos los archivos para que funcionen")
    print("correctamente con PyTorch 2.6 (añade weights_only=False)\n")
    
    # Lista de archivos a actualizar
    files_to_update = [
        ('inference/predict_char.py', 'Predicción de caracteres'),
        ('inference/predict_word.py', 'Predicción de palabras'),
        ('inference/predict_document.py', 'Predicción de documentos'),
        ('main.py', 'Script principal'),
        ('models/cnn.py', 'Modelo CNN'),
        ('test_trained_models.py', 'Test de modelos'),
    ]
    
    updated_count = 0
    
    for filepath, description in files_to_update:
        if update_file(filepath, description):
            updated_count += 1
    
    print("\n" + "="*70)
    
    if updated_count > 0:
        print(f" ✅ ACTUALIZACIÓN COMPLETADA")
        print("="*70)
        print(f"\nArchivos actualizados: {updated_count}")
        print("\nAhora puedes usar los scripts de predicción sin problemas:")
        print('  python inference\\predict_char.py --image "..." --model "..."')
        print('  python quick_predict.py --char "..."')
    else:
        print(f" ℹ️  NO SE NECESITARON ACTUALIZACIONES")
        print("="*70)
        print("\nTodos los archivos ya están actualizados.")
    
    print()


if __name__ == '__main__':
    main()