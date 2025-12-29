"""
PASO 0: Normalización automática de datasets OCR.

Este script normaliza datasets de compañeros y los prepara para entrenamiento.

Estructura de entrada esperada (datasets de compañeros):
    data/raw/
    ├── mayusculas/
    │   ├── A/
    │   ├── B/
    │   └── ...
    ├── minusculas/
    │   ├── a/
    │   ├── b/
    │   └── ...
    └── numeros/
        ├── 0/
        ├── 1/
        └── ...

Estructura de salida (para entrenamiento):
    data/normalized/
    ├── A/
    ├── a/
    ├── 0/
    └── ...
"""

import os
import sys
from pathlib import Path
import argparse
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from tqdm import tqdm


def normalize_single_image(img_array):
    """
    Normaliza una imagen de carácter manuscrito.
    
    Proceso:
    1. Detectar si letra es oscura o clara
    2. Invertir si es necesario (letra→blanca, fondo→negro)
    3. Binarizar con umbral Otsu
    4. Eliminar ruido
    
    Args:
        img_array: Array numpy con la imagen
    
    Returns:
        Imagen normalizada (array numpy)
    """
    # Detectar orientación (letra oscura o clara)
    mean_pixel = np.mean(img_array)
    
    # Si letra es oscura sobre fondo claro, invertir
    if mean_pixel > 127:
        img_array = 255 - img_array
    
    # Suavizar para eliminar ruido
    from scipy.ndimage import gaussian_filter
    img_smooth = gaussian_filter(img_array, sigma=0.5)
    
    # Umbral Otsu para binarizar
    hist, _ = np.histogram(img_smooth.flatten(), bins=256, range=[0, 256])
    hist = hist.astype(float)
    
    total = img_smooth.size
    sumT = np.dot(np.arange(256), hist)
    sumB = 0
    wB = 0
    
    current_max = 0
    threshold = 0
    
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sumT - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between > current_max:
            current_max = between
            threshold = i
    
    # Aplicar umbral
    binary = (img_smooth > threshold).astype(np.uint8) * 255
    
    return binary


def normalize_image(image_path, target_size=(28, 28)):
    """
    Normaliza una imagen completa.
    
    Args:
        image_path: Ruta a la imagen original
        target_size: Tamaño objetivo
    
    Returns:
        PIL Image normalizada o None si falla
    """
    try:
        # Cargar y convertir a escala de grises
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        
        # Normalizar
        binary = normalize_single_image(img_array)
        img_binary = Image.fromarray(binary)
        
        # Recortar espacio en blanco
        bbox = img_binary.getbbox()
        if bbox:
            img_cropped = img_binary.crop(bbox)
        else:
            img_cropped = img_binary
        
        # Redimensionar manteniendo aspect ratio
        img_cropped.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Crear imagen con padding
        img_padded = Image.new('L', target_size, color=0)  # Fondo negro
        offset = ((target_size[0] - img_cropped.width) // 2,
                 (target_size[1] - img_cropped.height) // 2)
        img_padded.paste(img_cropped, offset)
        
        return img_padded
        
    except Exception as e:
        print(f"  ⚠️  Error: {str(e)}")
        return None


def normalize_dataset(input_dir, output_dir, target_size=(28, 28)):
    """
    Normaliza dataset completo.
    """
    print("\n" + "="*70)
    print(" PASO 0: NORMALIZACIÓN DE DATASET")
    print("="*70 + "\n")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ ERROR: No existe {input_dir}")
        return False
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = {'total': 0, 'exitosas': 0, 'fallidas': 0}
    
    # Procesar cada categoría
    for categoria in ['mayusculas', 'minusculas', 'numeros']:
        cat_path = input_path / categoria
        
        if not cat_path.exists():
            print(f"⚠️  No encontrado: {categoria}/")
            continue
        
        print(f"\n{'='*70}")
        print(f" {categoria.upper()}")
        print(f"{'='*70}")
        
        # Procesar cada carácter
        for char_folder in sorted(cat_path.iterdir()):
            if not char_folder.is_dir():
                continue
            
            char_name = char_folder.name
            
            # Determinar nombre de salida
            if categoria == 'numeros':
                output_char = char_name
            elif categoria == 'mayusculas':
                output_char = char_name.upper()
            else:
                output_char = char_name.lower()
            
            # Buscar imágenes
            images = [f for f in char_folder.glob('*.*') 
                     if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
            
            if not images:
                continue
            
            print(f"\n  '{output_char}' - {len(images)} imágenes")
            
            # Crear carpeta de salida
            output_folder = output_path / output_char
            output_folder.mkdir(exist_ok=True)
            
            # Normalizar cada imagen
            for img_path in tqdm(images, desc=f"    Normalizando", leave=False):
                stats['total'] += 1
                
                normalized = normalize_image(img_path, target_size)
                
                if normalized:
                    output_file = output_folder / img_path.name
                    normalized.save(output_file)
                    stats['exitosas'] += 1
                else:
                    stats['fallidas'] += 1
    
    # Resumen
    print("\n" + "="*70)
    print(" RESUMEN")
    print("="*70)
    print(f"Total: {stats['total']} imágenes")
    print(f"Exitosas: {stats['exitosas']}")
    print(f"Fallidas: {stats['fallidas']}")
    
    if stats['exitosas'] > 0:
        tasa = 100 * stats['exitosas'] / stats['total']
        print(f"Tasa de éxito: {tasa:.1f}%")
    
    print(f"\n✅ Dataset normalizado en: {output_path}")
    print("\nPróximo paso:")
    print(f"  python verify_normalized.py --dataset \"{output_path}\"")
    print("="*70 + "\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='PASO 0: Normalizar dataset OCR'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='./data/raw',
        help='Directorio con dataset raw (default: ./data/raw)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./data/normalized',
        help='Directorio de salida (default: ./data/normalized)'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=28,
        help='Tamaño de las imágenes (default: 28)'
    )
    
    args = parser.parse_args()
    
    success = normalize_dataset(
        args.input,
        args.output,
        (args.size, args.size)
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()