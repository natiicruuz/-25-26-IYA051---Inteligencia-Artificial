"""
Script de diagnóstico para entender problemas con el modelo OCR.

Este script:
1. Verifica el dataset
2. Analiza el modelo entrenado
3. Prueba predicciones
4. Identifica problemas comunes
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from collections import Counter

from models.config import OCRConfig
from models.cnn import OCRCNN
from datasets.normalized_dataset import NormalizedDataset


def diagnose_dataset(dataset_path):
    """Diagnostica el dataset."""
    print("\n" + "="*70)
    print(" DIAGNÓSTICO DEL DATASET")
    print("="*70 + "\n")
    
    # Cargar dataset
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    
    config = OCRConfig(num_classes=64)
    dataset = NormalizedDataset(dataset_path, transform, config)
    
    # Estadísticas básicas
    print(f"Total de imágenes: {len(dataset)}")
    
    # Distribución de clases
    distribution = dataset.get_class_distribution()
    
    print(f"\nDistribución de clases:")
    print(f"  Número de clases: {len(distribution)}")
    
    # Encontrar desbalances
    counts = list(distribution.values())
    min_count = min(counts)
    max_count = max(counts)
    avg_count = sum(counts) / len(counts)
    
    print(f"\n  Mínimo por clase: {min_count}")
    print(f"  Máximo por clase: {max_count}")
    print(f"  Promedio por clase: {avg_count:.1f}")
    
    ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"  Ratio desbalance: {ratio:.2f}x")
    
    if ratio > 5:
        print(f"\n  ⚠️  PROBLEMA: Dataset muy desbalanceado (ratio > 5x)")
        print(f"     Clases con pocas imágenes tendrán mal desempeño")
    elif ratio > 3:
        print(f"\n  ⚠️  Dataset moderadamente desbalanceado (ratio > 3x)")
    else:
        print(f"\n  ✅ Dataset razonablemente balanceado")
    
    # Clases problemáticas
    print(f"\nClases con menos imágenes:")
    sorted_dist = sorted(distribution.items(), key=lambda x: x[1])
    for char, count in sorted_dist[:10]:
        status = "⚠️" if count < 20 else "✓"
        print(f"  {status} '{char}': {count} imágenes")
    
    # Verificar calidad de imágenes
    print(f"\nVerificando calidad de imágenes...")
    sample_indices = np.random.choice(len(dataset), min(10, len(dataset)), replace=False)
    
    pixel_means = []
    pixel_stds = []
    
    for idx in sample_indices:
        img_tensor, _ = dataset[idx]
        pixel_means.append(img_tensor.mean().item())
        pixel_stds.append(img_tensor.std().item())
    
    avg_mean = np.mean(pixel_means)
    avg_std = np.mean(pixel_stds)
    
    print(f"  Promedio de píxeles: {avg_mean:.3f}")
    print(f"  Desviación estándar: {avg_std:.3f}")
    
    if avg_mean < 0.1 or avg_mean > 0.9:
        print(f"  ⚠️  Imágenes muy oscuras/claras - pueden necesitar ajuste")
    else:
        print(f"  ✓ Rango de píxeles razonable")
    
    return len(distribution), min_count, ratio


def diagnose_model(model_path, dataset_path):
    """Diagnostica el modelo entrenado."""
    print("\n" + "="*70)
    print(" DIAGNÓSTICO DEL MODELO")
    print("="*70 + "\n")
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        return None
    
    # Cargar modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    model = OCRCNN(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Modelo cargado: {os.path.basename(model_path)}")
    print(f"Número de clases: {config.num_classes}")
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Cargar dataset
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    
    dataset = NormalizedDataset(dataset_path, transform, config)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    
    # Evaluar en TODO el dataset
    print(f"\nEvaluando en {len(dataset)} imágenes...")
    
    all_labels = []
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    # Calcular precisión global
    accuracy = 100 * (all_predictions == all_labels).sum() / len(all_labels)
    print(f"\nPrecisión global: {accuracy:.2f}%")
    
    # Precisión por clase
    char_mapping = config.get_character_mapping()
    class_accuracies = {}
    
    for i, char in enumerate(char_mapping):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = 100 * (all_predictions[mask] == all_labels[mask]).sum() / mask.sum()
            class_accuracies[char] = class_acc
    
    # Clases con peor desempeño
    print(f"\nClases con PEOR desempeño:")
    sorted_acc = sorted(class_accuracies.items(), key=lambda x: x[1])
    for char, acc in sorted_acc[:15]:
        status = "❌" if acc < 50 else "⚠️" if acc < 75 else "✓"
        print(f"  {status} '{char}': {acc:.1f}%")
    
    # Confusiones más comunes
    print(f"\nConfusiones más comunes:")
    confusion_pairs = Counter()
    
    for true_label, pred_label in zip(all_labels, all_predictions):
        if true_label != pred_label:
            true_char = char_mapping[true_label]
            pred_char = char_mapping[pred_label]
            confusion_pairs[(true_char, pred_char)] += 1
    
    for (true_char, pred_char), count in confusion_pairs.most_common(10):
        print(f"  '{true_char}' → '{pred_char}': {count} veces")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Diagnosticar problemas con el OCR'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Ruta al dataset'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Ruta al modelo (opcional)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" DIAGNÓSTICO COMPLETO DEL SISTEMA OCR")
    print("="*70)
    
    # Diagnosticar dataset
    num_classes, min_samples, ratio = diagnose_dataset(args.dataset)
    
    # Diagnosticar modelo si se proporciona
    accuracy = None
    if args.model:
        accuracy = diagnose_model(args.model, args.dataset)
    
    # Recomendaciones finales
    print("\n" + "="*70)
    print(" RECOMENDACIONES")
    print("="*70 + "\n")
    
    problems = []
    
    if min_samples < 20:
        problems.append("Dataset muy pequeño (< 20 imágenes por clase)")
        print("❌ Dataset muy pequeño")
        print("   → Agregar más imágenes (de compañeros)")
        print("   → Mínimo recomendado: 50 imágenes por clase")
    
    if ratio > 3:
        problems.append("Dataset desbalanceado")
        print("⚠️  Dataset desbalanceado")
        print("   → Balancear clases (mismas imágenes por letra)")
        print("   → O usar data augmentation agresivo")
    
    if accuracy and accuracy < 70:
        problems.append("Precisión muy baja (< 70%)")
        print("❌ Modelo con precisión muy baja")
        print("   → Reentrenar con más datos")
        print("   → Verificar que imágenes estén bien normalizadas")
        print("   → Aumentar épocas de entrenamiento")
    elif accuracy and accuracy < 85:
        print("⚠️  Modelo con precisión mejorable (70-85%)")
        print("   → Agregar más datos ayudaría")
    
    if not problems:
        print("✅ ¡Todo se ve bien!")
        print("   El sistema debería funcionar correctamente.")
    
    print()


if __name__ == '__main__':
    main()