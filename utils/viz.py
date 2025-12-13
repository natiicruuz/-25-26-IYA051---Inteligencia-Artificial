"""
Utilidades para visualización de datos e imágenes.

Este módulo contiene funciones para visualizar imágenes, predicciones,
métricas de entrenamiento y otros datos relevantes del modelo OCR.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
import torch


def plot_training_loss(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Grafica la pérdida durante el entrenamiento.
    
    Args:
        train_losses: Lista de pérdidas de entrenamiento por época.
        val_losses: Lista opcional de pérdidas de validación por época.
        save_path: Ruta opcional para guardar la figura.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Entrenamiento', linewidth=2)
    
    if val_losses:
        plt.plot(epochs, val_losses, 'r-s', label='Validación', linewidth=2)
    
    plt.title('Pérdida durante el entrenamiento', fontsize=14, fontweight='bold')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_metrics(
    train_losses: List[float],
    train_accuracies: List[float],
    val_losses: Optional[List[float]] = None,
    val_accuracies: Optional[List[float]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Grafica pérdida y precisión durante el entrenamiento en subplots.
    
    Args:
        train_losses: Pérdidas de entrenamiento.
        train_accuracies: Precisiones de entrenamiento.
        val_losses: Pérdidas de validación opcionales.
        val_accuracies: Precisiones de validación opcionales.
        save_path: Ruta opcional para guardar la figura.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(train_losses) + 1)
    
    # Subplot de pérdida
    ax1.plot(epochs, train_losses, 'b-o', label='Entrenamiento', linewidth=2)
    if val_losses:
        ax1.plot(epochs, val_losses, 'r-s', label='Validación', linewidth=2)
    ax1.set_title('Pérdida', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Pérdida', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Subplot de precisión
    ax2.plot(epochs, train_accuracies, 'b-o', label='Entrenamiento', linewidth=2)
    if val_accuracies:
        ax2.plot(epochs, val_accuracies, 'r-s', label='Validación', linewidth=2)
    ax2.set_title('Precisión', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Precisión (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_batch(
    images: torch.Tensor,
    labels: List[str],
    predictions: Optional[List[str]] = None,
    num_samples: int = 9,
    save_path: Optional[str] = None
) -> None:
    """
    Visualiza un batch de imágenes con sus etiquetas y predicciones.
    
    Args:
        images: Tensor de imágenes (B, C, H, W).
        labels: Lista de etiquetas verdaderas.
        predictions: Lista opcional de predicciones.
        num_samples: Número de muestras a visualizar.
        save_path: Ruta opcional para guardar la figura.
    """
    num_samples = min(num_samples, len(images))
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        img = images[i].squeeze().cpu().numpy()
        
        axes[i].imshow(img, cmap='gray')
        
        if predictions:
            title = f"Real: {labels[i]}\nPred: {predictions[i]}"
            color = 'green' if labels[i] == predictions[i] else 'red'
        else:
            title = f"Etiqueta: {labels[i]}"
            color = 'black'
        
        axes[i].set_title(title, fontsize=10, color=color)
        axes[i].axis('off')
    
    # Ocultar ejes sobrantes
    for j in range(num_samples, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def show_predictions_grid(
    images: List[np.ndarray],
    predictions: List[str],
    true_labels: Optional[List[str]] = None,
    title: str = "Predicciones",
    save_path: Optional[str] = None
) -> None:
    """
    Muestra una grilla de imágenes con sus predicciones.
    
    Args:
        images: Lista de imágenes numpy.
        predictions: Lista de predicciones.
        true_labels: Lista opcional de etiquetas verdaderas.
        title: Título de la figura.
        save_path: Ruta opcional para guardar.
    """
    n = len(images)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (img, pred) in enumerate(zip(images, predictions)):
        if i >= len(axes):
            break
        
        axes[i].imshow(img, cmap='gray')
        
        if true_labels and i < len(true_labels):
            label_text = f"Real: {true_labels[i]}\nPred: {pred}"
            color = 'green' if true_labels[i] == pred else 'red'
        else:
            label_text = f"Pred: {pred}"
            color = 'blue'
        
        axes[i].set_title(label_text, fontsize=10, color=color)
        axes[i].axis('off')
    
    # Ocultar ejes vacíos
    for j in range(len(images), len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_sample(
    true_labels: List[int],
    predicted_labels: List[int],
    mapping: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Visualiza un scatter plot comparando etiquetas verdaderas vs predichas.
    
    Args:
        true_labels: Etiquetas verdaderas (índices).
        predicted_labels: Etiquetas predichas (índices).
        mapping: Mapeo de índices a caracteres.
        save_path: Ruta opcional para guardar.
    """
    plt.figure(figsize=(12, 8))
    
    plt.scatter(true_labels, predicted_labels, alpha=0.5, s=20)
    plt.plot([min(true_labels), max(true_labels)], 
             [min(true_labels), max(true_labels)], 
             'r--', linewidth=2, label='Predicción perfecta')
    
    plt.xlabel('Etiquetas verdaderas', fontsize=12)
    plt.ylabel('Etiquetas predichas', fontsize=12)
    plt.title('Comparación de predicciones', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def display_letter_sequence(
    letter_images: List[np.ndarray],
    predictions: List[str],
    word_index: int = 0,
    save_path: Optional[str] = None
) -> None:
    """
    Muestra una secuencia de letras con sus predicciones.
    
    Args:
        letter_images: Lista de imágenes de letras.
        predictions: Lista de caracteres predichos.
        word_index: Índice de la palabra (para el título).
        save_path: Ruta opcional para guardar.
    """
    n = len(letter_images)
    
    fig, axes = plt.subplots(1, n, figsize=(2*n, 3))
    
    if n == 1:
        axes = [axes]
    
    for i, (img, pred) in enumerate(zip(letter_images, predictions)):
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
        axes[i].set_title(f"L{i+1}: {pred}", fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle(f"Palabra {word_index + 1}: {''.join(predictions)}", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()