"""
Wrapper del modelo OCR para predicción e inferencia.

Este módulo proporciona una interfaz de alto nivel para usar el modelo CNN
entrenado en tareas de predicción y visualización.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Optional
import matplotlib.pyplot as plt
import cv2

from models.cnn import OCRCNN
from models.config import OCRConfig
from utils.image_utils import resize_with_padding, convert_to_grayscale


class OCRModelWrapper:
    """
    Wrapper para el modelo OCR que facilita predicciones y visualizaciones.
    """
    
    def __init__(self, cnn_model: OCRCNN, transform, config: OCRConfig):
        """
        Inicializa el wrapper del modelo.
        
        Args:
            cnn_model: Modelo CNN entrenado.
            transform: Transformaciones de PyTorch a aplicar a las imágenes.
            config: Configuración del modelo.
        """
        self.model = cnn_model
        self.transform = transform
        self.config = config
        self.device = cnn_model.device
        self.mapping = config.get_character_mapping()
        
        # Poner modelo en modo evaluación
        self.model.eval()
    
    def evaluate(self, test_loader, verbose: bool = True) -> dict:
        """
        Evalúa el modelo en un conjunto de test y retorna métricas.
        
        Args:
            test_loader: DataLoader con los datos de test.
            verbose: Si True, imprime los resultados.
        
        Returns:
            Diccionario con las métricas calculadas.
        """
        self.model.eval()
        
        correct = 0
        total = 0
        all_labels = []
        all_predicted = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())
        
        accuracy = 100 * correct / total
        
        if verbose:
            print(f'\n{"="*50}')
            print(f'Evaluación del Modelo')
            print(f'{"="*50}')
            print(f'Total de muestras: {total}')
            print(f'Predicciones correctas: {correct}')
            print(f'Precisión: {accuracy:.2f}%')
            print(f'{"="*50}\n')
        
        return {
            'accuracy': accuracy,
            'total': total,
            'correct': correct,
            'true_labels': all_labels,
            'predicted_labels': all_predicted
        }
    
    def visualize_predictions(
        self,
        data_loader,
        num_samples: int = 9,
        save_path: Optional[str] = None
    ):
        """
        Visualiza predicciones del modelo en una cuadrícula.
        
        Args:
            data_loader: DataLoader con las imágenes.
            num_samples: Número de muestras a visualizar.
            save_path: Ruta opcional para guardar la imagen.
        """
        self.model.eval()
        
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        
        # Seleccionar solo num_samples muestras
        images = images[:num_samples]
        labels = labels[:num_samples]
        
        # Hacer predicciones
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
            _, predicted = torch.max(outputs, 1)
        
        # Crear visualización
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for i in range(num_samples):
            img = images[i].squeeze().cpu().numpy()
            pred_char = self.mapping[predicted[i].item()]
            true_char = self.mapping[labels[i].item()]
            
            axes[i].imshow(img, cmap='gray')
            
            # Color verde si es correcto, rojo si es incorrecto
            color = 'green' if pred_char == true_char else 'red'
            axes[i].set_title(f"Real: {true_char}\nPred: {pred_char}",
                            fontsize=10, color=color, fontweight='bold')
            axes[i].axis('off')
        
        # Ocultar ejes sobrantes
        for j in range(num_samples, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle('Predicciones del Modelo OCR', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def predict_single_char(self, image_path: str, show_image: bool = True) -> str:
        """
        Predice el carácter en una imagen de un solo carácter.
        
        Args:
            image_path: Ruta a la imagen del carácter.
            show_image: Si True, muestra la imagen con la predicción.
        
        Returns:
            Carácter predicho.
        """
        # Cargar imagen
        image = Image.open(image_path)
        
        # Aplicar transformaciones
        image_tensor = self.transform(image)
        
        # Hacer predicción
        with torch.no_grad():
            output = self.model(image_tensor.unsqueeze(0).to(self.device))
            _, predicted = torch.max(output, 1)
        
        pred_char = self.mapping[predicted.item()]
        
        # Visualizar si se solicita
        if show_image:
            plt.figure(figsize=(4, 4))
            plt.imshow(image, cmap='gray')
            plt.title(f"Predicción: {pred_char}", fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return pred_char
    
    def predict_letter_sequence(
        self,
        letter_images: List[np.ndarray],
        show_visualization: bool = False,
        word_index: int = 0
    ) -> List[str]:
        """
        Predice una secuencia de letras a partir de imágenes individuales.
        
        Args:
            letter_images: Lista de imágenes de letras (numpy arrays).
            show_visualization: Si True, muestra las letras con predicciones.
            word_index: Índice de la palabra (para visualización).
        
        Returns:
            Lista de caracteres predichos.
        """
        predictions = []
        
        if show_visualization:
            n = len(letter_images)
            fig, axes = plt.subplots(1, n, figsize=(2*n, 3))
            if n == 1:
                axes = [axes]
        
        for idx, letter_img in enumerate(letter_images):
            # Preprocesar letra
            if isinstance(letter_img, np.ndarray) and len(letter_img.shape) > 0:
                # Redimensionar y convertir
                letter_img = resize_with_padding(letter_img, (28, 28))
                
                if len(letter_img.shape) == 3:
                    letter_img = convert_to_grayscale(letter_img)
                
                # Convertir a PIL y aplicar transformaciones
                pil_img = Image.fromarray(letter_img)
                transformed_tensor = self.transform(pil_img)
                
                # Predecir
                with torch.no_grad():
                    output = self.model(transformed_tensor.unsqueeze(0).to(self.device))
                    _, predicted = torch.max(output, 1)
                
                pred_char = self.mapping[predicted.item()]
                predictions.append(pred_char)
                
                # Visualizar si se solicita
                if show_visualization:
                    if len(letter_img.shape) == 3:
                        axes[idx].imshow(cv2.cvtColor(letter_img, cv2.COLOR_BGR2RGB))
                    else:
                        axes[idx].imshow(letter_img, cmap='gray')
                    axes[idx].set_title(f"L{idx+1}: {pred_char}", fontsize=10)
                    axes[idx].axis('off')
        
        if show_visualization:
            plt.suptitle(f"Palabra {word_index + 1}: {''.join(predictions)}", 
                        fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.show()
        
        return predictions
    
    def save_model(self, path: str):
        """
        Guarda el modelo wrapper completo.
        
        Args:
            path: Ruta donde guardar el modelo.
        """
        self.model.save_checkpoint(path)
    
    def load_model(self, path: str):
        """
        Carga un modelo previamente guardado.
        
        Args:
            path: Ruta del modelo a cargar.
        """
        self.model.load_checkpoint(path)