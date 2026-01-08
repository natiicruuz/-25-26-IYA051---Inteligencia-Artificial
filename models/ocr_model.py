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
        Predice una secuencia de letras con preprocesamiento mejorado.
        
        Args:
            letter_images: Lista de imágenes de letras (numpy arrays).
            show_visualization: Si True, muestra las letras con predicciones.
            word_index: Índice de la palabra (para visualización).
        
        Returns:
            Lista de caracteres predichos.
        """
        predictions = []
        processed_images = []
        skipped_indices = []
        
        if show_visualization:
            n = len(letter_images)
            fig, axes = plt.subplots(2, n, figsize=(2*n, 6))
            if n == 1:
                axes = [[axes[0]], [axes[1]]]
            axes = np.array(axes).reshape(2, -1)
        
        for idx, letter_img in enumerate(letter_images):
            # ============== VALIDACIÓN BÁSICA ==============
            
            if not isinstance(letter_img, np.ndarray) or len(letter_img.shape) == 0:
                print(f"  ⚠️  Letra {idx+1}: Formato inválido, saltando...")
                skipped_indices.append(idx)
                continue
            
            h, w = letter_img.shape[:2]
            
            # SOLO filtrar ruido MUY obvio (punto pequeño < 10x10)
            if h < 10 and w < 10:
                print(f"  ⚠️  Letra {idx+1}: Muy pequeña ({w}x{h}), probablemente un punto. Saltando...")
                skipped_indices.append(idx)
                continue
            
            # Advertencia pero CONTINUAR procesando
            if h < 20 or w < 15:
                print(f"  ⚠️  Letra {idx+1}: Pequeña ({w}x{h}), puede ser 'i', 'l', '1' o ruido. Procesando de todos modos...")
            
            # ============== PREPROCESAMIENTO ==============
            
            # 1. Convertir a escala de grises
            if len(letter_img.shape) == 3:
                gray = cv2.cvtColor(letter_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = letter_img.copy()
            
            # 2. Binarizar con Otsu
            _, binary = cv2.threshold(
                gray, 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            # 3. Invertir si necesario (modelo espera fondo blanco, letra negra)
            mean_val = np.mean(binary)
            if mean_val < 127:
                binary = cv2.bitwise_not(binary)
            
            # 4. Redimensionar con padding (mantiene aspect ratio)
            from utils.image_utils import resize_with_padding
            resized = resize_with_padding(binary, (28, 28), pad_color=255)
            
            # 5. Suavizado ligero para reducir pixelación
            resized = cv2.GaussianBlur(resized, (3, 3), 0)
            
            # 6. Convertir a PIL y aplicar transforms
            from PIL import Image
            pil_img = Image.fromarray(resized)
            transformed_tensor = self.transform(pil_img)
            
            # ============== PREDICCIÓN ==============
            
            with torch.no_grad():
                output = self.model(transformed_tensor.unsqueeze(0).to(self.device))
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            pred_char = self.mapping[predicted.item()]
            conf_value = confidence.item()
            
            # Mostrar confianza
            if conf_value < 0.5:
                print(f"  ⚠️  Letra {idx+1}: Predicción '{pred_char}' con baja confianza ({conf_value:.2%})")
            
            predictions.append(pred_char)
            processed_images.append(resized)
            
            # ============== VISUALIZACIÓN ==============
            
            if show_visualization:
                # Fila 1: Imagen ORIGINAL
                if len(letter_img.shape) == 3:
                    axes[0, idx].imshow(cv2.cvtColor(letter_img, cv2.COLOR_BGR2RGB))
                else:
                    axes[0, idx].imshow(letter_img, cmap='gray')
                axes[0, idx].set_title(f"Original {idx+1}\n({w}x{h}px)", fontsize=8)
                axes[0, idx].axis('off')
                
                # Fila 2: Imagen PROCESADA + predicción
                axes[1, idx].imshow(resized, cmap='gray')
                
                # Color según confianza
                color = 'green' if conf_value > 0.7 else 'orange' if conf_value > 0.5 else 'red'
                axes[1, idx].set_title(
                    f"→ '{pred_char}'\n{conf_value:.0%}", 
                    fontsize=11, fontweight='bold', color=color
                )
                axes[1, idx].axis('off')
        
        if show_visualization:
            plt.suptitle(
                f"Palabra {word_index + 1}: {''.join(predictions)}\n"
                f"(Arriba: Original | Abajo: Procesada + Predicción)", 
                fontsize=14, fontweight='bold'
            )
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