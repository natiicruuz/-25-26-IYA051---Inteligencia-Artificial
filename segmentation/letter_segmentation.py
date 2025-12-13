"""
Segmentación de letras dentro de palabras.

Este módulo contiene la clase LetterSegmenter que divide imágenes de palabras
en imágenes individuales de letras usando detección de contornos.
"""

import cv2
import numpy as np
from typing import List, Tuple


class LetterSegmenter:
    """
    Segmentador de letras dentro de palabras.
    
    Detecta y extrae caracteres individuales de imágenes de palabras.
    """
    
    def __init__(
        self,
        min_size: int = 5,
        margin: int = 2,
        morph_kernel_size: int = 3
    ):
        """
        Inicializa el segmentador de letras.
        
        Args:
            min_size: Tamaño mínimo (ancho y alto) para considerar un contorno válido.
            margin: Margen adicional alrededor de cada letra extraída.
            morph_kernel_size: Tamaño del kernel para operación morfológica de cierre.
        """
        self.min_size = min_size
        self.margin = margin
        self.morph_kernel_size = morph_kernel_size
    
    def segment_word(self, word_image: np.ndarray) -> List[np.ndarray]:
        """
        Segmenta una imagen de palabra en letras individuales.
        
        Args:
            word_image: Imagen de la palabra (numpy array).
        
        Returns:
            Lista de imágenes de letras ordenadas de izquierda a derecha.
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(word_image, cv2.COLOR_BGR2GRAY)
        
        # Binarizar (invertida)
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Operación morfológica de cierre para conectar partes de letras
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos de letras
        boxes = self._find_letter_contours(binary)
        
        # Ordenar de izquierda a derecha
        boxes = sorted(boxes, key=lambda b: b[0])
        
        # Extraer letras
        letters = self._extract_letters(word_image, boxes)
        
        return letters
    
    def segment_multiple_words(
        self,
        word_images: List[np.ndarray]
    ) -> List[List[np.ndarray]]:
        """
        Segmenta múltiples palabras en letras.
        
        Args:
            word_images: Lista de imágenes de palabras.
        
        Returns:
            Lista de listas, donde cada sublista contiene las letras de una palabra.
        """
        all_letters = []
        
        for word_img in word_images:
            letters = self.segment_word(word_img)
            all_letters.append(letters)
        
        return all_letters
    
    def _find_letter_contours(self, binary_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Encuentra contornos de letras en la imagen binarizada.
        
        Args:
            binary_image: Imagen binarizada de la palabra.
        
        Returns:
            Lista de bounding boxes (x, y, w, h) de las letras.
        """
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filtrar contornos por tamaño
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > self.min_size and h > self.min_size:
                boxes.append((x, y, w, h))
        
        return boxes
    
    def _extract_letters(
        self,
        word_image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[np.ndarray]:
        """
        Extrae imágenes de letras a partir de los bounding boxes.
        
        Args:
            word_image: Imagen de la palabra.
            boxes: Lista de bounding boxes (x, y, w, h).
        
        Returns:
            Lista de imágenes de letras.
        """
        letters = []
        height, width = word_image.shape[:2]
        
        for x, y, w, h in boxes:
            # Añadir margen con límites seguros
            y1 = max(0, y - self.margin)
            y2 = min(height, y + h + self.margin)
            x1 = max(0, x - self.margin)
            x2 = min(width, x + w + self.margin)
            
            # Extraer letra
            letter_img = word_image[y1:y2, x1:x2]
            letters.append(letter_img)
        
        return letters
    
    def visualize_segmentation(
        self,
        word_image: np.ndarray,
        word_index: int = 0,
        save_path: str = None
    ):
        """
        Visualiza la segmentación de una palabra en letras.
        
        Args:
            word_image: Imagen de la palabra.
            word_index: Índice de la palabra (para el título).
            save_path: Ruta opcional para guardar la visualización.
        """
        import matplotlib.pyplot as plt
        
        # Segmentar palabra
        letters = self.segment_word(word_image)
        
        if not letters:
            print("No se detectaron letras en la palabra")
            return
        
        # Crear visualización
        n = len(letters)
        fig, axes = plt.subplots(1, n, figsize=(2*n, 3))
        
        if n == 1:
            axes = [axes]
        
        for i, letter_img in enumerate(letters):
            if len(letter_img.shape) == 3:
                axes[i].imshow(cv2.cvtColor(letter_img, cv2.COLOR_BGR2RGB))
            else:
                axes[i].imshow(letter_img, cmap='gray')
            axes[i].set_title(f"L{i+1}", fontsize=10)
            axes[i].axis('off')
        
        plt.suptitle(f"Palabra {word_index + 1} - {n} letras detectadas", 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_multiple_words(
        self,
        word_images: List[np.ndarray],
        save_dir: str = None
    ):
        """
        Visualiza la segmentación de múltiples palabras.
        
        Args:
            word_images: Lista de imágenes de palabras.
            save_dir: Directorio opcional donde guardar las visualizaciones.
        """
        for idx, word_img in enumerate(word_images):
            save_path = None
            if save_dir:
                save_path = f"{save_dir}/word_{idx+1}_letters.png"
            
            self.visualize_segmentation(word_img, word_index=idx, save_path=save_path)