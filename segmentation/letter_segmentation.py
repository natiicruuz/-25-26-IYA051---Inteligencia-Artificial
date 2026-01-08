"""
Segmentación de letras con proyección vertical.

Versión definitiva con:
- Filtrado de trazos horizontales (líneas de 't', guiones)
- Detección automática de letras fusionadas
- Sensibilidad ajustable para manuscritos
"""

import cv2
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt


class LetterSegmenter:
    """
    Segmentador de letras usando proyección vertical.
    
    Incluye filtros inteligentes para eliminar trazos y dividir letras fusionadas.
    """
    
    def __init__(
        self,
        min_width: int = 10,
        min_height: int = 15,
        margin: int = 2,
        valley_threshold: float = 0.10,
        min_valley_width: int = 2,
        max_aspect_ratio: float = 3.0,  # ⭐ NUEVO: filtrar líneas horizontales
        split_wide_threshold: float = 2.5  # ⭐ NUEVO: dividir letras fusionadas
    ):
        """
        Inicializa el segmentador de letras.
        
        Args:
            min_width: Ancho mínimo de letra válida (píxeles).
            min_height: Alto mínimo de letra válida (píxeles).
            margin: Margen adicional alrededor de cada letra.
            valley_threshold: Umbral para detectar valles (0-1).
            min_valley_width: Ancho mínimo de valle válido (píxeles).
            max_aspect_ratio: Relación ancho/alto máxima (filtrar líneas horizontales).
            split_wide_threshold: Si ancho > threshold * alto, intentar dividir.
        """
        self.min_width = min_width
        self.min_height = min_height
        self.margin = margin
        self.valley_threshold = valley_threshold
        self.min_valley_width = min_valley_width
        self.max_aspect_ratio = max_aspect_ratio
        self.split_wide_threshold = split_wide_threshold
    
    def segment_word(self, word_image: np.ndarray, debug: bool = False) -> List[np.ndarray]:
        """
        Segmenta palabra en letras usando proyección vertical.
        
        Args:
            word_image: Imagen de la palabra (numpy array).
            debug: Si True, muestra visualización del proceso.
        
        Returns:
            Lista de imágenes de letras ordenadas de izquierda a derecha.
        """
        # Paso 1: Preprocesar
        gray, binary = self._preprocess(word_image)
        
        # Paso 2: Calcular proyección vertical
        vertical_projection = self._calculate_vertical_projection(binary)
        
        # Paso 3: Encontrar puntos de corte
        split_points = self._find_split_points_improved(
            vertical_projection,
            threshold=self.valley_threshold,
            min_valley_width=self.min_valley_width
        )
        
        # Paso 4: Extraer segmentos con recorte vertical
        segments = self._extract_segments_improved(word_image, binary, split_points)
        
        # Paso 5: Filtrar y procesar segmentos
        valid_letters = []
        
        for i, seg in enumerate(segments):
            h, w = seg.shape[:2]
            
            # Filtrar segmentos muy pequeños (ruido)
            if w < self.min_width or h < self.min_height:
                if debug:
                    print(f"  ⚠️  Segmento {i+1}: {w}x{h} muy pequeño (filtrado)")
                continue
            
            # Filtrar segmentos que son toda la imagen (error de recorte)
            if h > word_image.shape[0] * 0.9:
                if debug:
                    print(f"  ⚠️  Segmento {i+1}: {w}x{h} demasiado alto (filtrado)")
                continue
            
            # ⭐ NUEVO: Filtrar trazos horizontales (líneas de 't', guiones)
            aspect_ratio = w / h if h > 0 else 0
            
            # Caso 1: Trazo horizontal (línea de 't', guion) → FILTRAR
            #   Características: ancho >> alto Y altura pequeña
            if aspect_ratio > self.max_aspect_ratio and h < 50:
                if debug:
                    print(f"  ⚠️  Segmento {i+1}: {w}x{h} (ratio {aspect_ratio:.1f}) → trazo horizontal (filtrado)")
                continue
            
            # Caso 2: Letras fusionadas (dos o más letras juntas) → DIVIDIR
            #   Características: ancho > 2.5 * alto Y altura razonable
            if aspect_ratio > self.split_wide_threshold and h >= 50:
                if debug:
                    print(f"  🔍 Segmento {i+1}: {w}x{h} (ratio {aspect_ratio:.1f}) → letras fusionadas, intentando dividir...")
                
                # Intentar subdividir este segmento
                subsegments = self._split_wide_segment(seg, binary[:, :seg.shape[1]], debug=debug)
                
                if len(subsegments) > 1:
                    if debug:
                        print(f"     ✓ Dividido en {len(subsegments)} letras")
                    valid_letters.extend(subsegments)
                else:
                    if debug:
                        print(f"     ✗ No se pudo dividir, manteniendo original")
                    valid_letters.append(seg)
            
            # Caso 3: Letra normal → ACEPTAR
            else:
                valid_letters.append(seg)
        
        # Debug: Visualización
        if debug:
            self._visualize_segmentation(
                word_image, binary, vertical_projection, split_points, valid_letters
            )
        
        if len(valid_letters) == 0 and debug:
            print("  ❌ No se detectaron letras válidas")
            print(f"  Ajusta valley_threshold (actual: {self.valley_threshold:.2f})")
        
        return valid_letters
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesa imagen: escala de grises + binarización.
        
        Returns:
            Tupla (gray, binary_inverted)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Binarizar con Otsu (invertida: texto blanco, fondo negro)
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        return gray, binary
    
    def _calculate_vertical_projection(self, binary: np.ndarray) -> np.ndarray:
        """
        Calcula proyección vertical (suma de píxeles blancos por columna).
        
        Returns:
            Array normalizado [0-1] con la proyección.
        """
        projection = np.sum(binary, axis=0).astype(float)
        
        # Normalizar a [0, 1]
        if np.max(projection) > 0:
            projection = projection / np.max(projection)
        
        return projection
    
    def _find_split_points_improved(
        self,
        projection: np.ndarray,
        threshold: float = 0.10,
        min_valley_width: int = 2
    ) -> List[int]:
        """
        Encuentra puntos de corte usando detección de valles.
        
        Args:
            projection: Proyección vertical normalizada.
            threshold: Umbral para considerar un valle.
            min_valley_width: Ancho mínimo de valle válido.
        
        Returns:
            Lista de posiciones X donde cortar.
        """
        split_points = [0]  # Inicio
        
        in_valley = False
        valley_start = 0
        
        for x in range(len(projection)):
            if projection[x] < threshold:
                # Dentro de un valle
                if not in_valley:
                    valley_start = x
                    in_valley = True
            else:
                # Saliendo de un valle
                if in_valley:
                    valley_end = x
                    valley_width = valley_end - valley_start
                    
                    # Solo valles suficientemente anchos
                    if valley_width >= min_valley_width:
                        valley_mid = (valley_start + valley_end) // 2
                        split_points.append(valley_mid)
                    
                    in_valley = False
        
        split_points.append(len(projection))  # Final
        
        # Filtrar puntos muy cercanos
        filtered_points = [split_points[0]]
        for point in split_points[1:]:
            if point - filtered_points[-1] >= 8:  # Mínimo 8 píxeles de separación
                filtered_points.append(point)
        
        # Asegurar inicio y fin
        if len(filtered_points) < 2:
            filtered_points = [0, len(projection)]
        
        return filtered_points
    
    def _split_wide_segment(
        self,
        segment: np.ndarray,
        segment_binary: np.ndarray,
        debug: bool = False
    ) -> List[np.ndarray]:
        """
        Intenta dividir un segmento muy ancho en subsegmentos.
        
        Args:
            segment: Imagen del segmento ancho.
            segment_binary: Versión binarizada del segmento.
            debug: Si True, muestra información de debug.
        
        Returns:
            Lista de subsegmentos (o el original si no se puede dividir).
        """
        # Calcular proyección vertical del segmento
        projection = self._calculate_vertical_projection(segment_binary)
        
        # Buscar puntos de corte con MAYOR sensibilidad
        split_points = self._find_split_points_improved(
            projection,
            threshold=0.05,  # Más sensible (0.05 vs 0.10)
            min_valley_width=1  # Acepta valles más pequeños
        )
        
        # Si encontramos puntos de corte internos (más de 2), dividir
        if len(split_points) > 2:  # Más de [inicio, fin]
            subsegments = []
            for i in range(len(split_points) - 1):
                x1 = max(0, split_points[i])
                x2 = min(segment.shape[1], split_points[i + 1])
                
                if x2 - x1 < 5:
                    continue
                
                subseg = segment[:, x1:x2]
                subseg_binary = segment_binary[:, x1:x2]
                
                # Recortar verticalmente
                subseg_cropped = self._crop_vertical_whitespace_fixed(subseg, subseg_binary)
                
                if subseg_cropped.size > 0:
                    subsegments.append(subseg_cropped)
            
            return subsegments
        
        return [segment]  # No se pudo dividir
    
    def _extract_segments_improved(
        self,
        image: np.ndarray,
        binary: np.ndarray,
        split_points: List[int]
    ) -> List[np.ndarray]:
        """
        Extrae segmentos CON RECORTE VERTICAL.
        
        Args:
            image: Imagen original.
            binary: Imagen binarizada (para calcular recorte).
            split_points: Posiciones X donde cortar.
        
        Returns:
            Lista de segmentos recortados.
        """
        segments = []
        height, width = image.shape[:2]
        
        for i in range(len(split_points) - 1):
            x1 = max(0, split_points[i])
            x2 = min(width, split_points[i + 1])
            
            if x2 - x1 < 5:
                continue
            
            # Extraer segmento horizontal
            segment_img = image[:, x1:x2]
            segment_binary = binary[:, x1:x2]
            
            # Recorte vertical
            segment_cropped = self._crop_vertical_whitespace_fixed(
                segment_img, segment_binary
            )
            
            if segment_cropped.size > 0:
                segments.append(segment_cropped)
        
        return segments
    
    def _crop_vertical_whitespace_fixed(
        self,
        image: np.ndarray,
        binary: np.ndarray
    ) -> np.ndarray:
        """
        Recorta espacios verticales usando la imagen binarizada.
        
        Args:
            image: Imagen RGB/grayscale a recortar.
            binary: Imagen binarizada (texto blanco, fondo negro).
        
        Returns:
            Imagen recortada verticalmente.
        """
        # Proyección horizontal (suma por filas)
        horizontal_projection = np.sum(binary, axis=1)
        
        # Encontrar filas con contenido
        non_zero_rows = np.where(horizontal_projection > 0)[0]
        
        if len(non_zero_rows) == 0:
            return image
        
        # Límites con margen
        y1 = max(0, non_zero_rows[0] - self.margin)
        y2 = min(image.shape[0], non_zero_rows[-1] + 1 + self.margin)
        
        return image[y1:y2, :]
    
    def _visualize_segmentation(
        self,
        original: np.ndarray,
        binary: np.ndarray,
        projection: np.ndarray,
        split_points: List[int],
        letters: List[np.ndarray]
    ):
        """
        Visualiza el proceso de segmentación completo.
        """
        # Figura 1: Proceso de segmentación
        fig = plt.figure(figsize=(16, 10))
        
        # Plot 1: Original con líneas
        ax1 = plt.subplot(3, 1, 1)
        if len(original.shape) == 3:
            ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            ax1.imshow(original, cmap='gray')
        
        for x in split_points[1:-1]:
            ax1.axvline(x, color='red', linewidth=2, linestyle='--', alpha=0.7)
        
        ax1.set_title(
            f'1. Imagen Original con {len(split_points)-2} puntos de corte', 
            fontsize=12, fontweight='bold'
        )
        ax1.axis('off')
        
        # Plot 2: Binarizada
        ax2 = plt.subplot(3, 1, 2)
        ax2.imshow(binary, cmap='gray')
        
        for x in split_points[1:-1]:
            ax2.axvline(x, color='red', linewidth=2, linestyle='--', alpha=0.7)
        
        ax2.set_title(
            '2. Imagen Binarizada (texto blanco, fondo negro)', 
            fontsize=12, fontweight='bold'
        )
        ax2.axis('off')
        
        # Plot 3: Proyección
        ax3 = plt.subplot(3, 1, 3)
        ax3.fill_between(range(len(projection)), projection, alpha=0.5, color='blue')
        ax3.plot(projection, color='darkblue', linewidth=2)
        ax3.axhline(
            self.valley_threshold, color='orange', linestyle='--', 
            linewidth=2, label=f'Umbral ({self.valley_threshold:.2f})'
        )
        
        for x in split_points[1:-1]:
            ax3.axvline(x, color='red', linewidth=2, linestyle='--', alpha=0.7)
        
        ax3.set_title(
            '3. Proyección Vertical', 
            fontsize=12, fontweight='bold'
        )
        ax3.set_xlabel('Posición X (píxeles)', fontsize=10)
        ax3.set_ylabel('Densidad normalizada', fontsize=10)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.show()
        
        # Figura 2: Letras extraídas
        if len(letters) > 0:
            n = len(letters)
            fig2, axes = plt.subplots(1, n, figsize=(2.5*n, 4))
            
            if n == 1:
                axes = [axes]
            
            for i, letter_img in enumerate(letters):
                h, w = letter_img.shape[:2]
                aspect = w / h if h > 0 else 0
                
                if len(letter_img.shape) == 3:
                    axes[i].imshow(cv2.cvtColor(letter_img, cv2.COLOR_BGR2RGB))
                else:
                    axes[i].imshow(letter_img, cmap='gray')
                
                axes[i].set_title(
                    f"L{i+1}\n{w}x{h}px\nratio={aspect:.1f}", 
                    fontsize=9, fontweight='bold'
                )
                axes[i].axis('off')
            
            plt.suptitle(
                f'{n} letras extraídas (filtradas y procesadas)', 
                fontsize=14, fontweight='bold'
            )
            plt.tight_layout()
            plt.show()
    
    def segment_multiple_words(
        self,
        word_images: List[np.ndarray]
    ) -> List[List[np.ndarray]]:
        """Segmenta múltiples palabras."""
        all_letters = []
        for word_img in word_images:
            letters = self.segment_word(word_img)
            all_letters.append(letters)
        return all_letters