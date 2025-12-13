"""
Segmentación de palabras en documentos.

Este módulo contiene la clase WordSegmenter que detecta y extrae palabras
de imágenes de documentos usando procesamiento morfológico y detección de contornos.
"""

import cv2
import numpy as np
from typing import List, Tuple


class WordSegmenter:
		"""
		Segmentador de palabras en imágenes de documentos.
		
		Utiliza operaciones morfológicas y detección de contornos para identificar
		y extraer regiones de texto (palabras) de una imagen.
		"""
		
		def __init__(
				self,
				morph_kernel_size: Tuple[int, int] = (30, 5),
				min_word_width: int = 10,
				max_word_width: int = 500,
				min_word_height: int = 10,
				max_word_height: int = 100,
				row_threshold: int = 20
		):
				"""
				Inicializa el segmentador de palabras.
				
				Args:
						morph_kernel_size: Tamaño del kernel morfológico (ancho, alto).
						min_word_width: Ancho mínimo de una palabra válida.
						max_word_width: Ancho máximo de una palabra válida.
						min_word_height: Alto mínimo de una palabra válida.
						max_word_height: Alto máximo de una palabra válida.
						row_threshold: Umbral de distancia vertical para agrupar palabras en filas.
				"""
				self.morph_kernel_size = morph_kernel_size
				self.min_word_width = min_word_width
				self.max_word_width = max_word_width
				self.min_word_height = min_word_height
				self.max_word_height = max_word_height
				self.row_threshold = row_threshold
		
		def detect_words(self, image_path: str) -> List[np.ndarray]:
				"""
				Detecta y extrae palabras de una imagen de documento.
				
				Args:
						image_path: Ruta a la imagen del documento.
				
				Returns:
						Lista de imágenes (numpy arrays), una por cada palabra detectada,
						ordenadas en orden de lectura (arriba-abajo, izquierda-derecha).
				"""
				# Cargar imagen
				image = cv2.imread(image_path)
				
				if image is None:
						raise ValueError(f"No se pudo cargar la imagen: {image_path}")
				
				# Preprocesar imagen
				binary = self._preprocess_image(image)
				
				# Encontrar contornos de palabras
				contours = self._find_word_contours(binary)
				
				# Extraer bounding boxes
				bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
				
				# Ordenar en orden de lectura
				ordered_boxes = self._group_boxes_into_rows(bounding_boxes)
				
				# Extraer imágenes de palabras
				word_images = []
				for x, y, w, h in ordered_boxes:
						# Filtrar por tamaño
						if (self.min_word_width < w < self.max_word_width and
								self.min_word_height < h < self.max_word_height):
								word_img = image[y:y+h, x:x+w]
								word_images.append(word_img)
				
				return word_images
		
		def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
				"""
				Preprocesa la imagen para detección de palabras.
				
				Args:
						image: Imagen en formato BGR.
				
				Returns:
						Imagen binarizada y procesada morfológicamente.
				"""
				# Convertir a escala de grises
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				
				# Binarización con Otsu (invertida: texto blanco, fondo negro)
				_, binary = cv2.threshold(
						gray, 0, 255,
						cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
				)
				
				# Crear kernel morfológico
				kernel = cv2.getStructuringElement(
						cv2.MORPH_RECT,
						self.morph_kernel_size
				)
				
				# Operación de cierre para conectar letras de palabras
				morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
				
				return morphed
		
		def _find_word_contours(self, binary_image: np.ndarray) -> List:
				"""
				Encuentra contornos de palabras en la imagen binarizada.
				
				Args:
						binary_image: Imagen binarizada.
				
				Returns:
						Lista de contornos detectados.
				"""
				contours, _ = cv2.findContours(
						binary_image,
						cv2.RETR_EXTERNAL,
						cv2.CHAIN_APPROX_SIMPLE
				)
				return contours
		
		def _group_boxes_into_rows(
				self,
				bounding_boxes: List[Tuple[int, int, int, int]]
		) -> List[Tuple[int, int, int, int]]:
				"""
				Agrupa los bounding boxes en filas y los ordena en orden de lectura.
				
				Args:
						bounding_boxes: Lista de bounding boxes (x, y, w, h).
				
				Returns:
						Lista de bounding boxes ordenados por fila y posición horizontal.
				"""
				if not bounding_boxes:
						return []
				
				# Ordenar inicialmente por posición vertical
				sorted_boxes = sorted(bounding_boxes, key=lambda b: (b[1], b[0]))
				
				# Agrupar en filas
				rows = []
				current_row = [sorted_boxes[0]]
				
				for box in sorted_boxes[1:]:
						# Si la diferencia vertical es pequeña, pertenece a la misma fila
						if abs(box[1] - current_row[-1][1]) < self.row_threshold:
								current_row.append(box)
						else:
								# Nueva fila: ordenar la actual por posición horizontal
								rows.append(sorted(current_row, key=lambda b: b[0]))
								current_row = [box]
				
				# Añadir la última fila
				if current_row:
						rows.append(sorted(current_row, key=lambda b: b[0]))
				
				# Aplanar la lista de filas
				ordered_boxes = [box for row in rows for box in row]
				
				return ordered_boxes
		
		def visualize_detections(self, image_path: str, save_path: str = None):
				"""
				Visualiza las palabras detectadas con bounding boxes.
				
				Args:
						image_path: Ruta a la imagen del documento.
						save_path: Ruta opcional para guardar la visualización.
				"""
				import matplotlib.pyplot as plt
				
				# Cargar imagen
				image = cv2.imread(image_path)
				image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				
				# Detectar palabras
				binary = self._preprocess_image(image)
				contours = self._find_word_contours(binary)
				bounding_boxes = [cv2.boundingRect(c) for c in contours]
				ordered_boxes = self._group_boxes_into_rows(bounding_boxes)
				
				# Dibujar bounding boxes
				for idx, (x, y, w, h) in enumerate(ordered_boxes):
						if (self.min_word_width < w < self.max_word_width and
								self.min_word_height < h < self.max_word_height):
								cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
								cv2.putText(
										image_rgb, str(idx+1), (x, y-5),
										cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
								)
				
				# Visualizar
				plt.figure(figsize=(15, 10))
				plt.imshow(image_rgb)
				plt.title('Palabras Detectadas', fontsize=14, fontweight='bold')
				plt.axis('off')
				plt.tight_layout()
				
				if save_path:
						plt.savefig(save_path, dpi=300, bbox_inches='tight')
				
				plt.show()