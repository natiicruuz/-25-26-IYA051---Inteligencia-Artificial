"""
Utilidades para procesamiento y manipulación de imágenes.

Este módulo contiene funciones para redimensionar, normalizar y preparar
imágenes para el modelo OCR.
"""

import cv2
import numpy as np
from typing import Tuple, Union


def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_color: Union[int, Tuple[int, int, int]] = 255
) -> np.ndarray:
    """
    Redimensiona una imagen manteniendo su proporción y añade padding para 
    ajustarla al tamaño objetivo.
    
    Args:
        image: Imagen a redimensionar (numpy array).
        target_size: Tamaño objetivo (altura, ancho).
        pad_color: Color de relleno. 255 para blanco, 0 para negro.
                   Puede ser int (escala de grises) o tupla RGB.
    
    Returns:
        Imagen redimensionada con padding.
    
    Example:
        >>> img = cv2.imread('character.png')
        >>> resized = resize_with_padding(img, (28, 28), pad_color=255)
    """
    height, width = image.shape[:2]
    target_height, target_width = target_size
    
    # Determinar método de interpolación según si se agranda o reduce
    if height > target_height or width > target_width:
        interpolation = cv2.INTER_AREA  # Mejor para reducción
    else:
        interpolation = cv2.INTER_CUBIC  # Mejor para ampliación
    
    # Calcular relación de aspecto
    aspect_ratio = width / height
    
    # Calcular nuevas dimensiones y padding
    if aspect_ratio > 1:  # Imagen horizontal
        new_width = target_width
        new_height = int(np.round(new_width / aspect_ratio))
        pad_vertical = (target_height - new_height) / 2
        pad_top = int(np.floor(pad_vertical))
        pad_bottom = int(np.ceil(pad_vertical))
        pad_left, pad_right = 0, 0
        
    elif aspect_ratio < 1:  # Imagen vertical
        new_height = target_height
        new_width = int(np.round(new_height * aspect_ratio))
        pad_horizontal = (target_width - new_width) / 2
        pad_left = int(np.floor(pad_horizontal))
        pad_right = int(np.ceil(pad_horizontal))
        pad_top, pad_bottom = 0, 0
        
    else:  # Imagen cuadrada
        new_height, new_width = target_height, target_width
        pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
    
    # Convertir pad_color a lista si la imagen es a color
    if len(image.shape) == 3 and not isinstance(pad_color, (list, tuple, np.ndarray)):
        pad_color = [pad_color] * 3
    
    # Redimensionar imagen
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    
    # Añadir padding
    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color
    )
    
    return padded


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen a escala de grises si no lo está ya.
    
    Args:
        image: Imagen en formato numpy array.
    
    Returns:
        Imagen en escala de grises.
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def binarize_image(
    image: np.ndarray,
    threshold: int = 0,
    invert: bool = True
) -> np.ndarray:
    """
    Binariza una imagen usando el método de Otsu.
    
    Args:
        image: Imagen en escala de grises.
        threshold: Valor de umbral (0 para automático con Otsu).
        invert: Si True, invierte los colores (texto blanco sobre fondo negro).
    
    Returns:
        Imagen binarizada.
    """
    if invert:
        _, binary = cv2.threshold(
            image, threshold, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        _, binary = cv2.threshold(
            image, threshold, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    return binary


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normaliza los valores de píxeles de una imagen al rango [0, 1].
    
    Args:
        image: Imagen en formato numpy array.
    
    Returns:
        Imagen normalizada.
    """
    return image.astype(np.float32) / 255.0


def preprocess_character_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (28, 28)
) -> np.ndarray:
    """
    Pipeline completo de preprocesamiento para una imagen de carácter.
    
    Args:
        image: Imagen del carácter.
        target_size: Tamaño objetivo para redimensionar.
    
    Returns:
        Imagen preprocesada lista para el modelo.
    """
    # Convertir a escala de grises
    gray = convert_to_grayscale(image)
    
    # Redimensionar con padding
    resized = resize_with_padding(gray, target_size, pad_color=255)
    
    return resized


def add_margin(image: np.ndarray, margin: int = 2) -> np.ndarray:
    """
    Añade un margen alrededor de una imagen.
    
    Args:
        image: Imagen a la que añadir margen.
        margin: Tamaño del margen en píxeles.
    
    Returns:
        Imagen con margen añadido.
    """
    if len(image.shape) == 3:
        color = [255, 255, 255]
    else:
        color = 255
    
    return cv2.copyMakeBorder(
        image, margin, margin, margin, margin,
        cv2.BORDER_CONSTANT, value=color
    )