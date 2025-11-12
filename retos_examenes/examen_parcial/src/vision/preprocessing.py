"""
Módulo de preprocesamiento de imágenes para detección de cartas.

Contiene funciones para:
    - Segmentación de cartas del fondo
    - Corrección de perspectiva
    - Normalización de dimensiones
    - Detección de múltiples cartas
"""

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import (
    CARD_WIDTH, CARD_HEIGHT,
    LOWER_COLOR_FONDO, UPPER_COLOR_FONDO,
    BLUR_KERNEL_SIZE, BLUR_SIGMA,
    MIN_CONTOUR_AREA, EPSILON_FACTOR
)

def order_points(pts):
    """
    Ordena los 4 puntos de un contorno rectangular.
    
    Args:
        pts (np.ndarray): Array de 4 puntos (puede ser shape (4,1,2) o (4,2))
    
    Returns:
        np.ndarray: Array (4,2) con puntos ordenados:
            [0] = superior-izquierda
            [1] = superior-derecha
            [2] = inferior-derecha
            [3] = inferior-izquierda
    
    Método:
        - Suma mínima → esquina superior-izquierda
        - Suma máxima → esquina inferior-derecha
        - Diferencia mínima (y-x) → esquina superior-derecha
        - Diferencia máxima (y-x) → esquina inferior-izquierda
    """
    # Reshape a (4, 2) si es necesario
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    # La suma de coordenadas identifica esquinas opuestas
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left (mínima suma)
    rect[2] = pts[np.argmax(s)]  # Bottom-Right (máxima suma)
    
    # Diferencia de coordenadas
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-Right (mínima diferencia)
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left (máxima diferencia)
    
    return rect

def preprocess_and_warp(frame, debug=False):
    """
    Segmenta una carta del fondo y corrige su perspectiva.
    
    Args:
        frame (np.ndarray): Frame BGR de la cámara
        debug (bool): Si True, retorna imágenes intermedias para debugging
    
    Returns:
        tuple: (warped_card, largest_contour, debug_images)
            - warped_card: Imagen de la carta normalizada (CARD_WIDTH x CARD_HEIGHT) o None
            - largest_contour: Contorno más grande detectado o None
            - debug_images: Dict con imágenes intermedias (solo si debug=True, sino dict vacío)
    
    Pipeline:
        1. Conversión a HSV
        2. Blur gaussiano para reducir ruido
        3. Máscara del fondo (tapete)
        4. Inversión de máscara para aislar cartas
        5. Detección de contornos
        6. Aproximación poligonal (buscar cuadrilátero)
        7. Transformación de perspectiva
    """
    debug_images = {}
    
    # 1. Conversión a HSV y Blur
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, BLUR_KERNEL_SIZE, BLUR_SIGMA)
    
    # 2. Segmentación por color
    mask_fondo = cv2.inRange(blurred, LOWER_COLOR_FONDO, UPPER_COLOR_FONDO)
    mask_carta = cv2.bitwise_not(mask_fondo)
    
    # 3. Operaciones morfológicas
    kernel = np.ones((3, 3), np.uint8)
    mask_carta = cv2.morphologyEx(mask_carta, cv2.MORPH_CLOSE, kernel)
    mask_carta = cv2.morphologyEx(mask_carta, cv2.MORPH_OPEN, kernel)
    
    # 4. Detección de contornos
    contours, _ = cv2.findContours(mask_carta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, {}
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Filtrar por área mínima
    if cv2.contourArea(largest_contour) < MIN_CONTOUR_AREA:
        return None, largest_contour, {}
    
    # 5. Aproximación poligonal
    epsilon = EPSILON_FACTOR * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # 6. Transformación de perspectiva
    if len(approx) == 4:
        points = order_points(approx)
        
        pts1 = np.float32(points)
        pts2 = np.float32([
            [0, 0],
            [CARD_WIDTH, 0],
            [CARD_WIDTH, CARD_HEIGHT],
            [0, CARD_HEIGHT]
        ])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(frame, M, (CARD_WIDTH, CARD_HEIGHT))
        
        return warped, largest_contour, {}
    
    return None, largest_contour, {}
def detect_multiple_cards(frame, debug=False):
    """
    Detecta y normaliza MÚLTIPLES cartas en el mismo frame.
    
    Args:
        frame (np.ndarray): Frame BGR de la cámara
        debug (bool): Si True, retorna info adicional
    
    Returns:
        list: Lista de tuplas (warped_card, contour, center)
            - warped_card: Imagen normalizada de la carta
            - contour: Contorno de la carta en el frame original
            - center: Coordenadas (x, y) del centro del contorno
    
    Diferencia con preprocess_and_warp():
        - Esta función procesa TODOS los contornos válidos
        - preprocess_and_warp() solo procesa el más grande
    """
    cards = []
    
    # 1. Pre-procesamiento (igual que en preprocess_and_warp)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, BLUR_KERNEL_SIZE, BLUR_SIGMA)
    
    # 2. Segmentación
    mask_fondo = cv2.inRange(blurred, LOWER_COLOR_FONDO, UPPER_COLOR_FONDO)
    mask_carta = cv2.bitwise_not(mask_fondo)
    
    # 3. Limpieza morfológica
    kernel = np.ones((3, 3), np.uint8)
    mask_carta = cv2.morphologyEx(mask_carta, cv2.MORPH_CLOSE, kernel)
    mask_carta = cv2.morphologyEx(mask_carta, cv2.MORPH_OPEN, kernel)
    
    # 4. Detección de contornos
    contours, _ = cv2.findContours(mask_carta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return cards
    
    # 5. Procesar cada contorno válido
    for contour in contours:
        # Filtrar por área
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue
        
        # Aproximación poligonal
        epsilon = 0.02 * cv2.arcLength(contour, True) 
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Solo procesar si es un cuadrilátero
        if len(approx) == 4:
            points = order_points(approx)
            
            # Transformación de perspectiva
            pts1 = np.float32(points)
            pts2 = np.float32([
                [0, 0],
                [CARD_WIDTH, 0],
                [CARD_WIDTH, CARD_HEIGHT],
                [0, CARD_HEIGHT]
            ])
            
            M = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(frame, M, (CARD_WIDTH, CARD_HEIGHT))
            
            # Calcular centro del contorno para etiquetar
            M_contour = cv2.moments(contour)
            if M_contour["m00"] != 0:
                cx = int(M_contour["m10"] / M_contour["m00"])
                cy = int(M_contour["m01"] / M_contour["m00"])
            else:
                cx, cy = 0, 0
            
            cards.append((warped, contour, (cx, cy)))
    
    return cards


def visualize_detection(frame, contours, predictions=None, valid_contours=True):
    """
    Dibuja los contornos detectados en el frame original.
    
    Args:
        frame (np.ndarray): Frame original BGR
        contours (list): Lista de contornos o lista de tuplas (warped, contour, center)
        predictions (list): Lista de predicciones (strings) correspondientes a cada carta
        valid_contours (bool): True = verde (válido), False = rojo (inválido)
    
    Returns:
        np.ndarray: Frame con visualización dibujada
    """
    output = frame.copy()
    color = (0, 255, 0) if valid_contours else (0, 0, 255)
    
    # Si contours es una lista de tuplas (de detect_multiple_cards)
    if contours and isinstance(contours[0], tuple):
        for i, (warped, contour, center) in enumerate(contours):
            cv2.drawContours(output, [contour], -1, color, 2)
            
            # Dibujar predicción si existe
            if predictions and i < len(predictions):
                cx, cy = center
                cv2.putText(output, predictions[i], (cx - 50, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        # Lista simple de contornos
        for contour in contours:
            cv2.drawContours(output, [contour], -1, color, 2)
    
    return output


def extract_roi_region(warped_card, roi_coords):
    """
    Extrae una Región de Interés (ROI) de la carta normalizada.
    
    Args:
        warped_card (np.ndarray): Imagen de carta normalizada
        roi_coords (tuple): (x, y, width, height)
    
    Returns:
        np.ndarray: Región recortada
    """
    x, y, w, h = roi_coords
    roi = warped_card[y:y+h, x:x+w]
    return roi


def is_red_card(roi):
    """
    Determina si el ROI contiene un símbolo rojo (Corazones o Diamantes).
    
    Returns:
        bool: True si es rojo, False si es negro
    
    Método:
        Analiza diferencias absolutas entre canales BGR.
    """
    # Calcular promedio de cada canal BGR
    mean_b = np.mean(roi[:,:,0])  # Azul
    mean_g = np.mean(roi[:,:,1])  # Verde
    mean_r = np.mean(roi[:,:,2])  # Rojo
    
    # Calcular diferencias
    diff_r_g = mean_r - mean_g
    diff_r_b = mean_r - mean_b
    
    es_rojo = (diff_r_g > 23) and (diff_r_b > 33)
    
    return es_rojo
def binarize_roi(roi, threshold=150):
    """
    Convierte ROI a imagen binaria para template matching.
    
    Args:
        roi (np.ndarray): Región de interés BGR o Grayscale
        threshold (int): Umbral de binarización (0-255)
    
    Returns:
        np.ndarray: Imagen binaria (0 o 255)
    
    Proceso:
        1. Conversión a escala de grises si es necesario
        2. Umbralización inversa (símbolos oscuros quedan blancos)
    """
    # Convertir a grayscale si es necesario
    if len(roi.shape) == 3:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi.copy()
    
    # Umbralización inversa (símbolos oscuros → blancos)
    _, thresh = cv2.threshold(roi_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    return thresh


def get_card_orientation(warped_card):
    """
    Determina si una carta está al derecho o invertida (180°).
    
    Args:
        warped_card (np.ndarray): Carta normalizada
    
    Returns:
        str: 'normal' o 'inverted'
    
    Método:
        Compara la densidad de píxeles oscuros en las esquinas superior e inferior.
        La esquina con el valor/palo tendrá mayor densidad.
    """
    from config.settings import ROI_CORNER_VALUE
    
    # Extraer esquina superior
    x, y, w, h = ROI_CORNER_VALUE
    top_roi = warped_card[y:y+h, x:x+w]
    
    # Extraer esquina inferior (invertida)
    bottom_roi = warped_card[CARD_HEIGHT-h-y:CARD_HEIGHT-y, CARD_WIDTH-w-x:CARD_WIDTH-x]
    
    # Convertir a escala de grises y contar píxeles oscuros
    top_gray = cv2.cvtColor(top_roi, cv2.COLOR_BGR2GRAY)
    bottom_gray = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2GRAY)
    
    top_dark_pixels = np.sum(top_gray < 100)
    bottom_dark_pixels = np.sum(bottom_gray < 100)
    
    # La esquina con más píxeles oscuros es donde está el símbolo
    if top_dark_pixels > bottom_dark_pixels * 1.2:
        return 'normal'
    else:
        return 'inverted'


def auto_rotate_card(warped_card):
    """
    Rota automáticamente la carta si está invertida.
    
    Args:
        warped_card (np.ndarray): Carta normalizada
    
    Returns:
        np.ndarray: Carta en orientación correcta
    """
    orientation = get_card_orientation(warped_card)
    
    if orientation == 'inverted':
        # Rotar 180 grados
        rotated = cv2.rotate(warped_card, cv2.ROTATE_180)
        return rotated
    
    return warped_card


# ============================================================================
# FUNCIONES DE DEBUGGING Y VISUALIZACIÓN
# ============================================================================

def show_processing_steps(frame, step_name="Processing"):
    """
    Muestra el frame actual en una ventana de debugging.
    
    Args:
        frame (np.ndarray): Imagen a mostrar
        step_name (str): Nombre de la etapa de procesamiento
    """
    cv2.imshow(f"DEBUG: {step_name}", frame)


def save_debug_images(debug_images, output_dir='debug_output'):
    """
    Guarda todas las imágenes de debugging en disco.
    
    Args:
        debug_images (dict): Diccionario {nombre: imagen}
        output_dir (str): Carpeta donde guardar las imágenes
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, img in debug_images.items():
        filepath = os.path.join(output_dir, f"{name}.jpg")
        cv2.imwrite(filepath, img)
        print(f"💾 Guardado: {filepath}")


# ============================================================================
# TESTS UNITARIOS
# ============================================================================

def test_order_points():
    """Test de la función order_points."""
    # Puntos desordenados de un rectángulo
    pts = np.array([[100, 150], [50, 50], [150, 50], [200, 150]])
    
    ordered = order_points(pts)
    
    # Verificar orden correcto
    assert ordered[0][0] < ordered[1][0], "Top-Left debe estar a la izquierda de Top-Right"
    assert ordered[0][1] < ordered[3][1], "Top-Left debe estar arriba de Bottom-Left"
    
    print("✅ test_order_points PASSED")


def test_is_red_card():
    """Test de detección de color rojo."""
    # Crear ROI sintético rojo
    roi_red = np.zeros((50, 50, 3), dtype=np.uint8)
    roi_red[:, :] = [0, 0, 255]  # BGR: Rojo puro
    
    # Crear ROI sintético negro
    roi_black = np.zeros((50, 50, 3), dtype=np.uint8)
    
    assert is_red_card(roi_red) == True, "Debe detectar rojo"
    assert is_red_card(roi_black) == False, "No debe detectar rojo en negro"
    
    print("✅ test_is_red_card PASSED")


if __name__ == "__main__":
    print("Ejecutando tests del módulo de preprocesamiento...")
    test_order_points()
    test_is_red_card()
    print("\n✅ Todos los tests pasaron correctamente")