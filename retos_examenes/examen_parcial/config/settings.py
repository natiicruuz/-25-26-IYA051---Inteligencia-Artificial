"""
Configuración global del proyecto de reconocimiento de cartas.
Centraliza todas las constantes para facilitar ajustes.

TÉCNICAS UTILIZADAS:
    ✅ Visión artificial clásica
    ✅ Template matching (correlación normalizada)
    ✅ Segmentación por color HSV
    ✅ Operaciones morfológicas
    ❌ SIN Machine Learning
    ❌ SIN Redes Neuronales
"""

import numpy as np
import os

RTSP_URL = 'rtsp://172.20.10.8:8080/h264.sdp'


# PARÁMETROS DE VISIÓN ARTIFICIAL


# Dimensiones normalizadas de la carta (en píxeles)
CARD_WIDTH = 200
CARD_HEIGHT = 300

# Región de Interés (ROI) para el valor y palo en la esquina superior izquierda
# Formato: (x_inicio, y_inicio, ancho, alto)
ROI_CORNER_VALUE = (0, 3, 85, 50)  # más ancho para capturar "10" completo
ROI_CORNER_SUIT = (5, 50, 40, 40)  # Sin cambios


# CALIBRACIÓN DE COLORES HSV (RESULTADO DE FASE 2)

# Estos valores deben ser actualizados después de ejecutar 1_calibrar_hsv.py
# IMPORTANTE: Estos valores son para DETECTAR EL TAPETE (fondo verde)

# VALORES ACTUALIZADOS DESDE TU CALIBRACIÓN:
LOWER_COLOR_FONDO = np.array([35, 153, 0])
UPPER_COLOR_FONDO = np.array([105, 255, 255])

# Valores HSV para detectar ROJOS (corazones y diamantes)
LOWER_RED_HSV_1 = np.array([0, 30, 30])    # Rojo en rango bajo
UPPER_RED_HSV_1 = np.array([15, 255, 255])
LOWER_RED_HSV_2 = np.array([150, 30, 30])  # Menos saturación
UPPER_RED_HSV_2 = np.array([179, 255, 255])


# PROCESAMIENTO DE IMAGEN


# Parámetros de GaussianBlur
BLUR_KERNEL_SIZE = (5, 5)
BLUR_SIGMA = 0  # 0 = calculado automáticamente

# Parámetros de umbralización
THRESHOLD_VALUE = 150      # Valor para separar carta del fondo
THRESHOLD_MAX = 255        # Valor máximo en imagen binaria

# Filtros de contornos
MIN_CONTOUR_AREA = 5000    # Área mínima para considerar una carta válida
EPSILON_FACTOR = 0.03      # Factor para aproximación poligonal (3% del perímetro)


# TEMPLATE MATCHING (TÉCNICA PRINCIPAL DE CLASIFICACIÓN)


# Dimensiones de los templates
TEMPLATE_VALUE_SIZE = (30, 50)   # Ancho x Alto para valores (AS, 2-10, J, Q, K)
TEMPLATE_SUIT_SIZE = (40, 40)    # Ancho x Alto para símbolos de palos

# Método de matching de OpenCV
# Opciones: cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED
TEMPLATE_MATCHING_METHOD = 'TM_CCOEFF_NORMED'

# Umbral mínimo para considerar un match válido (0.0 a 1.0)
TEMPLATE_MATCH_THRESHOLD = 0.35


# RUTAS Y DIRECTORIOS


# Directorio base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directorios de datos
DATA_DIR = os.path.join(BASE_DIR, 'data')
REFERENCE_DIR = os.path.join(DATA_DIR, 'imagenes_referencia')
TEMPLATES_DIR = os.path.join(DATA_DIR, 'templates')
TEMPLATES_VALUES_DIR = os.path.join(TEMPLATES_DIR, 'valores')
TEMPLATES_SUITS_DIR = os.path.join(TEMPLATES_DIR, 'palos')

# Crear directorios si no existen
for directory in [DATA_DIR, REFERENCE_DIR, TEMPLATES_DIR, 
                TEMPLATES_VALUES_DIR, TEMPLATES_SUITS_DIR]:
    os.makedirs(directory, exist_ok=True)


# DEFINICIÓN DE CARTAS


# Valores de las cartas (en orden)
CARD_VALUES = ['AS', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

# Palos de las cartas
CARD_SUITS = ['PICAS', 'CORAZONES', 'DIAMANTES', 'TREBOLES']

# Diccionario de colores por palo (para validación)
SUIT_COLORS = {
    'PICAS': 'negro',
    'CORAZONES': 'rojo',
    'DIAMANTES': 'rojo',
    'TREBOLES': 'negro'
}

# Todas las cartas posibles (52)
ALL_CARDS = [f"{value}_{suit}" for value in CARD_VALUES for suit in CARD_SUITS]


# PARÁMETROS DE VISUALIZACIÓN


# Colores para dibujar (formato BGR)
COLOR_GREEN = (0, 255, 0)      # Verde para contornos válidos
COLOR_RED = (0, 0, 255)        # Rojo para contornos inválidos
COLOR_BLUE = (255, 0, 0)       # Azul para ROIs
COLOR_YELLOW = (0, 255, 255)   # Amarillo para texto
COLOR_WHITE = (255, 255, 255)  # Blanco para texto principal

# Parámetros de texto
FONT = 'FONT_HERSHEY_SIMPLEX'
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# Tamaño de ventanas de visualización
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600


# CONFIGURACIÓN DE CAPTURA


# Tiempo mínimo entre capturas (segundos)
CAPTURE_COOLDOWN = 0.5

# Número mínimo de capturas por carta recomendado
MIN_CAPTURES_PER_CARD = 4


# CONFIGURACIÓN DE LOGGING


# Nivel de logging: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_LEVEL = 'INFO'

# Archivo de log
LOG_FILE = os.path.join(BASE_DIR, 'card_recognition.log')


# FUNCIONES AUXILIARES


def get_rtsp_url():
    """Retorna la URL RTSP configurada."""
    return RTSP_URL

def get_card_label(value, suit):
    """
    Genera la etiqueta estándar de una carta.
    
    Args:
        value (str): Valor de la carta (AS, 2-10, J, Q, K)
        suit (str): Palo (PICAS, CORAZONES, DIAMANTES, TREBOLES)
    
    Returns:
        str: Etiqueta en formato 'VALOR_PALO'
    """
    return f"{value.upper()}_{suit.upper()}"

def is_valid_card_label(label):
    """
    Verifica si una etiqueta es válida.
    
    Args:
        label (str): Etiqueta a validar
    
    Returns:
        bool: True si es válida, False en caso contrario
    """
    return label in ALL_CARDS

def print_config_summary():
    """Imprime un resumen de la configuración actual."""
    print("=" * 60)
    print("CONFIGURACIÓN DEL SISTEMA - VISIÓN ARTIFICIAL CLÁSICA")
    print("=" * 60)
    print(f"RTSP URL: {RTSP_URL}")
    print(f"Dimensiones carta: {CARD_WIDTH}x{CARD_HEIGHT}")
    print(f"ROI Valor: {ROI_CORNER_VALUE} (x, y, w, h)")  
    print(f"ROI Palo: {ROI_CORNER_SUIT} (x, y, w, h)")
    print(f"HSV Fondo: {LOWER_COLOR_FONDO} - {UPPER_COLOR_FONDO}")
    print(f"\nTÉCNICA DE CLASIFICACIÓN:")
    print(f"  Método: Template Matching")
    print(f"  Algoritmo: {TEMPLATE_MATCHING_METHOD}")
    print(f"  Umbral: {TEMPLATE_MATCH_THRESHOLD}")
    print(f"  Template Value Size: {TEMPLATE_VALUE_SIZE}") 
    print(f"  Template Suit Size: {TEMPLATE_SUIT_SIZE}")
    print(f"\nDIRECTORIOS:")
    print(f"  Referencias: {REFERENCE_DIR}")
    print(f"  Templates: {TEMPLATES_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    print_config_summary()