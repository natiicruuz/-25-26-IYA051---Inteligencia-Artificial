"""
Módulo de Template Matching para reconocimiento de cartas.

Implementa técnicas clásicas de visión artificial:
    - Correlación cruzada normalizada (cv2.matchTemplate)
    - Matching multi-escala (para diferentes tamaños)
    - Scoring y ranking de templates

"""

import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import (
    CARD_VALUES, CARD_SUITS,
    TEMPLATES_VALUES_DIR, TEMPLATES_SUITS_DIR,
    TEMPLATE_VALUE_SIZE, TEMPLATE_SUIT_SIZE,
    TEMPLATE_MATCH_THRESHOLD
)


class TemplateLibrary:
    """
    Biblioteca de templates para matching de valores y palos.
    
    Carga y gestiona los templates guardados en disco.
    """
    
    def __init__(self):
        """Inicializa la biblioteca de templates."""
        self.value_templates = {}  # {valor: imagen_template}
        self.suit_templates = {}   # {palo: imagen_template}
        self.loaded = False
    
    def load_templates(self):
        """
        Carga todos los templates desde disco.
        
        Returns:
            bool: True si se cargaron correctamente, False en caso contrario
        """
        print("📚 Cargando biblioteca de templates...")
        
        # Cargar templates de valores
        for valor in CARD_VALUES:
            template_path = os.path.join(TEMPLATES_VALUES_DIR, f"{valor}.png")
            
            if not os.path.exists(template_path):
                print(f"⚠️  Template no encontrado: {template_path}")
                continue
            
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            
            if template is not None:
                # Redimensionar al tamaño estándar
                template = cv2.resize(template, TEMPLATE_VALUE_SIZE)
                self.value_templates[valor] = template
                print(f"   ✅ Cargado: {valor}")
            else:
                print(f"   ❌ Error al cargar: {valor}")
        
        # Cargar templates de palos
        for palo in CARD_SUITS:
            template_path = os.path.join(TEMPLATES_SUITS_DIR, f"{palo}.png")
            
            if not os.path.exists(template_path):
                print(f"⚠️  Template no encontrado: {template_path}")
                continue
            
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            
            if template is not None:
                # Redimensionar al tamaño estándar
                template = cv2.resize(template, TEMPLATE_SUIT_SIZE)
                self.suit_templates[palo] = template
                print(f"   ✅ Cargado: {palo}")
            else:
                print(f"   ❌ Error al cargar: {palo}")
        
        # Verificar que se cargaron templates
        if len(self.value_templates) > 0 and len(self.suit_templates) > 0:
            self.loaded = True
            print(f"\n✅ Templates cargados: {len(self.value_templates)} valores, {len(self.suit_templates)} palos")
            return True
        else:
            print("\n❌ No se cargaron suficientes templates")
            return False
    
    def get_value_template(self, valor):
        """Retorna el template de un valor específico."""
        return self.value_templates.get(valor)
    
    def get_suit_template(self, palo):
        """Retorna el template de un palo específico."""
        return self.suit_templates.get(palo)
    
    def is_loaded(self):
        """Verifica si los templates están cargados."""
        return self.loaded


# Instancia global de la biblioteca (singleton)
_template_library = None

def get_template_library():
    """
    Obtiene la instancia global de TemplateLibrary.
    
    Returns:
        TemplateLibrary: Instancia única de la biblioteca
    """
    global _template_library
    
    if _template_library is None:
        _template_library = TemplateLibrary()
        _template_library.load_templates()
    
    return _template_library


def match_template(roi, template, method=cv2.TM_CCOEFF_NORMED):
    """
    Realiza template matching en una ROI.
    
    Args:
        roi (np.ndarray): Región de interés (grayscale)
        template (np.ndarray): Template a buscar (grayscale)
        method: Método de matching de OpenCV
    
    Returns:
        tuple: (max_score, max_location)
            - max_score (float): Puntuación máxima (0.0 - 1.0)
            - max_location (tuple): Coordenadas (x, y) del mejor match
    
    Métodos disponibles:
        - TM_CCOEFF_NORMED: Correlación normalizada (recomendado)
        - TM_CCORR_NORMED: Correlación cruzada normalizada
        - TM_SQDIFF_NORMED: Diferencia cuadrática (valores bajos = mejor match)
    """
    # Verificar que roi es más grande que template
    if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
        return 0.0, (0, 0)
    
    # Realizar matching
    result = cv2.matchTemplate(roi, template, method)
    
    # Obtener posición y valor del mejor match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Para TM_SQDIFF_NORMED, menor es mejor
    if method == cv2.TM_SQDIFF_NORMED:
        score = 1.0 - min_val
        location = min_loc
    else:
        score = max_val
        location = max_loc
    
    return score, location


def match_template_multiscale(roi, template, scales=[0.8, 0.9, 1.0, 1.1, 1.2]):
    """
    Template matching con múltiples escalas.
    
    Útil cuando el tamaño del símbolo en la carta puede variar.
    
    Args:
        roi (np.ndarray): Región de interés (grayscale)
        template (np.ndarray): Template a buscar (grayscale)
        scales (list): Lista de factores de escala a probar
    
    Returns:
        tuple: (best_score, best_scale, best_location)
    """
    best_score = 0.0
    best_scale = 1.0
    best_location = (0, 0)
    
    for scale in scales:
        # Redimensionar template
        width = int(template.shape[1] * scale)
        height = int(template.shape[0] * scale)
        
        if width <= 0 or height <= 0:
            continue
        
        scaled_template = cv2.resize(template, (width, height))
        
        # Verificar que scaled_template cabe en roi
        if scaled_template.shape[0] > roi.shape[0] or scaled_template.shape[1] > roi.shape[1]:
            continue
        
        # Realizar matching
        score, location = match_template(roi, scaled_template)
        
        # Actualizar mejor resultado
        if score > best_score:
            best_score = score
            best_scale = scale
            best_location = location
    
    return best_score, best_scale, best_location


def match_value_templates(roi_valor):
    """
    Compara ROI del valor con todos los templates de valores.
    
    Args:
        roi_valor (np.ndarray): ROI de la esquina con el valor (BGR o grayscale)
    
    Returns:
        dict: {valor: score} con las puntuaciones de cada valor
    
    Ejemplo:
        {'AS': 0.95, '2': 0.23, '3': 0.18, ..., 'K': 0.31}
    """
    library = get_template_library()
    
    if not library.is_loaded():
        print("❌ Templates no cargados")
        return {}
    
    # Convertir a grayscale si es necesario
    if len(roi_valor.shape) == 3:
        roi_gray = cv2.cvtColor(roi_valor, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi_valor.copy()
    
    # Binarizar (símbolos oscuros quedan blancos)
    _, roi_thresh = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    scores = {}
    
    for valor in CARD_VALUES:
        template = library.get_value_template(valor)
        
        if template is None:
            scores[valor] = 0.0
            continue
        
        # Matching multi-escala (los valores pueden tener tamaños ligeramente diferentes)
        score, _, _ = match_template_multiscale(roi_thresh, template)
        scores[valor] = score
    
    return scores


def match_suit_templates(roi_palo):
    """
    Compara ROI del palo con todos los templates de palos.
    
    Args:
        roi_palo (np.ndarray): ROI con el símbolo del palo (BGR o grayscale)
    
    Returns:
        dict: {palo: score} con las puntuaciones de cada palo
    
    Ejemplo:
        {'PICAS': 0.88, 'CORAZONES': 0.34, 'DIAMANTES': 0.29, 'TREBOLES': 0.41}
    """
    library = get_template_library()
    
    if not library.is_loaded():
        print("❌ Templates no cargados")
        return {}
    
    # Convertir a grayscale si es necesario
    if len(roi_palo.shape) == 3:
        roi_gray = cv2.cvtColor(roi_palo, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi_palo.copy()
    
    # Binarizar
    _, roi_thresh = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    scores = {}
    
    for palo in CARD_SUITS:
        template = library.get_suit_template(palo)
        
        if template is None:
            scores[palo] = 0.0
            continue
        
        # Matching multi-escala
        score, _, _ = match_template_multiscale(roi_thresh, template)
        scores[palo] = score
    
    return scores


def get_best_match(scores, threshold=TEMPLATE_MATCH_THRESHOLD):
    """
    Obtiene el mejor match de un diccionario de scores.
    
    Args:
        scores (dict): {etiqueta: score}
        threshold (float): Umbral mínimo para considerar válido
    
    Returns:
        tuple: (mejor_etiqueta, mejor_score) o (None, 0.0) si no hay match válido
    """
    if not scores:
        return None, 0.0
    
    # Encontrar el máximo
    mejor_etiqueta = max(scores, key=scores.get)
    mejor_score = scores[mejor_etiqueta]
    
    # Verificar umbral
    if mejor_score < threshold:
        return None, mejor_score
    
    return mejor_etiqueta, mejor_score


def visualize_match(roi, template, location, score):
    """
    Dibuja un rectángulo mostrando dónde se encontró el template.
    
    Args:
        roi (np.ndarray): Imagen donde se buscó
        template (np.ndarray): Template encontrado
        location (tuple): Coordenadas (x, y) del match
        score (float): Puntuación del match
    
    Returns:
        np.ndarray: Imagen con rectángulo dibujado
    """
    # Convertir a BGR para dibujar en color
    if len(roi.shape) == 2:
        result = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    else:
        result = roi.copy()
    
    # Coordenadas del rectángulo
    top_left = location
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    
    # Dibujar rectángulo
    color = (0, 255, 0) if score > TEMPLATE_MATCH_THRESHOLD else (0, 0, 255)
    cv2.rectangle(result, top_left, bottom_right, color, 2)
    
    # Añadir texto con score
    text = f"Score: {score:.2f}"
    cv2.putText(result, text, (top_left[0], top_left[1] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return result


def create_template_from_roi(roi, output_path, threshold=150):
    """
    Crea un template limpio desde una ROI.
    
    Proceso:
        1. Convertir a grayscale
        2. Binarizar
        3. Encontrar contorno principal
        4. Recortar y guardar
    
    Args:
        roi (np.ndarray): Región de interés
        output_path (str): Ruta donde guardar el template
        threshold (int): Umbral de binarización
    
    Returns:
        bool: True si se creó exitosamente
    """
    # Convertir a grayscale
    if len(roi.shape) == 3:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi.copy()
    
    # Binarizar
    _, thresh = cv2.threshold(roi_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("⚠️  No se encontraron contornos en ROI")
        return False
    
    # Tomar el contorno más grande
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Obtener bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Recortar
    template = thresh[y:y+h, x:x+w]
    
    # Guardar
    cv2.imwrite(output_path, template)
    print(f"💾 Template guardado: {output_path}")
    
    return True


# ============================================================================
# FUNCIONES DE TEST
# ============================================================================

def test_template_matching():
    """Test básico de template matching."""
    print("\n🧪 Ejecutando test de template matching...")
    
    # Crear imagen sintética con un rectángulo
    test_image = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(test_image, (50, 50), (150, 150), 255, -1)
    
    # Crear template (rectángulo más pequeño)
    test_template = np.zeros((50, 50), dtype=np.uint8)
    cv2.rectangle(test_template, (10, 10), (40, 40), 255, -1)
    
    # Hacer matching
    score, location = match_template(test_image, test_template)
    
    print(f"   Score: {score:.3f}")
    print(f"   Location: {location}")
    
    if score > 0.8:
        print("   ✅ Test PASSED")
        return True
    else:
        print("   ❌ Test FAILED")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("MÓDULO DE TEMPLATE MATCHING")
    print("=" * 60)
    
    # Ejecutar test
    test_template_matching()
    
    # Intentar cargar biblioteca
    print("\n" + "=" * 60)
    library = get_template_library()
    
    if library.is_loaded():
        print("\n✅ Biblioteca de templates lista para usar")
    else:
        print("\n⚠️  Biblioteca no cargada. Ejecuta scripts/3_crear_templates.py primero")