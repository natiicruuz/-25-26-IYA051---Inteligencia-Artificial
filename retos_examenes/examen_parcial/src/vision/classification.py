"""
Módulo de clasificación de cartas sin Machine Learning.

Utiliza SOLO técnicas clásicas:
    - Template matching
    - Detección de color
    - Reglas lógicas de validación

NO utiliza ningún algoritmo de aprendizaje automático.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np

from src.vision.preprocessing import (
    extract_roi_region,
    is_red_card,
    binarize_roi,
    auto_rotate_card
)
from src.vision.template_matching import (
    match_value_templates,
    match_suit_templates,
    get_best_match
)
from config.settings import (
    ROI_CORNER_VALUE,
    ROI_CORNER_SUIT,
    SUIT_COLORS,
    TEMPLATE_MATCH_THRESHOLD
)


def classify_card(warped_card, debug=False):
    """
    Clasifica una carta usando solo técnicas clásicas de visión artificial.
    
    Pipeline de clasificación:
        1. Auto-rotación (detectar si está invertida)
        2. Extracción de ROIs (valor y palo)
        3. Detección de color (rojo vs negro)
        4. Template matching para valor
        5. Template matching para palo
        6. Validación por reglas lógicas
        7. Retorno de resultado
    
    Args:
        warped_card (np.ndarray): Carta normalizada (CARD_WIDTH x CARD_HEIGHT)
        debug (bool): Si True, muestra información de debugging
    
    Returns:
        dict: Resultado de la clasificación con estructura:
            {
                'carta': 'AS_PICAS' o 'DESCONOCIDA',
                'valor': 'AS',
                'palo': 'PICAS',
                'confianza_valor': 0.95,
                'confianza_palo': 0.88,
                'color_detectado': 'negro',
                'valido': True/False
            }
    """
    result = {
        'carta': 'DESCONOCIDA',
        'valor': None,
        'palo': None,
        'confianza_valor': 0.0,
        'confianza_palo': 0.0,
        'color_detectado': None,
        'valido': False
    }
    
    # 1. Auto-rotación de la carta
    warped_card = auto_rotate_card(warped_card)
    
    # 2. Extraer ROIs
    roi_valor = extract_roi_region(warped_card, ROI_CORNER_VALUE)
    roi_palo = extract_roi_region(warped_card, ROI_CORNER_SUIT)
    
    if debug:
        cv2.imshow('DEBUG: ROI Valor', roi_valor)
        cv2.imshow('DEBUG: ROI Palo', roi_palo)
    
    # 3. Detectar color
    es_roja = is_red_card(roi_palo)
    result['color_detectado'] = 'rojo' if es_roja else 'negro'
    
    # 4. Template matching para valor
    scores_valores = match_value_templates(roi_valor)
    mejor_valor, confianza_valor = get_best_match(scores_valores, TEMPLATE_MATCH_THRESHOLD)
    
    result['valor'] = mejor_valor
    result['confianza_valor'] = confianza_valor
    
    if debug and scores_valores:
        print(f"\n📊 Scores de valores:")
        sorted_valores = sorted(scores_valores.items(), key=lambda x: x[1], reverse=True)
        for valor, score in sorted_valores[:5]:  # Top 5
            print(f"   {valor}: {score:.3f}")
    
    # 5. Template matching para palo
    scores_palos = match_suit_templates(roi_palo)
    mejor_palo, confianza_palo = get_best_match(scores_palos, TEMPLATE_MATCH_THRESHOLD)
    
    result['palo'] = mejor_palo
    result['confianza_palo'] = confianza_palo
    
    if debug and scores_palos:
        print(f"\n📊 Scores de palos:")
        for palo, score in sorted(scores_palos.items(), key=lambda x: x[1], reverse=True):
            print(f"   {palo}: {score:.3f}")
    
    # 6. Validación por reglas lógicas
    if mejor_valor is None or mejor_palo is None:
        if debug:
            print("\n❌ No se pudo identificar valor o palo")
        return result
    
    # REGLA 1: Validar coherencia entre color detectado y palo
    color_esperado = SUIT_COLORS.get(mejor_palo)
    
    if color_esperado == 'rojo' and not es_roja:
        # Detectamos negro pero el palo es rojo → Corregir palo
        if debug:
            print(f"⚠️  Inconsistencia: Detectado {result['color_detectado']} pero palo es {mejor_palo}")
            print(f"   Corrigiendo a palo negro...")
        
        # Elegir el mejor palo negro (Picas o Tréboles)
        palos_negros = {'PICAS': scores_palos.get('PICAS', 0), 
                       'TREBOLES': scores_palos.get('TREBOLES', 0)}
        mejor_palo = max(palos_negros, key=palos_negros.get)
        confianza_palo = palos_negros[mejor_palo]
        result['palo'] = mejor_palo
        result['confianza_palo'] = confianza_palo
    
    elif color_esperado == 'negro' and es_roja:
        # Detectamos rojo pero el palo es negro → Corregir palo
        if debug:
            print(f"⚠️  Inconsistencia: Detectado {result['color_detectado']} pero palo es {mejor_palo}")
            print(f"   Corrigiendo a palo rojo...")
        
        # Elegir el mejor palo rojo (Corazones o Diamantes)
        palos_rojos = {'CORAZONES': scores_palos.get('CORAZONES', 0), 
                      'DIAMANTES': scores_palos.get('DIAMANTES', 0)}
        mejor_palo = max(palos_rojos, key=palos_rojos.get)
        confianza_palo = palos_rojos[mejor_palo]
        result['palo'] = mejor_palo
        result['confianza_palo'] = confianza_palo
    
    # REGLA 2: Verificar confianza mínima
    confianza_minima_global = 0.5
    
    if confianza_valor < confianza_minima_global or confianza_palo < confianza_minima_global:
        if debug:
            print(f"\n⚠️  Confianza baja: Valor={confianza_valor:.2f}, Palo={confianza_palo:.2f}")
        result['valido'] = False
        return result
    
    # 7. Construir etiqueta final
    result['carta'] = f"{mejor_valor}_{mejor_palo}"
    result['valido'] = True
    
    if debug:
        print(f"\n✅ Clasificación exitosa: {result['carta']}")
        print(f"   Confianza: Valor={confianza_valor:.2f}, Palo={confianza_palo:.2f}")
    
    return result


def classify_multiple_cards(cards_list, debug=False):
    """
    Clasifica múltiples cartas detectadas en un frame.
    
    Args:
        cards_list (list): Lista de tuplas (warped_card, contour, center) 
                          de detect_multiple_cards()
        debug (bool): Modo debugging
    
    Returns:
        list: Lista de resultados de clasificación
    """
    results = []
    
    for i, (warped_card, contour, center) in enumerate(cards_list):
        if debug:
            print(f"\n{'='*60}")
            print(f"Clasificando carta {i+1}/{len(cards_list)}")
            print(f"{'='*60}")
        
        result = classify_card(warped_card, debug=debug)
        result['center'] = center
        result['contour'] = contour
        results.append(result)
    
    return results


def get_classification_stats(results):
    """
    Calcula estadísticas de un conjunto de clasificaciones.
    
    Args:
        results (list): Lista de resultados de classify_card()
    
    Returns:
        dict: Estadísticas agregadas
    """
    total = len(results)
    validos = sum(1 for r in results if r['valido'])
    invalidos = total - validos
    
    if validos > 0:
        confianza_promedio_valor = np.mean([r['confianza_valor'] for r in results if r['valido']])
        confianza_promedio_palo = np.mean([r['confianza_palo'] for r in results if r['valido']])
    else:
        confianza_promedio_valor = 0.0
        confianza_promedio_palo = 0.0
    
    stats = {
        'total': total,
        'validos': validos,
        'invalidos': invalidos,
        'tasa_exito': validos / total if total > 0 else 0.0,
        'confianza_promedio_valor': confianza_promedio_valor,
        'confianza_promedio_palo': confianza_promedio_palo
    }
    
    return stats


def format_classification_text(result, include_confidence=True):
    """
    Formatea el resultado de clasificación como texto para mostrar en pantalla.
    
    Args:
        result (dict): Resultado de classify_card()
        include_confidence (bool): Si incluir niveles de confianza
    
    Returns:
        str: Texto formateado
    """
    if not result['valido']:
        return "DESCONOCIDA"
    
    carta = result['carta']
    
    if include_confidence:
        conf_v = result['confianza_valor']
        conf_p = result['confianza_palo']
        return f"{carta} (V:{conf_v:.2f} P:{conf_p:.2f})"
    
    return carta


def validate_card_existence(valor, palo):
    """
    Valida que una combinación valor-palo sea válida en una baraja estándar.
    
    Args:
        valor (str): Valor de la carta
        palo (str): Palo de la carta
    
    Returns:
        bool: True si es una carta válida
    """
    from config.settings import CARD_VALUES, CARD_SUITS
    
    return valor in CARD_VALUES and palo in CARD_SUITS


def compare_classifications(result1, result2):
    """
    Compara dos clasificaciones (útil para validación manual).
    
    Args:
        result1 (dict): Primer resultado
        result2 (dict): Segundo resultado
    
    Returns:
        bool: True si ambas clasificaciones coinciden
    """
    return (result1['valido'] and result2['valido'] and 
            result1['carta'] == result2['carta'])


# ============================================================================
# FUNCIONES DE ANÁLISIS Y DEBUGGING
# ============================================================================

def create_classification_report(results, output_path=None):
    """
    Genera un reporte detallado de clasificaciones.
    
    Args:
        results (list): Lista de resultados de classify_card()
        output_path (str): Ruta donde guardar el reporte (opcional)
    
    Returns:
        str: Reporte en formato texto
    """
    stats = get_classification_stats(results)
    
    report = []
    report.append("=" * 60)
    report.append("REPORTE DE CLASIFICACIÓN")
    report.append("=" * 60)
    report.append(f"\nTotal de cartas procesadas: {stats['total']}")
    report.append(f"Cartas válidas: {stats['validos']} ({stats['tasa_exito']*100:.1f}%)")
    report.append(f"Cartas no identificadas: {stats['invalidos']}")
    report.append(f"\nConfianza promedio:")
    report.append(f"  - Valores: {stats['confianza_promedio_valor']:.3f}")
    report.append(f"  - Palos: {stats['confianza_promedio_palo']:.3f}")
    
    report.append("\n" + "-" * 60)
    report.append("DETALLE POR CARTA:")
    report.append("-" * 60)
    
    for i, result in enumerate(results, 1):
        report.append(f"\nCarta {i}:")
        report.append(f"  Resultado: {result['carta']}")
        report.append(f"  Válido: {'✅' if result['valido'] else '❌'}")
        if result['valido']:
            report.append(f"  Valor: {result['valor']} (confianza: {result['confianza_valor']:.3f})")
            report.append(f"  Palo: {result['palo']} (confianza: {result['confianza_palo']:.3f})")
            report.append(f"  Color: {result['color_detectado']}")
    
    report.append("\n" + "=" * 60)
    
    report_text = "\n".join(report)
    
    # Guardar si se especifica ruta
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"💾 Reporte guardado en: {output_path}")
    
    return report_text


def analyze_misclassifications(ground_truth, predictions):
    """
    Analiza las clasificaciones incorrectas comparando con ground truth.
    
    Args:
        ground_truth (list): Lista de etiquetas correctas ['AS_PICAS', '2_CORAZONES', ...]
        predictions (list): Lista de resultados de classify_card()
    
    Returns:
        dict: Análisis de errores
    """
    if len(ground_truth) != len(predictions):
        print("⚠️  Las listas deben tener la misma longitud")
        return {}
    
    total = len(ground_truth)
    correctas = 0
    errores_valor = 0
    errores_palo = 0
    errores_ambos = 0
    no_detectadas = 0
    
    confusion_valores = {}
    confusion_palos = {}
    
    for gt, pred in zip(ground_truth, predictions):
        if not pred['valido']:
            no_detectadas += 1
            continue
        
        gt_valor, gt_palo = gt.split('_')
        pred_carta = pred['carta']
        
        if pred_carta == gt:
            correctas += 1
        else:
            pred_valor, pred_palo = pred_carta.split('_')
            
            if pred_valor != gt_valor and pred_palo != gt_palo:
                errores_ambos += 1
            elif pred_valor != gt_valor:
                errores_valor += 1
                confusion_valores[f"{gt_valor}->{pred_valor}"] = confusion_valores.get(f"{gt_valor}->{pred_valor}", 0) + 1
            else:
                errores_palo += 1
                confusion_palos[f"{gt_palo}->{pred_palo}"] = confusion_palos.get(f"{gt_palo}->{pred_palo}", 0) + 1
    
    accuracy = correctas / total if total > 0 else 0.0
    
    analysis = {
        'total': total,
        'correctas': correctas,
        'accuracy': accuracy,
        'errores_valor': errores_valor,
        'errores_palo': errores_palo,
        'errores_ambos': errores_ambos,
        'no_detectadas': no_detectadas,
        'confusion_valores': confusion_valores,
        'confusion_palos': confusion_palos
    }
    
    return analysis


def print_confusion_analysis(analysis):
    """
    Imprime de forma legible el análisis de confusiones.
    
    Args:
        analysis (dict): Resultado de analyze_misclassifications()
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS DE CLASIFICACIONES")
    print("=" * 60)
    print(f"\nTotal: {analysis['total']}")
    print(f"Correctas: {analysis['correctas']} ({analysis['accuracy']*100:.1f}%)")
    print(f"Errores solo en valor: {analysis['errores_valor']}")
    print(f"Errores solo en palo: {analysis['errores_palo']}")
    print(f"Errores en ambos: {analysis['errores_ambos']}")
    print(f"No detectadas: {analysis['no_detectadas']}")
    
    if analysis['confusion_valores']:
        print("\n📊 Confusiones en VALORES:")
        for confusion, count in sorted(analysis['confusion_valores'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {confusion}: {count} veces")
    
    if analysis['confusion_palos']:
        print("\n📊 Confusiones en PALOS:")
        for confusion, count in sorted(analysis['confusion_palos'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {confusion}: {count} veces")


# ============================================================================
# FUNCIONES DE TEST
# ============================================================================

def test_classification():
    """Test sintético de clasificación."""
    print("\n🧪 Test de clasificación (requiere templates cargados)...")
    
    from src.vision.template_matching import get_template_library
    
    library = get_template_library()
    
    if not library.is_loaded():
        print("⚠️  No se pueden ejecutar tests sin templates cargados")
        return False
    
    print("✅ Templates cargados, sistema listo para clasificar")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("MÓDULO DE CLASIFICACIÓN (SIN MACHINE LEARNING)")
    print("=" * 60)
    print("\nTécnicas utilizadas:")
    print("  ✅ Template matching (cv2.matchTemplate)")
    print("  ✅ Detección de color en HSV")
    print("  ✅ Reglas lógicas de validación")
    print("  ✅ Operaciones morfológicas")
    print("  ❌ NO Machine Learning")
    print("  ❌ NO Redes Neuronales")
    print("  ❌ NO SVM")
    
    # Ejecutar test
    test_classification()