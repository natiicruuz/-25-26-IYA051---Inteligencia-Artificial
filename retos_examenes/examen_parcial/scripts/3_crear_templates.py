"""
Script para crear templates de valores y palos desde imágenes de referencia.

OBJETIVO:
    Extraer templates limpios de valores (AS, 2-10, J, Q, K) y palos (♠♥♦♣)
    desde las imágenes de referencia capturadas.

PROCESO INTERACTIVO:
    1. Carga una imagen de referencia
    2. Permite seleccionar con el mouse la ROI del VALOR
    3. Guarda el template del valor
    4. Permite seleccionar la ROI del PALO
    5. Guarda el template del palo
    6. Continúa con la siguiente carta

CONTROLES:
    - Click y arrastra = Seleccionar ROI
    - 'c' = Confirmar ROI y guardar template
    - 's' = Saltar esta imagen
    - 'q' = Salir
    - 'r' = Reiniciar selección

REQUISITOS:
    - Haber ejecutado scripts/2_capturar_imagenes_referencia.py
    - Tener imágenes en data/imagenes_referencia/

USO:
    python3 scripts/3_crear_templates.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config.settings import (
    DATA_DIR,
    TEMPLATES_VALUES_DIR,
    TEMPLATES_SUITS_DIR,
    CARD_VALUES,
    CARD_SUITS,
    TEMPLATE_VALUE_SIZE,
    TEMPLATE_SUIT_SIZE
)

# Directorio de imágenes de referencia
IMAGENES_REFERENCIA_DIR = os.path.join(DATA_DIR, 'imagenes_referencia')

# Variables globales para selección de ROI
roi_selecting = False
roi_start_point = None
roi_end_point = None
current_image = None
current_image_display = None


def mouse_callback(event, x, y, flags, param):
    """
    Callback para eventos del mouse (selección de ROI).
    
    Args:
        event: Tipo de evento del mouse
        x, y: Coordenadas del cursor
        flags: Flags adicionales
        param: Parámetros adicionales
    """
    global roi_selecting, roi_start_point, roi_end_point, current_image_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Inicio de selección
        roi_selecting = True
        roi_start_point = (x, y)
        roi_end_point = (x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE:
        # Arrastrando
        if roi_selecting:
            roi_end_point = (x, y)
            # Actualizar visualización
            temp_image = current_image_display.copy()
            cv2.rectangle(temp_image, roi_start_point, roi_end_point, (0, 255, 0), 2)
            cv2.imshow('Seleccionar ROI', temp_image)
    
    elif event == cv2.EVENT_LBUTTONUP:
        # Fin de selección
        roi_selecting = False
        roi_end_point = (x, y)
        # Dibujar rectángulo final
        temp_image = current_image_display.copy()
        cv2.rectangle(temp_image, roi_start_point, roi_end_point, (0, 255, 0), 2)
        cv2.imshow('Seleccionar ROI', temp_image)


def seleccionar_roi(imagen, titulo="Seleccionar ROI"):
    """
    Permite al usuario seleccionar una ROI interactivamente.
    
    Args:
        imagen (np.ndarray): Imagen donde seleccionar
        titulo (str): Título de la ventana
    
    Returns:
        np.ndarray o None: ROI seleccionada o None si se canceló
    """
    global roi_start_point, roi_end_point, current_image, current_image_display
    
    # Resetear variables
    roi_start_point = None
    roi_end_point = None
    current_image = imagen.copy()
    current_image_display = imagen.copy()
    
    # Crear ventana y configurar callback
    cv2.namedWindow(titulo)
    cv2.setMouseCallback(titulo, mouse_callback)
    
    print(f"\n{'='*60}")
    print(f"{titulo}")
    print(f"{'='*60}")
    print("📝 INSTRUCCIONES:")
    print("  1. Click y arrastra para seleccionar ROI")
    print("  2. Presiona 'c' para CONFIRMAR")
    print("  3. Presiona 'r' para REINICIAR selección")
    print("  4. Presiona 's' para SALTAR")
    print(f"{'='*60}\n")
    
    while True:
        cv2.imshow(titulo, current_image_display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            # Confirmar selección
            if roi_start_point and roi_end_point:
                # Calcular coordenadas de la ROI
                x1 = min(roi_start_point[0], roi_end_point[0])
                y1 = min(roi_start_point[1], roi_end_point[1])
                x2 = max(roi_start_point[0], roi_end_point[0])
                y2 = max(roi_start_point[1], roi_end_point[1])
                
                # Validar que la ROI tenga tamaño
                if x2 - x1 < 5 or y2 - y1 < 5:
                    print("⚠️  ROI demasiado pequeña. Intenta de nuevo.")
                    continue
                
                # Extraer ROI
                roi = current_image[y1:y2, x1:x2]
                cv2.destroyWindow(titulo)
                return roi
            else:
                print("⚠️  No has seleccionado una ROI. Intenta de nuevo.")
        
        elif key == ord('r'):
            # Reiniciar selección
            print("🔄 Reiniciando selección...")
            roi_start_point = None
            roi_end_point = None
            current_image_display = current_image.copy()
        
        elif key == ord('s'):
            # Saltar
            print("⏭️  Saltando...")
            cv2.destroyWindow(titulo)
            return None
        
        elif key == ord('q'):
            # Salir completamente
            print("🛑 Saliendo...")
            cv2.destroyAllWindows()
            sys.exit(0)


def procesar_roi_a_template(roi, threshold=150):
    """
    Procesa una ROI para convertirla en un template limpio.
    
    Proceso:
        1. Convertir a grayscale
        2. Binarizar (invertir para que símbolos oscuros queden blancos)
        3. Encontrar contorno principal
        4. Recortar solo el contorno (eliminar fondo extra)
        5. Aplicar operaciones morfológicas de limpieza
    
    Args:
        roi (np.ndarray): ROI seleccionada
        threshold (int): Umbral de binarización
    
    Returns:
        np.ndarray: Template procesado
    """
    # 1. Convertir a grayscale
    if len(roi.shape) == 3:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi.copy()
    
    # 2. Binarizar (símbolos oscuros → blancos)
    _, thresh = cv2.threshold(roi_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Limpieza morfológica
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 4. Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("⚠️  No se encontraron contornos. Usando ROI completa.")
        return thresh
    
    # 5. Tomar el contorno más grande
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 6. Obtener bounding box y recortar
    x, y, w, h = cv2.boundingRect(largest_contour)
    template = thresh[y:y+h, x:x+w]
    
    # 7. Añadir padding blanco (5 píxeles) para mejorar matching
    padding = 5
    template_padded = cv2.copyMakeBorder(
        template, padding, padding, padding, padding,
        cv2.BORDER_CONSTANT, value=0
    )
    
    return template_padded


def procesar_imagen_referencia(imagen_path):
    """
    Procesa una imagen de referencia para extraer templates de valor y palo.
    
    Args:
        imagen_path (str): Ruta de la imagen de referencia
    
    Returns:
        tuple: (template_valor, template_palo, etiqueta_carta)
    """
    # Cargar imagen
    imagen = cv2.imread(imagen_path)
    
    if imagen is None:
        print(f"❌ Error al cargar: {imagen_path}")
        return None, None, None
    
    # Extraer etiqueta del nombre del archivo
    filename = os.path.basename(imagen_path)
    # Formato: AS_PICAS_0.jpg → AS_PICAS
    etiqueta = '_'.join(filename.replace('.jpg', '').split('_')[:-1])
    valor, palo = etiqueta.split('_')
    
    print(f"\n{'='*60}")
    print(f"Procesando: {etiqueta}")
    print(f"{'='*60}")
    
    # Mostrar imagen completa
    cv2.imshow('Imagen Completa', imagen)
    cv2.waitKey(500)  # Pausa breve para visualizar
    
    # 1. Seleccionar ROI del VALOR
    print("\n🔢 PASO 1: Selecciona la ROI del VALOR (número o letra)")
    roi_valor = seleccionar_roi(imagen, f"Seleccionar VALOR de {etiqueta}")
    
    if roi_valor is None:
        print("⏭️  Saltando esta imagen...")
        cv2.destroyAllWindows()
        return None, None, None
    
    # Procesar template de valor
    template_valor = procesar_roi_a_template(roi_valor)
    
    # Mostrar template procesado
    cv2.imshow('Template VALOR procesado', template_valor)
    cv2.waitKey(1000)
    
    # 2. Seleccionar ROI del PALO
    print(f"\n♠♥♦♣ PASO 2: Selecciona la ROI del PALO ({palo})")
    roi_palo = seleccionar_roi(imagen, f"Seleccionar PALO de {etiqueta}")
    
    if roi_palo is None:
        print("⏭️  Saltando esta imagen...")
        cv2.destroyAllWindows()
        return None, None, None
    
    # Procesar template de palo
    template_palo = procesar_roi_a_template(roi_palo)
    
    # Mostrar template procesado
    cv2.imshow('Template PALO procesado', template_palo)
    cv2.waitKey(1000)
    
    cv2.destroyAllWindows()
    
    return template_valor, template_palo, (valor, palo)


def guardar_template(template, tipo, etiqueta, target_size):
    """
    Guarda un template redimensionándolo al tamaño estándar.
    
    Args:
        template (np.ndarray): Template a guardar
        tipo (str): 'valor' o 'palo'
        etiqueta (str): Etiqueta (ejemplo: 'AS' o 'PICAS')
        target_size (tuple): Tamaño objetivo (width, height)
    
    Returns:
        str: Ruta donde se guardó
    """
    # Redimensionar al tamaño estándar
    template_resized = cv2.resize(template, target_size)
    
    # Determinar directorio
    if tipo == 'valor':
        output_dir = TEMPLATES_VALUES_DIR
    else:
        output_dir = TEMPLATES_SUITS_DIR
    
    # Crear ruta
    filename = f"{etiqueta}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Guardar
    cv2.imwrite(filepath, template_resized)
    
    return filepath


def crear_templates():
    """Función principal para crear templates."""
    
    print("\n" + "=" * 60)
    print("CREACIÓN DE TEMPLATES")
    print("=" * 60)
    print("\nOBJETIVO:")
    print("  Extraer templates limpios de valores y palos desde imágenes de referencia")
    print("\nPROCESO:")
    print("  Para cada imagen:")
    print("    1. Selecciona ROI del valor (número/letra)")
    print("    2. Selecciona ROI del palo (símbolo)")
    print("    3. El sistema procesa y guarda los templates")
    print("\nRECOMENDACIÓN:")
    print("  - Selecciona ROIs ajustadas (solo el símbolo, sin fondo extra)")
    print("  - Usa imágenes con buena iluminación y contraste")
    print("  - Para cada valor/palo, procesa solo UNA imagen (la mejor)")
    print("=" * 60)
    
    # Verificar que existen imágenes de referencia
    if not os.path.exists(IMAGENES_REFERENCIA_DIR):
        print(f"\n❌ ERROR: No existe el directorio {IMAGENES_REFERENCIA_DIR}")
        print("   Ejecuta primero: python3 scripts/2_capturar_imagenes_referencia.py")
        return
    
    imagenes = [f for f in os.listdir(IMAGENES_REFERENCIA_DIR) if f.endswith('.jpg')]
    
    if not imagenes:
        print(f"\n❌ ERROR: No hay imágenes en {IMAGENES_REFERENCIA_DIR}")
        print("   Ejecuta primero: python3 scripts/2_capturar_imagenes_referencia.py")
        return
    
    print(f"\n✅ Encontradas {len(imagenes)} imágenes de referencia")
    
    # Agrupar imágenes por carta (pueden haber múltiples capturas de la misma carta)
    cartas_dict = {}
    for img in imagenes:
        # Formato: AS_PICAS_0.jpg → AS_PICAS
        etiqueta = '_'.join(img.replace('.jpg', '').split('_')[:-1])
        if etiqueta not in cartas_dict:
            cartas_dict[etiqueta] = []
        cartas_dict[etiqueta].append(img)
    
    print(f"📊 Total de cartas únicas: {len(cartas_dict)}")
    
    # Verificar templates existentes
    valores_existentes = set([f.replace('.png', '') for f in os.listdir(TEMPLATES_VALUES_DIR) if f.endswith('.png')])
    palos_existentes = set([f.replace('.png', '') for f in os.listdir(TEMPLATES_SUITS_DIR) if f.endswith('.png')])
    
    print(f"\n📁 Templates existentes:")
    print(f"   Valores: {len(valores_existentes)}/{len(CARD_VALUES)}")
    print(f"   Palos: {len(palos_existentes)}/{len(CARD_SUITS)}")
    
    # Preguntar si sobrescribir
    if valores_existentes or palos_existentes:
        respuesta = input("\n⚠️  ¿Sobrescribir templates existentes? (s/n): ").lower()
        sobrescribir = respuesta == 's'
    else:
        sobrescribir = True
    
    # Procesar cada carta
    templates_creados_valores = 0
    templates_creados_palos = 0
    
    for i, (etiqueta, imagenes_carta) in enumerate(sorted(cartas_dict.items()), 1):
        valor, palo = etiqueta.split('_')
        
        print(f"\n{'='*60}")
        print(f"[{i}/{len(cartas_dict)}] {etiqueta}")
        print(f"{'='*60}")
        
        # Verificar si ya existen templates
        if not sobrescribir:
            if valor in valores_existentes and palo in palos_existentes:
                print(f"⏭️  Templates ya existen. Saltando...")
                continue
        
        # Usar la primera imagen de esta carta
        imagen_path = os.path.join(IMAGENES_REFERENCIA_DIR, imagenes_carta[0])
        
        if len(imagenes_carta) > 1:
            print(f"ℹ️  Hay {len(imagenes_carta)} imágenes de esta carta. Usando: {imagenes_carta[0]}")
        
        # Procesar imagen
        template_valor, template_palo, resultado = procesar_imagen_referencia(imagen_path)
        
        if template_valor is None or template_palo is None:
            print("❌ No se pudieron crear los templates")
            continue
        
        # Guardar templates
        if valor not in valores_existentes or sobrescribir:
            path_valor = guardar_template(template_valor, 'valor', valor, TEMPLATE_VALUE_SIZE)
            print(f"✅ Template VALOR guardado: {path_valor}")
            valores_existentes.add(valor)
            templates_creados_valores += 1
        else:
            print(f"⏭️  Template de valor '{valor}' ya existe")
        
        if palo not in palos_existentes or sobrescribir:
            path_palo = guardar_template(template_palo, 'palo', palo, TEMPLATE_SUIT_SIZE)
            print(f"✅ Template PALO guardado: {path_palo}")
            palos_existentes.add(palo)
            templates_creados_palos += 1
        else:
            print(f"⏭️  Template de palo '{palo}' ya existe")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"Templates de VALORES creados: {templates_creados_valores}")
    print(f"Templates de PALOS creados: {templates_creados_palos}")
    print(f"\n📁 Templates guardados en:")
    print(f"   Valores: {TEMPLATES_VALUES_DIR}")
    print(f"   Palos: {TEMPLATES_SUITS_DIR}")
    
    # Verificar completitud
    valores_faltantes = set(CARD_VALUES) - valores_existentes
    palos_faltantes = set(CARD_SUITS) - palos_existentes
    
    if valores_faltantes:
        print(f"\n⚠️  Valores faltantes: {', '.join(valores_faltantes)}")
    else:
        print(f"\n✅ Todos los valores completos ({len(CARD_VALUES)}/{len(CARD_VALUES)})")
    
    if palos_faltantes:
        print(f"⚠️  Palos faltantes: {', '.join(palos_faltantes)}")
    else:
        print(f"✅ Todos los palos completos ({len(CARD_SUITS)}/{len(CARD_SUITS)})")
    
    print("\n📌 SIGUIENTE PASO:")
    print("   Ejecuta: python3 scripts/4_validar_templates.py")
    print("=" * 60)


if __name__ == "__main__":
    try:
        crear_templates()
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()