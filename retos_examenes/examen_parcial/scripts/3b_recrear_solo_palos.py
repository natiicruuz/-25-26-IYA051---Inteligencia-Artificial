"""
Script para recrear SOLO los templates de palos.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config.settings import (
    TEMPLATES_SUITS_DIR,
    CARD_SUITS,
    TEMPLATE_SUIT_SIZE,
    DATA_DIR
)

# Variables globales para selección de ROI
roi_selecting = False
roi_start_point = None
roi_end_point = None
current_image = None
current_image_display = None


def mouse_callback(event, x, y, flags, param):
    """Callback para eventos del mouse."""
    global roi_selecting, roi_start_point, roi_end_point, current_image_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_selecting = True
        roi_start_point = (x, y)
        roi_end_point = (x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if roi_selecting:
            roi_end_point = (x, y)
            temp_image = current_image_display.copy()
            cv2.rectangle(temp_image, roi_start_point, roi_end_point, (0, 255, 0), 2)
            cv2.imshow('Seleccionar ROI', temp_image)
    
    elif event == cv2.EVENT_LBUTTONUP:
        roi_selecting = False
        roi_end_point = (x, y)
        temp_image = current_image_display.copy()
        cv2.rectangle(temp_image, roi_start_point, roi_end_point, (0, 255, 0), 2)
        cv2.imshow('Seleccionar ROI', temp_image)


def seleccionar_roi(imagen, titulo="Seleccionar ROI"):
    """Permite seleccionar una ROI interactivamente."""
    global roi_start_point, roi_end_point, current_image, current_image_display
    
    roi_start_point = None
    roi_end_point = None
    current_image = imagen.copy()
    current_image_display = imagen.copy()
    
    cv2.namedWindow(titulo)
    cv2.setMouseCallback(titulo, mouse_callback)
    
    print(f"\n📝 {titulo}")
    print("   Click y arrastra para seleccionar ROI")
    print("   'c' = Confirmar | 'r' = Reiniciar | 's' = Saltar")
    
    while True:
        cv2.imshow(titulo, current_image_display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            if roi_start_point and roi_end_point:
                x1 = min(roi_start_point[0], roi_end_point[0])
                y1 = min(roi_start_point[1], roi_end_point[1])
                x2 = max(roi_start_point[0], roi_end_point[0])
                y2 = max(roi_start_point[1], roi_end_point[1])
                
                width = x2 - x1
                height = y2 - y1
                
                if width < 10 or height < 10:
                    print("⚠️  ROI demasiado pequeña")
                    continue
                
                roi = current_image[y1:y2, x1:x2]
                
                if roi is None or roi.size == 0:
                    print("❌ ROI vacía")
                    continue
                
                # Mostrar ROI antes de confirmar
                roi_grande = cv2.resize(roi, (200, 200))
                cv2.imshow('ROI - ENTER=Confirmar ESC=Reiniciar', roi_grande)
                key_confirm = cv2.waitKey(0) & 0xFF
                cv2.destroyWindow('ROI - ENTER=Confirmar ESC=Reiniciar')
                
                if key_confirm == 13:  # ENTER
                    cv2.destroyWindow(titulo)
                    return roi
                else:
                    roi_start_point = None
                    roi_end_point = None
                    current_image_display = current_image.copy()
            else:
                print("⚠️  No has seleccionado ROI")
        
        elif key == ord('r'):
            roi_start_point = None
            roi_end_point = None
            current_image_display = current_image.copy()
        
        elif key == ord('s'):
            cv2.destroyWindow(titulo)
            return None
        
        elif key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)


def procesar_roi_a_template(roi, threshold=150):
    """Procesa ROI en template limpio."""
    if roi is None or roi.size == 0:
        return None
    
    # Convertir a grayscale
    if len(roi.shape) == 3:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi.copy()
    
    # Binarizar (símbolos oscuros → blancos)
    _, thresh = cv2.threshold(roi_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Limpieza morfológica
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return thresh
    
    # Contorno más grande
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    template = thresh[y:y+h, x:x+w]
    
    # Padding
    padding = 5
    template_padded = cv2.copyMakeBorder(
        template, padding, padding, padding, padding,
        cv2.BORDER_CONSTANT, value=0
    )
    
    return template_padded


def recrear_palos():
    """Recrea templates de palos."""
    
    print("="*60)
    print("RECREAR TEMPLATES DE PALOS")
    print("="*60)
    print("\nESTRATEGIA:")
    print("  1. Para cada palo, selecciona UNA imagen con ese palo")
    print("  2. Selecciona SOLO el símbolo del palo (ajustado)")
    print("  3. El sistema creará el template automáticamente")
    print("\n⚠️  IMPORTANTE:")
    print("  - Selecciona el símbolo del CENTRO de la carta")
    print("  - Ajusta la ROI lo más posible al símbolo")
    print("  - NO incluyas mucho espacio blanco")
    print("="*60)
    
    imagenes_dir = os.path.join(DATA_DIR, 'imagenes_referencia')
    imagenes = [f for f in os.listdir(imagenes_dir) if f.endswith('.jpg')]
    
    # Agrupar por palo
    cartas_por_palo = {palo: [] for palo in CARD_SUITS}
    
    for img in imagenes:
        for palo in CARD_SUITS:
            if palo in img:
                cartas_por_palo[palo].append(img)
                break
    
    print(f"\n📊 Imágenes disponibles por palo:")
    for palo, imgs in cartas_por_palo.items():
        print(f"   {palo}: {len(imgs)} imágenes")
    
    # Procesar cada palo
    for palo in CARD_SUITS:
        print("\n" + "="*60)
        print(f"PALO: {palo}")
        print("="*60)
        
        if not cartas_por_palo[palo]:
            print(f"❌ No hay imágenes con {palo}")
            continue
        
        print(f"\nImágenes disponibles con {palo}:")
        for i, img in enumerate(cartas_por_palo[palo][:10]):
            print(f"   {i}: {img}")
        
        seleccion = input(f"\n📝 Selecciona imagen para {palo} (número o 's' para saltar): ")
        
        if seleccion.lower() == 's':
            print(f"⏭️  Saltando {palo}")
            continue
        
        try:
            idx = int(seleccion)
            imagen_path = os.path.join(imagenes_dir, cartas_por_palo[palo][idx])
        except:
            print("❌ Selección inválida")
            continue
        
        # Cargar imagen
        carta = cv2.imread(imagen_path)
        if carta is None:
            print(f"❌ No se pudo cargar {imagen_path}")
            continue
        
        print(f"\n✅ Cargada: {imagen_path}")
        
        # Mostrar carta completa
        carta_display = cv2.resize(carta, (400, 600))
        cv2.imshow('Carta Completa', carta_display)
        cv2.waitKey(1000)
        
        # Seleccionar ROI del palo
        print(f"\n♠♥♦♣ Selecciona el símbolo {palo} (centro de la carta)")
        roi_palo = seleccionar_roi(carta, f"Seleccionar {palo}")
        
        if roi_palo is None:
            print(f"⏭️  Saltando {palo}")
            cv2.destroyAllWindows()
            continue
        
        # Procesar template
        template = procesar_roi_a_template(roi_palo)
        
        if template is None:
            print(f"❌ Error al procesar template de {palo}")
            continue
        
        # Mostrar template procesado
        template_grande = cv2.resize(template, (200, 200))
        cv2.imshow(f'Template {palo} procesado', template_grande)
        cv2.waitKey(2000)
        
        # Redimensionar y guardar
        template_resized = cv2.resize(template, TEMPLATE_SUIT_SIZE)
        filepath = os.path.join(TEMPLATES_SUITS_DIR, f"{palo}.png")
        cv2.imwrite(filepath, template_resized)
        
        print(f"✅ Template guardado: {filepath}")
        
        cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("RECREACIÓN COMPLETADA")
    print("="*60)
    print("\n📌 SIGUIENTE PASO:")
    print("   Ejecuta: python scripts/test_clasificacion.py")
    print("="*60)


if __name__ == "__main__":
    try:
        recrear_palos()
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()