import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from config.settings import TEMPLATES_VALUES_DIR, TEMPLATE_VALUE_SIZE

print("="*60)
print("RECREAR TEMPLATE DEL '10'")
print("="*60)

# Cargar imagen de referencia con un 10
img_path = input("Ruta de imagen con 10 (ej: data/imagenes_referencia/10_CORAZONES_0.jpg): ")
carta = cv2.imread(img_path)

if carta is None:
    print("❌ No se pudo cargar")
    exit(1)

# Mostrar
carta_grande = cv2.resize(carta, (400, 600))
cv2.imshow('Carta - Haz click y arrastra sobre el "10"', carta_grande)

# Variables globales para ROI
roi_coords = []

def mouse_callback(event, x, y, flags, param):
    global roi_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_coords = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        roi_coords.append(x)
        roi_coords.append(y)
        
        # Escalar coordenadas al tamaño original
        scale_x = carta.shape[1] / 400
        scale_y = carta.shape[0] / 600
        
        x1 = int(roi_coords[0] * scale_x)
        y1 = int(roi_coords[1] * scale_y)
        x2 = int(roi_coords[2] * scale_x)
        y2 = int(roi_coords[3] * scale_y)
        
        # Extraer ROI
        roi = carta[y1:y2, x1:x2]
        
        # Convertir a grayscale y binarizar
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Mostrar
        cv2.imshow('Template extraido', cv2.resize(thresh, (150, 200)))
        
        # Confirmar
        print("\nPresiona 'y' para guardar, 'n' para reintentar")
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('y'):
            # Redimensionar y guardar
            template = cv2.resize(thresh, TEMPLATE_VALUE_SIZE)
            output_path = os.path.join(TEMPLATES_VALUES_DIR, '10.png')
            cv2.imwrite(output_path, template)
            print(f"✅ Template guardado: {output_path}")
            cv2.destroyAllWindows()
            exit(0)
        else:
            roi_coords = []
            cv2.destroyWindow('Template extraido')

cv2.setMouseCallback('Carta - Haz click y arrastra sobre el "10"', mouse_callback)
cv2.waitKey(0)