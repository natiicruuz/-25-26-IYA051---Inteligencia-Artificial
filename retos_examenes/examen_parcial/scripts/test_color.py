import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from src.vision.preprocessing import is_red_card, extract_roi_region
from config.settings import ROI_CORNER_SUIT

# Cargar imagen de CORAZONES
img_path = input("Introduce ruta de imagen de CORAZONES (ej: data/imagenes_referencia/AS_CORAZONES_0.jpg): ")
carta = cv2.imread(img_path)

if carta is None:
    print("Error al cargar")
    exit(1)

# Redimensionar
from config.settings import CARD_WIDTH, CARD_HEIGHT
carta = cv2.resize(carta, (CARD_WIDTH, CARD_HEIGHT))

# Extraer ROI del palo
roi_palo = extract_roi_region(carta, ROI_CORNER_SUIT)

# Mostrar
cv2.imshow('Carta completa', carta)
roi_grande = cv2.resize(roi_palo, (200, 200))
cv2.imshow('ROI del palo', roi_grande)

# Detectar color
es_roja = is_red_card(roi_palo)

print(f"\n{'='*60}")
print(f"RESULTADO:")
print(f"{'='*60}")
print(f"Es roja: {es_roja}")
print(f"Debería ser: True (CORAZONES)")

# Analizar HSV
hsv_roi = cv2.cvtColor(roi_palo, cv2.COLOR_BGR2HSV)
mean_h = np.mean(hsv_roi[:,:,0])
mean_s = np.mean(hsv_roi[:,:,1])
mean_v = np.mean(hsv_roi[:,:,2])

print(f"\nPromedios HSV:")
print(f"  H (Hue): {mean_h:.1f}")
print(f"  S (Saturation): {mean_s:.1f}")
print(f"  V (Value): {mean_v:.1f}")

# Crear máscaras
from config.settings import LOWER_RED_HSV_1, UPPER_RED_HSV_1, LOWER_RED_HSV_2, UPPER_RED_HSV_2

mask_red_1 = cv2.inRange(hsv_roi, LOWER_RED_HSV_1, UPPER_RED_HSV_1)
mask_red_2 = cv2.inRange(hsv_roi, LOWER_RED_HSV_2, UPPER_RED_HSV_2)
mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

red_pixels = np.sum(mask_red > 0)
total_pixels = roi_palo.shape[0] * roi_palo.shape[1]
red_percentage = (red_pixels / total_pixels) * 100

print(f"\nAnálisis de color:")
print(f"  Píxeles rojos: {red_pixels}/{total_pixels}")
print(f"  Porcentaje rojo: {red_percentage:.1f}%")
print(f"  Umbral actual: 20%")

if red_percentage < 20:
    print(f"\n⚠️  PROBLEMA: Porcentaje muy bajo para detectar rojo")
    print(f"  Solución: Ajustar rangos HSV o bajar umbral")

# Mostrar máscaras
cv2.imshow('Mascara roja 1', mask_red_1)
cv2.imshow('Mascara roja 2', mask_red_2)
cv2.imshow('Mascara roja combinada', mask_red)

cv2.waitKey(0)
cv2.destroyAllWindows()