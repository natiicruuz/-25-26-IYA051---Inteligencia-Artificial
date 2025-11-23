import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from src.vision.preprocessing import is_red_card, extract_roi_region
from config.settings import ROI_CORNER_SUIT, CARD_WIDTH, CARD_HEIGHT

imagenes_dir = 'data/imagenes_referencia'
imagenes = [f for f in os.listdir(imagenes_dir) if f.endswith('.jpg')]

# Filtrar solo ROJAS (Corazones y Diamantes)
rojas = [img for img in imagenes if 'CORAZONES' in img or 'DIAMANTES' in img]  

print(f"{'='*60}")
print(f"VERIFICANDO DETECCIÓN DE COLOR EN {len(rojas)} CARTAS ROJAS")
print(f"{'='*60}\n")

correctas = 0
incorrectas = 0

for img_name in sorted(rojas):
    img_path = os.path.join(imagenes_dir, img_name)
    carta = cv2.imread(img_path)
    carta = cv2.resize(carta, (CARD_WIDTH, CARD_HEIGHT))
    
    # Extraer ROI del palo
    roi_palo = extract_roi_region(carta, ROI_CORNER_SUIT)
    
    # Detectar color
    es_roja = is_red_card(roi_palo)
    
    # Analizar BGR
    mean_b = np.mean(roi_palo[:,:,0])
    mean_g = np.mean(roi_palo[:,:,1])
    mean_r = np.mean(roi_palo[:,:,2])
    diff_r_g = mean_r - mean_g
    diff_r_b = mean_r - mean_b
    
    if es_roja:
        print(f"✅ {img_name:30s} → Rojo detectado (R-G={diff_r_g:.1f}, R-B={diff_r_b:.1f})")
        correctas += 1
    else:
        print(f"❌ {img_name:30s} → NO detectado  (R-G={diff_r_g:.1f}, R-B={diff_r_b:.1f})")
        incorrectas += 1

print(f"\n{'='*60}")
print(f"RESULTADO: {correctas}/{len(rojas)} correctas ({correctas/len(rojas)*100:.1f}%)")
if incorrectas > 0:
    print(f"⚠️  {incorrectas} cartas rojas NO fueron detectadas")
print(f"{'='*60}")