import cv2
import numpy as np

print("="*60)
print("RECREANDO TEMPLATE DEL '10' LIMPIO")
print("="*60)

# 1. Cargar imagen de referencia
img = cv2.imread('data/imagenes_referencia/10_PICAS_1.jpg')
img = cv2.resize(img, (200, 300))

# 2. Extraer ROI
x, y, w, h = (0, 3, 70, 48)
roi_color = img[y:y+h, x:x+w]

print(f"ROI extraída: {roi_color.shape}")
cv2.imshow('1. ROI en COLOR', cv2.resize(roi_color, (300, 300)))
cv2.waitKey(1000)

# 3. Convertir a grayscale
roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
cv2.imshow('2. GRAYSCALE', cv2.resize(roi_gray, (300, 300)))
cv2.waitKey(1000)

# 4. Binarizar
_, roi_thresh = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY_INV)
print(f"Binarizado - píxeles blancos: {np.sum(roi_thresh > 0)}/{roi_thresh.size}")
cv2.imshow('3. BINARIZADO', cv2.resize(roi_thresh, (300, 300)))
cv2.waitKey(1000)

# 5. LIMPIEZA MORFOLÓGICA SUAVE (para no perder detalles)
kernel_small = np.ones((2, 2), np.uint8)
roi_clean = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, kernel_small)
roi_clean = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN, kernel_small)

print(f"Después de limpieza: {np.sum(roi_clean > 0)}/{roi_clean.size}")
cv2.imshow('4. LIMPIADO', cv2.resize(roi_clean, (300, 300)))
cv2.waitKey(1000)

# 6. Encontrar contornos y recortar
contours, _ = cv2.findContours(roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Contornos encontrados: {len(contours)}")

if contours:
    # Bounding box de TODOS los contornos juntos
    all_contours = np.vstack([c for c in contours])
    x_min = all_contours[:, 0, 0].min()
    y_min = all_contours[:, 0, 1].min()
    x_max = all_contours[:, 0, 0].max()
    y_max = all_contours[:, 0, 1].max()
    
    # Recortar
    template_crop = roi_clean[y_min:y_max+1, x_min:x_max+1]
    
    print(f"Recortado: {template_crop.shape}")
    cv2.imshow('5. RECORTADO', cv2.resize(template_crop, (150, 250)))
    cv2.waitKey(1000)
    
    # Padding pequeño
    padding = 2
    template_padded = cv2.copyMakeBorder(
        template_crop, padding, padding, padding, padding,
        cv2.BORDER_CONSTANT, value=0
    )
    
    print(f"Con padding: {template_padded.shape}")
    cv2.imshow('6. CON PADDING', cv2.resize(template_padded, (150, 250)))
    cv2.waitKey(1000)
    
    # Redimensionar a 30x50 (tamaño estándar)
    template_final = cv2.resize(template_padded, (30, 50))
    
    print(f"\n✅ TEMPLATE FINAL:")
    print(f"   Shape: {template_final.shape}")
    print(f"   Píxeles blancos: {np.sum(template_final > 0)}/{template_final.size}")
    print(f"   Porcentaje: {np.sum(template_final > 0)/template_final.size*100:.1f}%")
    
    cv2.imshow('7. TEMPLATE FINAL (30x50)', cv2.resize(template_final, (150, 250)))
    
    # TEST DE MATCHING
    print(f"\n🧪 TEST DE MATCHING:")
    
    # Cargar templates existentes
    template_6 = cv2.imread('data/templates/valores/6.png', cv2.IMREAD_GRAYSCALE)
    template_Q = cv2.imread('data/templates/valores/Q.png', cv2.IMREAD_GRAYSCALE)
    
    # Hacer matching
    for label, tmpl in [('10_NUEVO', template_final), ('6', template_6), ('Q', template_Q)]:
        if tmpl is None:
            continue
        
        max_score = 0.0
        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            w = int(tmpl.shape[1] * scale)
            h = int(tmpl.shape[0] * scale)
            
            if w > roi_thresh.shape[1] or h > roi_thresh.shape[0]:
                continue
            
            scaled = cv2.resize(tmpl, (w, h))
            
            try:
                result = cv2.matchTemplate(roi_thresh, scaled, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)
                max_score = max(max_score, score)
            except:
                pass
        
        print(f"   '{label}': {max_score:.3f}")
    
    print(f"\n{'='*60}")
    print("Si el score de '10_NUEVO' es el MÁS ALTO:")
    print("  → Presiona 'y' para GUARDAR")
    print("  → Presiona 'n' para NO guardar")
    print("='*60}")
    
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord('y'):
        cv2.imwrite('data/templates/valores/10.png', template_final)
        print("\n✅ Template guardado: data/templates/valores/10.png")
        print("\n🎯 SIGUIENTE PASO:")
        print("   python scripts/test_clasificacion.py")
        print("   Selecciona: 10_CORAZONES_0.jpg")
    else:
        print("\n❌ Template NO guardado")
        print("\nSi el score del '10_NUEVO' NO fue el más alto,")
        print("el problema es que la ROI captura mal el '10'.")
        print("Solución: Aumenta ROI_CORNER_VALUE en config/settings.py")
else:
    print("❌ No se encontraron contornos")

cv2.destroyAllWindows()