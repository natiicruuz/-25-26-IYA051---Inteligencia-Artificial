import cv2
import numpy as np

# Cargar imagen de DIAMANTES
img = cv2.imread('data/imagenes_referencia/3_DIAMANTES_1.jpg')
img = cv2.resize(img, (200, 300))

# Extraer ROI del palo
x, y, w, h = (5, 50, 40, 40)  # ROI_CORNER_SUIT
roi_palo = img[y:y+h, x:x+w]

# Analizar color
mean_b = np.mean(roi_palo[:,:,0])
mean_g = np.mean(roi_palo[:,:,1])
mean_r = np.mean(roi_palo[:,:,2])

diff_r_g = mean_r - mean_g
diff_r_b = mean_r - mean_b

print(f"BGR: B={mean_b:.1f}, G={mean_g:.1f}, R={mean_r:.1f}")
print(f"Diferencias: R-G={diff_r_g:.1f}, R-B={diff_r_b:.1f}")
print(f"¿Es rojo? R-G>30 AND R-B>40")
print(f"  R-G>30: {diff_r_g > 30}")
print(f"  R-B>40: {diff_r_b > 40}")
print(f"  Resultado: {(diff_r_g > 30) and (diff_r_b > 40)}")

cv2.imshow('ROI Palo', cv2.resize(roi_palo, (200, 200)))
cv2.waitKey(0)