"""
Script de test para debuggear clasificación de una carta específica.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.vision.preprocessing import extract_roi_region
from src.vision.template_matching import (
    match_value_templates,
    match_suit_templates,
    get_template_library
)
from src.vision.classification import classify_card
from config.settings import ROI_CORNER_VALUE, ROI_CORNER_SUIT

# Cargar templates
library = get_template_library()

if not library.is_loaded():
    print("❌ Templates no cargados")
    sys.exit(1)

print("✅ Templates cargados correctamente\n")

# PASO 1: Cargar una imagen de referencia para testear
print("="*60)
print("Selecciona una imagen de referencia para testear")
print("="*60)

imagenes_dir = "data/imagenes_referencia"
imagenes = [f for f in os.listdir(imagenes_dir) if f.endswith('.jpg')]

print("\nImágenes disponibles:")
for i, img in enumerate(imagenes[:10]):  # Mostrar solo las primeras 10
    print(f"  {i}: {img}")

seleccion = int(input("\n📝 Introduce el número de la imagen a testear: "))
imagen_path = os.path.join(imagenes_dir, imagenes[seleccion])

print(f"\n📁 Cargando: {imagen_path}")

# Cargar imagen
carta = cv2.imread(imagen_path)

if carta is None:
    print(f"❌ No se pudo cargar la imagen")
    sys.exit(1)

# Redimensionar a tamaño estándar si es necesario
from config.settings import CARD_WIDTH, CARD_HEIGHT
carta = cv2.resize(carta, (CARD_WIDTH, CARD_HEIGHT))

print(f"✅ Imagen cargada: {carta.shape}")

# Mostrar imagen completa
cv2.imshow('Carta Completa', carta)
cv2.waitKey(1000)

# PASO 2: Extraer ROIs
print("\n" + "="*60)
print("EXTRACCIÓN DE ROIs")
print("="*60)

roi_valor = extract_roi_region(carta, ROI_CORNER_VALUE)
roi_palo = extract_roi_region(carta, ROI_CORNER_SUIT)

print(f"ROI Valor: {roi_valor.shape}")
print(f"ROI Palo: {roi_palo.shape}")

# Mostrar ROIs extraídas
roi_valor_grande = cv2.resize(roi_valor, (200, 300))
roi_palo_grande = cv2.resize(roi_palo, (200, 200))

cv2.imshow('ROI VALOR (lo que ve el sistema)', roi_valor_grande)
cv2.imshow('ROI PALO (lo que ve el sistema)', roi_palo_grande)
cv2.waitKey(2000)

# PASO 3: Template matching
print("\n" + "="*60)
print("TEMPLATE MATCHING - VALORES")
print("="*60)

scores_valores = match_value_templates(roi_valor)

print("\n📊 Top 5 matches de VALORES:")
sorted_valores = sorted(scores_valores.items(), key=lambda x: x[1], reverse=True)
for valor, score in sorted_valores[:5]:
    print(f"   {valor}: {score:.3f}")

print("\n" + "="*60)
print("TEMPLATE MATCHING - PALOS")
print("="*60)

scores_palos = match_suit_templates(roi_palo)

print("\n📊 Matches de PALOS:")
for palo, score in sorted(scores_palos.items(), key=lambda x: x[1], reverse=True):
    print(f"   {palo}: {score:.3f}")

# PASO 4: Clasificación completa
print("\n" + "="*60)
print("CLASIFICACIÓN COMPLETA")
print("="*60)

result = classify_card(carta, debug=True)

print("\n" + "="*60)
print("RESULTADO FINAL")
print("="*60)
print(f"Válido: {result['valido']}")
print(f"Carta: {result['carta']}")
print(f"Valor: {result['valor']} (confianza: {result['confianza_valor']:.3f})")
print(f"Palo: {result['palo']} (confianza: {result['confianza_palo']:.3f})")
print(f"Color detectado: {result['color_detectado']}")

# Mostrar templates de los mejores matches
print("\n" + "="*60)
print("MOSTRANDO TEMPLATES DE LOS MEJORES MATCHES")
print("="*60)

mejor_valor = sorted_valores[0][0]
mejor_palo = sorted(scores_palos.items(), key=lambda x: x[1], reverse=True)[0][0]

template_valor = library.get_value_template(mejor_valor)
template_palo = library.get_suit_template(mejor_palo)

if template_valor is not None:
    template_valor_grande = cv2.resize(template_valor, (150, 200))
    cv2.imshow(f'TEMPLATE: {mejor_valor}', template_valor_grande)

if template_palo is not None:
    template_palo_grande = cv2.resize(template_palo, (150, 150))
    cv2.imshow(f'TEMPLATE: {mejor_palo}', template_palo_grande)

print("\n📝 Presiona cualquier tecla para salir...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# DIAGNÓSTICO
print("\n" + "="*60)
print("DIAGNÓSTICO")
print("="*60)

if result['confianza_valor'] < 0.6:
    print("⚠️  PROBLEMA: Confianza del valor muy baja")
    print("   Solución: Recrea el template del valor con mejor ROI")

if result['confianza_palo'] < 0.6:
    print("⚠️  PROBLEMA: Confianza del palo muy baja")
    print("   Solución: Recrea el template del palo con mejor ROI")

if not result['valido']:
    print("\n❌ CARTA NO VÁLIDA")
    print("   Posibles causas:")
    print("   1. Templates mal creados (ROI muy grande o muy pequeña)")
    print("   2. ROI_CORNER_VALUE o ROI_CORNER_SUIT mal configuradas")
    print("   3. Umbral de confianza muy alto (TEMPLATE_MATCH_THRESHOLD)")
else:
    print("\n✅ CLASIFICACIÓN EXITOSA")