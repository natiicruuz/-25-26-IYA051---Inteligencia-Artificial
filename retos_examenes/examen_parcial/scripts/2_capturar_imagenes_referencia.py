"""
Script para capturar imágenes de referencia de cartas.

OBJETIVO:
    Capturar imágenes limpias de cada carta para luego extraer los templates
    de valores (AS, 2-10, J, Q, K) y palos (♠♥♦♣).

PROTOCOLO DE CAPTURA:
    1. Coloca UNA carta sobre el tapete verde
    2. Centra la carta en el campo de visión
    3. Presiona 's' para capturar
    4. Introduce la etiqueta (ejemplo: AS_PICAS)
    5. El sistema guardará la carta normalizada
    6. Repite para las 52 cartas

REQUISITOS:
    - Tapete verde calibrado
    - Iluminación uniforme
    - Cartas limpias y sin dobleces
    - Capturar al menos 1 imagen por carta (recomendado: 2-3 orientaciones)

USO:
    python3 scripts/2_capturar_imagenes_referencia.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
from config.settings import (
    get_rtsp_url,
    CARD_VALUES,
    CARD_SUITS,
    get_card_label,
    is_valid_card_label,
    DATA_DIR,
    CAPTURE_COOLDOWN
)
from src.vision.preprocessing import preprocess_and_warp, visualize_detection

# Directorio para guardar imágenes de referencia
IMAGENES_REFERENCIA_DIR = os.path.join(DATA_DIR, 'imagenes_referencia')
os.makedirs(IMAGENES_REFERENCIA_DIR, exist_ok=True)


def mostrar_instrucciones():
    """Muestra las instrucciones de uso."""
    print("\n" + "=" * 60)
    print("CAPTURA DE IMÁGENES DE REFERENCIA")
    print("=" * 60)
    print("\nOBJETIVO:")
    print("  Capturar imágenes limpias de cada carta para crear templates")
    print("\nPROTOCOLO:")
    print("  1. Coloca UNA carta sobre el tapete verde")
    print("  2. Centra la carta en el campo de visión")
    print("  3. Presiona 's' para CAPTURAR")
    print("  4. Introduce la etiqueta (ejemplo: AS_PICAS)")
    print("  5. El sistema guardará la carta normalizada")
    print("  6. Repite para las 52 cartas")
    print("\nFORMATO DE ETIQUETAS:")
    print(f"  Valores: {', '.join(CARD_VALUES)}")
    print(f"  Palos: {', '.join(CARD_SUITS)}")
    print("  Ejemplo: AS_PICAS, 7_CORAZONES, K_DIAMANTES")
    print("\nCONTROLES:")
    print("  's' = Capturar imagen")
    print("  'q' = Salir")
    print("  'h' = Mostrar ayuda")
    print("=" * 60 + "\n")


def mostrar_progreso():
    """Muestra cuántas cartas se han capturado."""
    archivos = os.listdir(IMAGENES_REFERENCIA_DIR)
    imagenes = [f for f in archivos if f.endswith('.jpg')]
    
    # Contar cartas únicas (sin contar múltiples capturas de la misma carta)
    cartas_capturadas = set()
    for img in imagenes:
        # Formato: AS_PICAS_0.jpg, AS_PICAS_1.jpg, etc.
        partes = img.replace('.jpg', '').rsplit('_', 1)
        if len(partes) > 1:
            carta = partes[0]
            cartas_capturadas.add(carta)
    
    total_cartas = len(CARD_VALUES) * len(CARD_SUITS)  # 52
    progreso = len(cartas_capturadas)
    porcentaje = (progreso / total_cartas) * 100
    
    print(f"\n📊 PROGRESO: {progreso}/{total_cartas} cartas únicas ({porcentaje:.1f}%)")
    print(f"   Total de imágenes capturadas: {len(imagenes)}")
    
    # Mostrar cartas faltantes
    todas_cartas = set([f"{v}_{p}" for v in CARD_VALUES for p in CARD_SUITS])
    faltantes = todas_cartas - cartas_capturadas
    
    if faltantes:
        print(f"\n⚠️  Cartas pendientes ({len(faltantes)}):")
        faltantes_sorted = sorted(list(faltantes))
        for i in range(0, len(faltantes_sorted), 5):
            print(f"   {', '.join(faltantes_sorted[i:i+5])}")


def validar_etiqueta(etiqueta):
    """
    Valida que la etiqueta introducida sea correcta.
    
    Args:
        etiqueta (str): Etiqueta a validar
    
    Returns:
        bool: True si es válida, False en caso contrario
    """
    if not etiqueta:
        return False
    
    etiqueta = etiqueta.upper().strip()
    
    if '_' not in etiqueta:
        print("❌ Formato incorrecto. Usa: VALOR_PALO (ejemplo: AS_PICAS)")
        return False
    
    partes = etiqueta.split('_')
    
    if len(partes) != 2:
        print("❌ Formato incorrecto. Debe tener exactamente un '_'")
        return False
    
    valor, palo = partes
    
    if valor not in CARD_VALUES:
        print(f"❌ Valor '{valor}' no válido. Valores permitidos: {', '.join(CARD_VALUES)}")
        return False
    
    if palo not in CARD_SUITS:
        print(f"❌ Palo '{palo}' no válido. Palos permitidos: {', '.join(CARD_SUITS)}")
        return False
    
    return True


def obtener_siguiente_numero(etiqueta):
    """
    Obtiene el siguiente número disponible para una carta.
    
    Si ya existe AS_PICAS_0.jpg, retornará 1 para AS_PICAS_1.jpg
    
    Args:
        etiqueta (str): Etiqueta de la carta (ejemplo: AS_PICAS)
    
    Returns:
        int: Siguiente número disponible
    """
    archivos = os.listdir(IMAGENES_REFERENCIA_DIR)
    numeros_existentes = []
    
    for archivo in archivos:
        if archivo.startswith(etiqueta) and archivo.endswith('.jpg'):
            # Extraer número: AS_PICAS_3.jpg -> 3
            partes = archivo.replace('.jpg', '').split('_')
            try:
                num = int(partes[-1])
                numeros_existentes.append(num)
            except ValueError:
                continue
    
    if not numeros_existentes:
        return 0
    
    return max(numeros_existentes) + 1


def capturar_imagenes_referencia():
    """Función principal de captura."""
    
    mostrar_instrucciones()
    
    # Conectar al stream RTSP
    rtsp_url = get_rtsp_url()
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print(f"❌ ERROR: No se pudo conectar al stream RTSP")
        print(f"   URL: {rtsp_url}")
        return
    
    print("✅ Conexión RTSP exitosa")
    mostrar_progreso()
    
    ultimo_capture = 0  # Timestamp del último capture
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("⚠️  No se pudo recibir frame. Reintentando...")
            time.sleep(1)
            continue
        
        # Intentar detectar y normalizar carta
        warped_card, contour, _ = preprocess_and_warp(frame, debug=False)
        # Preparar frame para visualización
        display_frame = frame.copy()
        
        # Información en pantalla
        info_text = "Presiona 's' para CAPTURAR | 'q' para SALIR | 'h' para AYUDA"
        cv2.putText(display_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if warped_card is not None:
            # Carta detectada correctamente
            status_text = "CARTA DETECTADA - Lista para capturar"
            cv2.putText(display_frame, status_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Dibujar contorno verde
            cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 3)
            
            # Mostrar carta normalizada
            cv2.imshow('Carta Normalizada (se guardará esta)', warped_card)
        else:
            # No se detectó carta válida
            status_text = "ESPERANDO CARTA..."
            cv2.putText(display_frame, status_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if contour is not None:
                # Contorno detectado pero no válido
                cv2.drawContours(display_frame, [contour], -1, (0, 0, 255), 2)
        
        # Mostrar frame principal
        cv2.imshow('CAPTURA DE REFERENCIA', display_frame)
        
        # Detectar tecla
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Saliendo...")
            break
        
        elif key == ord('h'):
            mostrar_instrucciones()
            mostrar_progreso()
        
        elif key == ord('s'):
            # Verificar cooldown
            tiempo_actual = time.time()
            if tiempo_actual - ultimo_capture < CAPTURE_COOLDOWN:
                print("⚠️  Espera un momento antes de capturar otra carta")
                continue
            
            # Verificar que hay carta detectada
            if warped_card is None:
                print("❌ No hay carta detectada. Coloca una carta sobre el tapete.")
                continue
            
            # Solicitar etiqueta
            print("\n" + "-" * 60)
            etiqueta = input("📝 Introduce la etiqueta (ejemplo: AS_PICAS): ").upper().strip()
            
            # Validar etiqueta
            if not validar_etiqueta(etiqueta):
                continue
            
            # Obtener número siguiente
            numero = obtener_siguiente_numero(etiqueta)
            
            # Guardar imagen
            filename = f"{etiqueta}_{numero}.jpg"
            filepath = os.path.join(IMAGENES_REFERENCIA_DIR, filename)
            
            cv2.imwrite(filepath, warped_card)
            
            print(f"✅ Guardado: {filename}")
            print(f"   Ruta: {filepath}")
            
            # Actualizar timestamp
            ultimo_capture = tiempo_actual
            
            # Mostrar progreso actualizado
            mostrar_progreso()
    
    # Limpieza
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    mostrar_progreso()
    print(f"\n📁 Imágenes guardadas en: {IMAGENES_REFERENCIA_DIR}")
    print("\n📌 SIGUIENTE PASO:")
    print("   Ejecuta: python3 scripts/3_crear_templates.py")
    print("=" * 60)


if __name__ == "__main__":
    try:
        capturar_imagenes_referencia()
    except KeyboardInterrupt:
        print("\n\n⚠️  Captura interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()