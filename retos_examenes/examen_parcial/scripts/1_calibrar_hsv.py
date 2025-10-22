"""
Script para calibrar los valores HSV del tapete verde/azul.

OBJETIVO:
    Encontrar los valores óptimos de HSV que detecten SOLO el fondo (tapete)
    dejando las cartas en negro.

RESULTADO ESPERADO:
    - Máscara: Tapete = BLANCO, Carta = NEGRO
    - Esto permite segmentar las cartas correctamente

USO:
    python3 scripts/1_calibrar_hsv.py
    
    - Ajusta los trackbars hasta lograr buena segmentación
    - Presiona 'q' para salir
    - Copia los valores mostrados a config/settings.py
"""

import sys
import os

# Agregar el directorio raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config.settings import get_rtsp_url

def nothing(x):
    """Función dummy para los trackbars."""
    pass

def calibrar_hsv():
    """
    Función principal de calibración HSV.
    
    Abre una ventana con trackbars para ajustar los valores HSV en tiempo real.
    Muestra 3 vistas:
        1. Original: Stream de la cámara
        2. Máscara: Resultado de la segmentación HSV
        3. Resultado: Frame original con máscara aplicada
    """
    
    # Conectar al stream RTSP
    rtsp_url = get_rtsp_url()
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print(f"❌ ERROR: No se pudo conectar al stream RTSP")
        print(f"   URL: {rtsp_url}")
        print(f"   Verifica que la app IP Webcam esté activa")
        return
    
    print("✅ Conexión RTSP exitosa")
    print("\n" + "="*60)
    print("INSTRUCCIONES DE CALIBRACIÓN")
    print("="*60)
    print("1. Coloca UNA carta sobre el tapete verde")
    print("2. Ajusta los trackbars hasta que:")
    print("   - La MÁSCARA muestre el tapete en BLANCO")
    print("   - La carta quede en NEGRO (completamente)")
    print("3. Presiona 'q' cuando estés satisfecho")
    print("4. Copia los valores mostrados a config/settings.py")
    print("="*60 + "\n")
    
    # Crear ventana con trackbars
    cv2.namedWindow('Calibracion HSV')
    
    # Valores iniciales
    cv2.createTrackbar('H_min', 'Calibracion HSV', 35, 179, nothing) 
    cv2.createTrackbar('H_max', 'Calibracion HSV', 105, 179, nothing) 
    cv2.createTrackbar('S_min', 'Calibracion HSV', 153, 255, nothing) 
    cv2.createTrackbar('S_max', 'Calibracion HSV', 255, 255, nothing)
    cv2.createTrackbar('V_min', 'Calibracion HSV', 0, 255, nothing)
    cv2.createTrackbar('V_max', 'Calibracion HSV', 255, 255, nothing)
    
    # Variables para estadísticas
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("⚠️  No se pudo recibir frame. Reintentando...")
            continue
        
        frame_count += 1
        
        # Convertir a HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Leer valores de los trackbars
        h_min = cv2.getTrackbarPos('H_min', 'Calibracion HSV')
        h_max = cv2.getTrackbarPos('H_max', 'Calibracion HSV')
        s_min = cv2.getTrackbarPos('S_min', 'Calibracion HSV')
        s_max = cv2.getTrackbarPos('S_max', 'Calibracion HSV')
        v_min = cv2.getTrackbarPos('V_min', 'Calibracion HSV')
        v_max = cv2.getTrackbarPos('V_max', 'Calibracion HSV')
        
        # Crear rangos
        lower_hsv = np.array([h_min, s_min, v_min])
        upper_hsv = np.array([h_max, s_max, v_max])
        
        # Crear máscara
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # Aplicar máscara al frame original
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Añadir información en pantalla
        info_text = f"Frame: {frame_count} | Presiona 'q' para salir"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Añadir guía visual en la máscara
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_color, "OBJETIVO: Tapete BLANCO, Carta NEGRA", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Calcular porcentaje de píxeles blancos (para verificar)
        white_pixels = np.sum(mask == 255)
        total_pixels = mask.shape[0] * mask.shape[1]
        white_percentage = (white_pixels / total_pixels) * 100
        
        stats_text = f"Fondo detectado: {white_percentage:.1f}%"
        cv2.putText(mask_color, stats_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mostrar las 3 vistas
        cv2.imshow('1. Original', frame)
        cv2.imshow('2. Mascara (OBJETIVO: Fondo BLANCO)', mask_color)
        cv2.imshow('3. Resultado', result)
        
        # Detectar tecla presionada
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Mostrar valores finales
            print("\n" + "="*60)
            print("VALORES CALIBRADOS (Copiar a config/settings.py)")
            print("="*60)
            print(f"LOWER_COLOR_FONDO = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"UPPER_COLOR_FONDO = np.array([{h_max}, {s_max}, {v_max}])")
            print("="*60)
            print("\n✅ Calibración completada")
            print(f"📊 Porcentaje de fondo detectado: {white_percentage:.1f}%")
            
            # Evaluación de calidad
            if 60 < white_percentage < 85:
                print("✅ Porcentaje óptimo para detección de cartas")
            elif white_percentage < 60:
                print("⚠️  Porcentaje bajo - puede que no detecte todo el tapete")
            else:
                print("⚠️  Porcentaje alto - puede detectar partes de las cartas")
            
            break
        
        elif key == ord('s'):
            # Guardar frame actual para debugging
            cv2.imwrite('debug_calibracion_original.jpg', frame)
            cv2.imwrite('debug_calibracion_mascara.jpg', mask)
            print("💾 Frames guardados para debugging")
    
    # Limpieza
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        calibrar_hsv()
    except KeyboardInterrupt:
        print("\n⚠️  Calibración interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()