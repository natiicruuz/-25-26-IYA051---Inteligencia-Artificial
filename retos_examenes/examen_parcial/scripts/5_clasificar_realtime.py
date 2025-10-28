"""
Script para clasificación de cartas en tiempo real.

OBJETIVO:
    Detectar y clasificar cartas en tiempo real usando la cámara RTSP
    mediante técnicas clásicas de visión artificial (template matching).

FUNCIONALIDADES:
    - Detección automática de cartas en el tapete
    - Clasificación usando templates
    - Visualización en tiempo real con etiquetas
    - Estadísticas de confianza
    - Modo de múltiples cartas (opcional)

CONTROLES:
    - 'q' = Salir
    - 'm' = Cambiar modo (1 carta / múltiples cartas)
    - 'd' = Toggle debug (mostrar ROIs y scores)
    - 'p' = Pausar/Reanudar
    - 's' = Capturar screenshot
    - 'r' = Reiniciar estadísticas

REQUISITOS:
    - Templates creados (scripts/3_crear_templates.py)
    - Calibración HSV configurada
    - Conexión RTSP activa

USO:
    python3 scripts/5_clasificar_realtime.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time
from collections import deque

from config.settings import (
    get_rtsp_url,
    CARD_WIDTH, CARD_HEIGHT,
    COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_WHITE
)

from src.vision.preprocessing import (
    preprocess_and_warp,
    detect_multiple_cards,
    visualize_detection
)

from src.vision.classification import (
    classify_card,
    classify_multiple_cards,
    get_classification_stats,
    format_classification_text
)

from src.vision.template_matching import get_template_library


class CardRecognitionSystem:
    """Sistema de reconocimiento de cartas en tiempo real."""
    
    def __init__(self, rtsp_url):
        """
        Inicializa el sistema de reconocimiento.
        
        Args:
            rtsp_url (str): URL del stream RTSP
        """
        self.rtsp_url = rtsp_url
        self.cap = None
        
        # Estado del sistema
        self.running = False
        self.paused = False
        self.debug_mode = False
        self.multi_card_mode = False
        
        # Estadísticas
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        
        # Historial de clasificaciones (para suavizado)
        self.classification_history = deque(maxlen=5)
        
        # Estadísticas de sesión
        self.session_stats = {
            'total_clasificaciones': 0,
            'clasificaciones_validas': 0,
            'clasificaciones_invalidas': 0,
            'cartas_detectadas': set(),
            'tiempo_inicio': time.time()
        }
        
        # Cargar templates
        print("📚 Cargando biblioteca de templates...")
        self.template_library = get_template_library()
        
        if not self.template_library.is_loaded():
            raise Exception("❌ No se pudieron cargar los templates. Ejecuta scripts/3_crear_templates.py")
        
        print("✅ Templates cargados correctamente")
    
    def conectar_camara(self):
        """Conecta a la cámara RTSP."""
        print(f"📹 Conectando a: {self.rtsp_url}")
        
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        if not self.cap.isOpened():
            raise Exception(f"❌ No se pudo conectar al stream RTSP: {self.rtsp_url}")
        
        print("✅ Conexión RTSP exitosa")
        
        # Configurar buffer mínimo para reducir latencia
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    def calcular_fps(self):
        """Calcula los FPS del sistema."""
        self.fps_counter += 1
        tiempo_actual = time.time()
        
        if tiempo_actual - self.last_fps_time >= 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = tiempo_actual
    
    def clasificar_frame(self, frame):
        """
        Procesa y clasifica un frame.
        
        Args:
            frame (np.ndarray): Frame de la cámara
        
        Returns:
            tuple: (frame_procesado, resultados_clasificacion)
        """
        if self.multi_card_mode:
            # Modo múltiples cartas
            cards = detect_multiple_cards(frame, debug=False)
            
            if cards:
                results = classify_multiple_cards(cards, debug=self.debug_mode)
                
                # Dibujar resultados
                for result in results:
                    if result['valido']:
                        # Dibujar contorno verde
                        cv2.drawContours(frame, [result['contour']], -1, COLOR_GREEN, 3)
                        
                        # Dibujar etiqueta
                        cx, cy = result['center']
                        texto = format_classification_text(result, include_confidence=True)
                        
                        # Fondo semi-transparente para el texto
                        (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (cx - w//2 - 5, cy - h - 10), 
                                     (cx + w//2 + 5, cy + 5), (0, 0, 0), -1)
                        
                        cv2.putText(frame, texto, (cx - w//2, cy),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
                        
                        # Actualizar estadísticas
                        self.session_stats['cartas_detectadas'].add(result['carta'])
                    else:
                        # Contorno rojo para cartas no identificadas
                        cv2.drawContours(frame, [result['contour']], -1, COLOR_RED, 3)
                
                return frame, results
            else:
                return frame, []
        
        else:
            # Modo una sola carta
            warped_card, contour, _ = preprocess_and_warp(frame, debug=False)
            
            if warped_card is not None:
                # Clasificar
                result = classify_card(warped_card, debug=self.debug_mode)
                
                if result['valido']:
                    # Dibujar contorno verde
                    cv2.drawContours(frame, [contour], -1, COLOR_GREEN, 3)
                    
                    # Calcular centro del contorno
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = frame.shape[1]//2, frame.shape[0]//2
                    
                    # Texto de clasificación
                    texto = format_classification_text(result, include_confidence=True)
                    
                    # Fondo para el texto
                    (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                    cv2.rectangle(frame, (cx - w//2 - 10, cy - h - 20), 
                                 (cx + w//2 + 10, cy + 10), (0, 0, 0), -1)
                    
                    cv2.putText(frame, texto, (cx - w//2, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)
                    
                    # Suavizado temporal (voting)
                    self.classification_history.append(result['carta'])
                    
                    # Actualizar estadísticas
                    self.session_stats['cartas_detectadas'].add(result['carta'])
                else:
                    # Contorno rojo
                    cv2.drawContours(frame, [contour], -1, COLOR_RED, 3)
                
                return frame, [result]
            else:
                return frame, []
    
    def dibujar_interfaz(self, frame):
        """
        Dibuja la interfaz de usuario en el frame.
        
        Args:
            frame (np.ndarray): Frame a decorar
        
        Returns:
            np.ndarray: Frame con interfaz dibujada
        """
        h, w = frame.shape[:2]
        
        # Panel superior: Información del sistema
        panel_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Título
        cv2.putText(frame, "RECONOCIMIENTO DE CARTAS - VISION ARTIFICIAL CLASICA", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_YELLOW, 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
        
        # Modo
        modo_texto = "MULTIPLES CARTAS" if self.multi_card_mode else "UNA CARTA"
        cv2.putText(frame, f"Modo: {modo_texto}", (150, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
        
        # Debug
        if self.debug_mode:
            cv2.putText(frame, "DEBUG: ON", (400, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Pausa
        if self.paused:
            cv2.putText(frame, "PAUSADO", (550, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
        
        # Estadísticas de sesión
        stats_y = 90
        cv2.putText(frame, f"Cartas unicas: {len(self.session_stats['cartas_detectadas'])}", 
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
        
        tiempo_sesion = int(time.time() - self.session_stats['tiempo_inicio'])
        mins = tiempo_sesion // 60
        secs = tiempo_sesion % 60
        cv2.putText(frame, f"Tiempo: {mins:02d}:{secs:02d}", 
                   (250, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
        
        # Panel inferior: Controles
        controles_y = h - 30
        controles = "q:Salir | m:Modo | d:Debug | p:Pausa | s:Screenshot | r:Reset"
        cv2.putText(frame, controles, (10, controles_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1)
        
        return frame
    
    def guardar_screenshot(self, frame):
        """Guarda un screenshot del frame actual."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        filepath = os.path.join("data", filename)
        
        cv2.imwrite(filepath, frame)
        print(f"📸 Screenshot guardado: {filepath}")
    
    def reiniciar_estadisticas(self):
        """Reinicia las estadísticas de la sesión."""
        self.session_stats = {
            'total_clasificaciones': 0,
            'clasificaciones_validas': 0,
            'clasificaciones_invalidas': 0,
            'cartas_detectadas': set(),
            'tiempo_inicio': time.time()
        }
        self.classification_history.clear()
        print("🔄 Estadísticas reiniciadas")
    
    def mostrar_resumen_final(self):
        """Muestra un resumen de la sesión al finalizar."""
        print("\n" + "="*60)
        print("RESUMEN DE LA SESIÓN")
        print("="*60)
        
        tiempo_total = int(time.time() - self.session_stats['tiempo_inicio'])
        mins = tiempo_total // 60
        secs = tiempo_total % 60
        
        print(f"⏱️  Duración: {mins} minutos, {secs} segundos")
        print(f"📊 Cartas únicas detectadas: {len(self.session_stats['cartas_detectadas'])}")
        
        if self.session_stats['cartas_detectadas']:
            print(f"\n🎴 Cartas identificadas:")
            for carta in sorted(self.session_stats['cartas_detectadas']):
                print(f"   - {carta}")
        
        print("="*60)
    
    def run(self):
        """Bucle principal del sistema."""
        try:
            self.conectar_camara()
            
            print("\n" + "="*60)
            print("SISTEMA DE RECONOCIMIENTO EN TIEMPO REAL")
            print("="*60)
            print("\n📝 CONTROLES:")
            print("  q = Salir")
            print("  m = Cambiar modo (1 carta / múltiples)")
            print("  d = Toggle debug")
            print("  p = Pausar/Reanudar")
            print("  s = Capturar screenshot")
            print("  r = Reiniciar estadísticas")
            print("\n🚀 Iniciando reconocimiento...\n")
            print("="*60 + "\n")
            
            self.running = True
            
            while self.running:
                if not self.paused:
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        print("⚠️  No se pudo recibir frame")
                        time.sleep(0.1)
                        continue
                    
                    self.frame_count += 1
                    self.calcular_fps()
                    
                    # Clasificar frame
                    frame_procesado, results = self.clasificar_frame(frame)
                    
                    # Actualizar estadísticas
                    if results:
                        self.session_stats['total_clasificaciones'] += len(results)
                        self.session_stats['clasificaciones_validas'] += sum(
                            1 for r in results if r['valido']
                        )
                        self.session_stats['clasificaciones_invalidas'] += sum(
                            1 for r in results if not r['valido']
                        )
                    
                    # Dibujar interfaz
                    frame_final = self.dibujar_interfaz(frame_procesado)
                    
                    # Mostrar frame
                    cv2.imshow('Reconocimiento de Cartas', frame_final)
                else:
                    # Modo pausa: solo mostrar el último frame
                    cv2.imshow('Reconocimiento de Cartas', frame_final)
                
                # Procesar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Saliendo...")
                    self.running = False
                
                elif key == ord('m'):
                    self.multi_card_mode = not self.multi_card_mode
                    modo = "MÚLTIPLES CARTAS" if self.multi_card_mode else "UNA CARTA"
                    print(f"🔄 Modo cambiado a: {modo}")
                
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    estado = "ACTIVADO" if self.debug_mode else "DESACTIVADO"
                    print(f"🔍 Debug {estado}")
                
                elif key == ord('p'):
                    self.paused = not self.paused
                    estado = "PAUSADO" if self.paused else "REANUDADO"
                    print(f"⏸️  {estado}")
                
                elif key == ord('s'):
                    self.guardar_screenshot(frame_final)
                
                elif key == ord('r'):
                    self.reiniciar_estadisticas()
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrumpido por el usuario")
        
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpia recursos al finalizar."""
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        self.mostrar_resumen_final()


def main():
    """Función principal."""
    print("="*60)
    print("RECONOCIMIENTO DE CARTAS - VISIÓN ARTIFICIAL CLÁSICA")
    print("="*60)
    print("\nTécnicas utilizadas:")
    print("  ✅ Segmentación por color (HSV)")
    print("  ✅ Detección de contornos")
    print("  ✅ Transformación de perspectiva")
    print("  ✅ Template matching (correlación normalizada)")
    print("  ✅ Validación por reglas lógicas")
    print("  ❌ SIN Machine Learning")
    print("  ❌ SIN Redes Neuronales")
    print("="*60 + "\n")
    
    # Obtener URL RTSP
    rtsp_url = get_rtsp_url()
    
    # Crear y ejecutar sistema
    system = CardRecognitionSystem(rtsp_url)
    system.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR FATAL: {e}")
        import traceback
        traceback.print_exc()