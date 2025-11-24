# Guía de Documentación del Código Fuente

## 📝 Resumen de Archivos Principales

### config/settings.py
**Propósito**: Centralización de todos los parámetros configurables del sistema.

**Contenido clave**:
- URLs y conexiones (RTSP_URL)
- Dimensiones normalizadas (CARD_WIDTH, CARD_HEIGHT)
- Coordenadas de ROI (ROI_CORNER_VALUE, ROI_CORNER_SUIT)
- Rangos HSV calibrados (LOWER_COLOR_FONDO, UPPER_COLOR_FONDO)
- Parámetros de procesamiento (BLUR_KERNEL_SIZE, MIN_CONTOUR_AREA)
- Configuración de templates (TEMPLATE_VALUE_SIZE, TEMPLATE_SUIT_SIZE)
- Umbrales de matching (TEMPLATE_MATCH_THRESHOLD)
- Definiciones de cartas (CARD_VALUES, CARD_SUITS)

**Funciones auxiliares**:
```python
get_rtsp_url()           # Retorna URL RTSP configurada
get_card_label(v, p)     # Genera etiqueta "VALOR_PALO"
is_valid_card_label(l)   # Valida formato de etiqueta
print_config_summary()   # Imprime configuración actual
```

---

### src/vision/preprocessing.py
**Propósito**: Preprocesamiento de imágenes y detección de cartas.

**Funciones principales**:

#### `order_points(pts)` 
Ordena 4 puntos de un cuadrilátero en orden estándar.
- **Input**: Array de 4 puntos (cualquier orden)
- **Output**: Array ordenado [TL, TR, BR, BL]
- **Método**: Suma y diferencia de coordenadas
- **Uso**: Preparar puntos para transformación de perspectiva

#### `preprocess_and_warp(frame, debug=False)`
Pipeline completo de detección y normalización de UNA carta.
- **Input**: Frame BGR de cámara
- **Output**: (carta_normalizada, contorno, debug_images)
- **Pasos**:
  1. Conversión BGR → HSV
  2. Blur gaussiano (reducir ruido)
  3. Segmentación por color HSV
  4. Inversión de máscara (tapete→negro, carta→blanco)
  5. Morfología (Close+Open para limpieza)
  6. Detección de contornos (RETR_EXTERNAL)
  7. Filtrado por área (≥5000px²)
  8. Aproximación poligonal (buscar 4 lados)
  9. Ordenamiento de puntos
  10. Transformación de perspectiva

#### `detect_multiple_cards(frame, debug=False)`
Detecta MÚLTIPLES cartas en el mismo frame.
- **Input**: Frame BGR
- **Output**: Lista de tuplas (carta_normalizada, contorno, centro)
- **Diferencia con preprocess_and_warp**: Procesa TODOS los contornos válidos

#### `is_red_card(roi)`
Determina si ROI contiene símbolo rojo o negro.
- **Input**: ROI BGR (región del palo)
- **Output**: True si es roja, False si es negra
- **Método**: Ratios BGR + umbrales absolutos
- **Criterios** (requiere ≥2 de 3):
  1. ratio R/G > 1.03
  2. ratio R/B > 1.05
  3. Canal R absoluto > 210

#### `binarize_roi(roi, threshold=150)`
Convierte ROI a imagen binaria para template matching.
- **Input**: ROI BGR o grayscale
- **Output**: Imagen binaria (símbolos oscuros → blancos)
- **Método**: THRESH_BINARY_INV con umbral=150

#### `extract_roi_region(warped_card, roi_coords)`
Extrae región de interés de carta normalizada.
- **Input**: Carta 200x300, coordenadas (x,y,w,h)
- **Output**: ROI recortada

---

### src/vision/template_matching.py
**Propósito**: Gestión de templates y correlación cruzada.

**Clase TemplateLibrary**:
Biblioteca singleton que carga y gestiona templates.

```python
library = get_template_library()  # Instancia global
library.is_loaded()               # Verifica carga exitosa
library.get_value_template(valor) # Obtiene template de valor
library.get_suit_template(palo)   # Obtiene template de palo
```

**Funciones de matching**:

#### `match_template(roi, template, method=TM_CCOEFF_NORMED)`
Realiza template matching básico.
- **Input**: ROI y template (ambos grayscale)
- **Output**: (score, location)
- **Método**: cv2.matchTemplate con TM_CCOEFF_NORMED
- **Score**: 0.0-1.0 (mayor = mejor match)

#### `match_template_multiscale(roi, template, scales=[...])`
Template matching en múltiples escalas.
- **Input**: ROI, template, lista de escalas
- **Escalas default**: [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
- **Output**: (best_score, best_scale, best_location)
- **Uso**: Compensa variaciones de tamaño en símbolos

#### `match_value_templates(roi_valor)`
Compara ROI con todos los templates de valores.
- **Input**: ROI del valor (esquina superior izquierda)
- **Output**: Dict {valor: score} para AS, 2-10, J, Q, K
- **Proceso**:
  1. Convierte ROI a grayscale
  2. Binariza (threshold=150, INV)
  3. Hace matching multi-escala con cada template
  4. Retorna scores de todos los valores

#### `match_suit_templates(roi_palo)`
Compara ROI con templates de palos (con filtro de color).
- **Input**: ROI del palo (debajo del valor)
- **Output**: Dict {palo: score} para ♠♥♦♣
- **Optimización**: Solo compara con palos del color detectado
  - Rojo → solo CORAZONES y DIAMANTES
  - Negro → solo PICAS y TREBOLES
- **Efecto**: Reduce falsos positivos al 50%

#### `get_best_match(scores, threshold=0.35)`
Selecciona el mejor match de un diccionario de scores.
- **Input**: Dict {etiqueta: score}, umbral mínimo
- **Output**: (mejor_etiqueta, mejor_score) o (None, 0.0)
- **Validación**: Score debe superar threshold

---

### src/vision/classification.py
**Propósito**: Pipeline completo de clasificación con validación.

#### `classify_card(warped_card, debug=False)`
Función principal de clasificación.

**Input**: Carta normalizada 200x300px

**Output**: Dict con estructura:
```python
{
    'carta': 'AS_PICAS' o 'DESCONOCIDA',
    'valor': 'AS',
    'palo': 'PICAS',
    'confianza_valor': 0.95,
    'confianza_palo': 0.88,
    'color_detectado': 'rojo' o 'negro',
    'valido': True/False
}
```

**Pipeline**:
1. Extracción de ROIs (valor y palo)
2. Detección de color (rojo/negro)
3. Template matching de valores
4. Template matching de palos (filtrado por color)
5. Obtención de mejores matches
6. **VALIDACIÓN 1**: Color coherente con palo
   - Si inconsistencia → Re-clasifica palo dentro del color correcto
7. **VALIDACIÓN 2**: Confianza mínima (≥0.5)
8. Construcción de etiqueta final

**Reglas de validación**:
```python
if SUIT_COLORS[palo] == 'rojo' and not es_roja:
    # Corregir: elegir mejor entre CORAZONES/DIAMANTES
    
if SUIT_COLORS[palo] == 'negro' and es_roja:
    # Corregir: elegir mejor entre PICAS/TREBOLES
```

#### `classify_multiple_cards(cards_list, debug=False)`
Clasifica múltiples cartas.
- **Input**: Lista de tuplas de detect_multiple_cards()
- **Output**: Lista de resultados de clasificación

---

## 🔍 Scripts Interactivos

### 1_calibrar_hsv.py
**Propósito**: Calibración interactiva de valores HSV del tapete.

**Flujo**:
1. Conecta a stream RTSP
2. Crea ventana con 6 trackbars (H_min, H_max, S_min, S_max, V_min, V_max)
3. En tiempo real:
   - Convierte frame a HSV
   - Aplica valores de trackbars
   - Muestra máscara resultante
4. Usuario ajusta hasta lograr:
   - Tapete = BLANCO
   - Carta = NEGRO
5. Al presionar 'q', imprime valores finales para copiar a settings.py

**Funciones clave**:
```python
nothing(x)           # Callback dummy para trackbars
calibrar_hsv()       # Función principal
```

---

### 2_capturar_imagenes_referencia.py
**Propósito**: Captura de dataset de 52 cartas.

**Funciones**:

#### `mostrar_instrucciones()`
Imprime ayuda en consola.

#### `mostrar_progreso()`
Calcula y muestra:
- Cartas únicas capturadas
- Total de imágenes
- Cartas faltantes (listadas)

#### `validar_etiqueta(etiqueta)`
Verifica formato correcto:
- Debe contener exactamente un '_'
- Valor debe estar en CARD_VALUES
- Palo debe estar en CARD_SUITS

#### `obtener_siguiente_numero(etiqueta)`
Encuentra próximo número disponible.
- Si existe AS_PICAS_0.jpg → retorna 1
- Si no existe ninguna → retorna 0

**Flujo principal**:
```python
while True:
    frame = cap.read()
    warped_card, contour = preprocess_and_warp(frame)
    
    if warped_card detectada:
        dibujar contorno verde
        mostrar carta normalizada
    else:
        dibujar contorno rojo (si existe)
    
    if tecla == 's' y carta detectada:
        solicitar etiqueta
        validar etiqueta
        guardar imagen normalizada
        actualizar progreso
```

---

### 3_crear_templates.py
**Propósito**: Extracción interactiva de templates.

**Funciones**:

#### `mouse_callback(event, x, y, flags, param)`
Maneja eventos del mouse para selección de ROI:
- LBUTTONDOWN: Inicia selección
- MOUSEMOVE: Actualiza rectángulo
- LBUTTONUP: Finaliza selección

#### `seleccionar_roi(imagen, titulo)`
Interfaz interactiva para seleccionar región.
- **Controles**:
  - Click+drag: Seleccionar
  - 'c': Confirmar
  - 'r': Reiniciar
  - 's': Saltar
  - 'q': Salir
- **Output**: ROI recortada o None

#### `procesar_roi_a_template(roi, threshold=150)`
Limpia ROI para crear template:
1. Convertir a grayscale
2. Binarizar (THRESH_BINARY_INV)
3. Morfología (Close+Open)
4. Encontrar contorno más grande
5. Recortar bounding box
6. Añadir padding (5px)
7. Retornar template limpio

#### `guardar_template(template, tipo, etiqueta, target_size)`
Redimensiona y guarda:
- Valores → TEMPLATE_VALUE_SIZE (30x50)
- Palos → TEMPLATE_SUIT_SIZE (40x40)

**Flujo principal**:
```python
for cada imagen de referencia:
    cargar imagen
    mostrar imagen completa
    
    # Valor
    roi_valor = seleccionar_roi("Seleccionar VALOR")
    if roi_valor:
        template_valor = procesar_roi_a_template(roi_valor)
        guardar_template(template_valor, 'valor', ...)
    
    # Palo
    roi_palo = seleccionar_roi("Seleccionar PALO")
    if roi_palo:
        template_palo = procesar_roi_a_template(roi_palo)
        guardar_template(template_palo, 'palo', ...)
```

---

### 5_clasificar_realtime.py
**Propósito**: Sistema de reconocimiento en tiempo real.

**Clase CardRecognitionSystem**:

#### `__init__(rtsp_url)`
Inicializa sistema:
- Carga templates
- Configura variables de estado
- Inicializa estadísticas

#### `conectar_camara()`
Establece conexión RTSP con buffer mínimo.

#### `calcular_fps()`
Calcula frames por segundo en ventana de 1 segundo.

#### `clasificar_frame(frame)`
Procesa frame completo:

**Modo single card**:
```python
warped_card, contour = preprocess_and_warp(frame)
if warped_card:
    result = classify_card(warped_card)
    if result['valido']:
        dibujar contorno verde
        mostrar etiqueta
        actualizar estadísticas
```

**Modo multi card**:
```python
cards = detect_multiple_cards(frame)
results = classify_multiple_cards(cards)
for cada result:
    dibujar contorno (verde si válido, rojo si no)
    mostrar etiqueta en centro
```

#### `dibujar_interfaz(frame)`
Renderiza UI con información:
- Panel superior: Título, FPS, modo, debug, pausa
- Estadísticas: Cartas únicas, tiempo de sesión
- Panel inferior: Controles disponibles

#### `run()`
Bucle principal:
```python
while running:
    if not paused:
        frame = capturar()
        frame_procesado, results = clasificar_frame(frame)
        actualizar_estadisticas(results)
        frame_final = dibujar_interfaz(frame_procesado)
    
    mostrar(frame_final)
    procesar_teclas()
```

**Controles implementados**:
- `q`: Salir (muestra resumen)
- `m`: Toggle modo single/multi
- `d`: Toggle debug
- `p`: Pausar/reanudar
- `s`: Screenshot
- `r`: Reset estadísticas

---

## 📊 Flujo de Datos Completo

```
Frame RTSP (BGR 1280x720)
    ↓
[preprocessing.preprocess_and_warp]
    ↓
Carta normalizada (BGR 200x300)
    ↓
[classification.classify_card]
    ↓
ROI valor (0,3,85,50) + ROI palo (5,50,40,40)
    ↓
[preprocessing.is_red_card] → Color detectado
    ↓
[template_matching.match_value_templates] → Scores valores
[template_matching.match_suit_templates]  → Scores palos (filtrados)
    ↓
[template_matching.get_best_match] × 2
    ↓
Mejor valor + Mejor palo
    ↓
[Validaciones en classification.py]
  - Coherencia color-palo
  - Confianza mínima
    ↓
Resultado: {carta, valor, palo, confianzas, valido}
```

---

## 🎯 Decisiones de Diseño Clave

### 1. ¿Por qué ROI_CORNER_VALUE con ancho=85px?
**Problema**: El '10' tiene dos dígitos ('1' y '0').
**Solución**: Ancho de 40px cortaba el '1', dejando solo '0' → clasificado como '6'.
**Fix**: Aumentar a 85px captura ambos dígitos completos.

### 2. ¿Por qué método de ratios BGR para detectar rojo?
**Problema**: HSV fallaba en distinguir rojos de cartas.
**Causa**: Los rojos de impresión no son rojos puros (contienen algo de azul/verde).
**Solución**: Comparar ratios R/G y R/B en lugar de rangos absolutos HSV.

### 3. ¿Por qué filtrar palos por color antes de matching?
**Beneficio**: Reduce espacio de búsqueda al 50%.
- Rojo → solo 2 palos (♥♦) en vez de 4
- Negro → solo 2 palos (♠♣) en vez de 4
**Resultado**: Menos falsos positivos, mayor confianza.

### 4. ¿Por qué validación cruzada color-palo?
**Problema**: Template matching podía dar score alto a palo incorrecto.
**Ejemplo**: Detecta rojo + matching dice "PICAS" (imposible).
**Solución**: Si inconsistencia, re-clasifica palo dentro del color correcto.

### 5. ¿Por qué TM_CCOEFF_NORMED en vez de otros métodos?
**Ventajas**:
- Normalizado (rango 0-1 predecible)
- Robusto a cambios de iluminación
- Mayor score = mejor match (intuitivo)
**Alternativas descartadas**:
- TM_SQDIFF: Menor = mejor (confuso)
- TM_CCORR: No normalizado (scores variables)

---

## 🧪 Testing y Debugging

### Validar templates cargados
```bash
python scripts/4_validar_templates.py
```
Output esperado:
```
✅ Templates cargados correctamente
   Valores: 13/13
   Palos: 4/4
```

### Debug de clasificación
En `5_clasificar_realtime.py`, presionar `d` para activar modo debug:
- Muestra ROIs extraídas en ventanas separadas
- Imprime scores de todos los valores/palos en consola
- Permite identificar por qué una carta fue mal clasificada

### Test manual de detección de color
```python
from src.vision.preprocessing import is_red_card
import cv2

roi = cv2.imread('roi_palo.jpg')
resultado = is_red_card(roi)
print(f"Es roja: {resultado}")
```

---

## 📝 Buenas Prácticas Implementadas

1. **Configuración centralizada**: Todos los parámetros en `settings.py`
2. **Separación de responsabilidades**: Cada módulo tiene propósito único
3. **Docstrings completos**: Todas las funciones documentadas
4. **Manejo de errores**: Try-catch en scripts principales
5. **Mensajes informativos**: Feedback claro al usuario (emoji + texto)
6. **Validación de entrada**: Verificar etiquetas, archivos, conexiones
7. **Logging de estadísticas**: Tracking de FPS, confianzas, cartas detectadas
8. **Código reutilizable**: Funciones genéricas (order_points, binarize_roi)

---

## 🔧 Parámetros Críticos para Ajustar

Si el sistema no funciona bien, ajustar en orden de prioridad:

1. **HSV del tapete** (1_calibrar_hsv.py)
   - Más importante: Asegurar segmentación limpia

2. **ROI_CORNER_VALUE ancho** (settings.py)
   - Si '10' se confunde con '6' → aumentar ancho

3. **TEMPLATE_MATCH_THRESHOLD** (settings.py)
   - Muy alto (>0.5) → muchas cartas no identificadas
   - Muy bajo (<0.2) → muchos falsos positivos
   - Óptimo: 0.35

4. **MIN_CONTOUR_AREA** (settings.py)
   - Muy alto → cartas pequeñas no detectadas
   - Muy bajo → ruido detectado como cartas

5. **Templates de palos** (3_crear_templates.py)
   - Si confusión ♠↔♣ o ♥↔♦ → recrear templates con mejor contraste


