# Reconocimiento de Cartas mediante Visión Artificial Clásica

**Proyecto de Procesamiento de Imágenes - Reconocimiento de Baraja Francesa**


## 📋 Descripción

Sistema de reconocimiento automático de cartas de poker (baraja francesa de 52 cartas) utilizando **únicamente técnicas clásicas de visión artificial**. El proyecto **NO utiliza** machine learning, redes neuronales ni clasificadores entrenados, cumpliendo estrictamente con las restricciones académicas establecidas.

### ✨ Características Principales

- ✅ **100% Visión Clásica**: Template matching, segmentación HSV, transformación de perspectiva
- ✅ **Tiempo Real**: 25-30 FPS de procesamiento
- ✅ **Alta Precisión**: >95% de tasa de éxito en condiciones controladas
- ✅ **Múltiples Cartas**: Detección simultánea de hasta 5 cartas
- ✅ **Rotación Invariante**: Funciona con cartas en cualquier orientación (0-360°)
- ✅ **Arquitectura Modular**: Código organizado y fácil de mantener

---

## 🎯 Requisitos del Proyecto

### Restricciones Técnicas (Obligatorias)

❌ **Prohibido**:
- Redes neuronales o deep learning
- Modelos pre-entrenados (CNN, YOLO, etc.)
- Clasificadores de machine learning (SVM, Random Forest, etc.)
- Librerías con funciones de reconocimiento basadas en aprendizaje

✅ **Permitido**:
- Transformaciones de color (RGB, HSV, LAB, etc.)
- Umbralización y segmentación
- Operaciones morfológicas
- Detección de bordes, contornos y esquinas
- Filtros espaciales y convoluciones
- Transformaciones geométricas
- Template matching mediante correlación
- Operaciones matriciales directas

---

## 🛠️ Tecnologías Utilizadas

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **Python** | 3.13 | Lenguaje principal |
| **OpenCV** | 4.12 | Procesamiento de imágenes y video |
| **NumPy** | 1.26+ | Operaciones matriciales |
| **IP Webcam** | - | Streaming RTSP desde tablet Android |

---

## 📦 Instalación

### 1. Clonar el Repositorio

```bash
git clone [URL_REPO]
cd proyecto_cartas
```

### 2. Instalar Dependencias

```bash
pip install opencv-python numpy --break-system-packages
```

### 3. Configurar Hardware

1. Instalar **IP Webcam** en tablet/smartphone Android
2. Conectar dispositivo a la misma red WiFi que el ordenador
3. Iniciar servidor de video en la app
4. Anotar la URL RTSP mostrada (ej: `rtsp://192.168.1.100:8080/h264.sdp`)
5. Actualizar `RTSP_URL` en `config/settings.py`

### 4. Preparar Tapete

- Usar cartulina o superficie **verde uniforme**
- Dimensiones recomendadas: A3 o superior
- Asegurar iluminación uniforme sin sombras directas

---

## Guía resumida
[Esta guía complementa la memoria técnica y el README, proporcionando
detalles específicos del código para facilitar el mantenimiento y
extensión del proyecto.](/retos_examenes/examen_parcial/GuiaTecnica.md)


## 🚀 Uso del Sistema

### Workflow Completo

El sistema se utiliza en 4 fases secuenciales:

#### **Fase 1: Calibración del Fondo Verde** 🎨

```bash
python scripts/1_calibrar_hsv.py
```

**Objetivo**: Encontrar valores HSV óptimos para segmentar el tapete verde.

**Instrucciones**:
1. Coloca una carta sobre el tapete
2. Ajusta los trackbars hasta que:
   - La máscara muestre el tapete en **BLANCO**
   - La carta quede en **NEGRO** (completamente)
3. Presiona `q` cuando estés satisfecho
4. **Copia los valores mostrados** a `config/settings.py`:
   ```python
   LOWER_COLOR_FONDO = np.array([H_min, S_min, V_min])
   UPPER_COLOR_FONDO = np.array([H_max, S_max, V_max])
   ```

---

#### **Fase 2: Captura de Imágenes de Referencia** 📸

```bash
python scripts/2_capturar_imagenes_referencia.py
```

**Objetivo**: Capturar al menos 1 imagen de cada una de las 52 cartas.

**Protocolo**:
1. Coloca **UNA** carta sobre el tapete
2. Centra la carta en el campo de visión
3. Presiona `s` para capturar
4. Introduce etiqueta (ej: `AS_PICAS`, `7_CORAZONES`, `K_DIAMANTES`)
5. Repite para las 52 cartas

**Formato de etiquetas**:
- **Valores**: AS, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K
- **Palos**: PICAS, CORAZONES, DIAMANTES, TREBOLES
- Formato: `VALOR_PALO` (ej: `10_TREBOLES`)

**Resultado**: Imágenes normalizadas guardadas en `data/imagenes_referencia/`

---

#### **Fase 3: Creación de Templates** 🎴

```bash
python scripts/3_crear_templates.py
```

**Objetivo**: Extraer templates limpios de valores y palos desde las imágenes de referencia.

**Proceso interactivo**:
1. Para cada carta, selecciona con el mouse:
   - **ROI del valor** (número/letra)
   - **ROI del palo** (símbolo ♠♥♦♣)
2. El sistema procesa y guarda automáticamente
3. Controles:
   - Click y arrastra = Seleccionar ROI
   - `c` = Confirmar y guardar
   - `r` = Reiniciar selección
   - `s` = Saltar imagen

**Resultado**: 
- 13 templates de valores en `data/templates/valores/`
- 4 templates de palos en `data/templates/palos/`

---

#### **Fase 4: Clasificación en Tiempo Real** 🎬

```bash
python scripts/5_clasificar_realtime.py
```

**Objetivo**: Sistema de reconocimiento en vivo.

**Controles**:
- `q` = Salir
- `m` = Cambiar modo (1 carta / múltiples cartas)
- `d` = Toggle debug (mostrar ROIs y scores)
- `p` = Pausar/Reanudar
- `s` = Capturar screenshot
- `r` = Reiniciar estadísticas

**Interfaz**:
- Panel superior: FPS, modo, estado
- Cartas detectadas: Contorno verde + etiqueta
- Panel inferior: Controles disponibles

---

## 📂 Estructura del Proyecto

```
proyecto_cartas/
│
├── config/
│   └── settings.py              # ⚙️ Configuración centralizada
│
├── src/
│   └── vision/
│       ├── preprocessing.py     # 🔍 Segmentación y normalización
│       ├── template_matching.py # 🎯 Template matching y scoring
│       └── classification.py    # 🧠 Pipeline de clasificación
│
├── scripts/
│   ├── 1_calibrar_hsv.py                 # 🎨 Calibración interactiva
│   ├── 2_capturar_imagenes_referencia.py # 📸 Captura de dataset
│   ├── 3_crear_templates.py              # 🎴 Extracción de templates
│   ├── 4_validar_templates.py            # ✅ Validación
│   └── 5_clasificar_realtime.py          # 🎬 Sistema en tiempo real
│
├── data/
│   ├── imagenes_referencia/    # Cartas capturadas (52+)
│   └── templates/
│       ├── valores/            # Templates AS, 2-10, J, Q, K
│       └── palos/              # Templates ♠ ♥ ♦ ♣
│
└── README.md
```

---

## 🔧 Configuración Avanzada

### Parámetros Clave en `config/settings.py`

#### Dimensiones de Carta Normalizada
```python
CARD_WIDTH = 200   # Ancho en píxeles
CARD_HEIGHT = 300  # Alto en píxeles (ratio 2:3)
```

#### ROI (Regiones de Interés)
```python
# ROI del valor (esquina superior izquierda)
ROI_CORNER_VALUE = (0, 3, 85, 50)  # (x, y, ancho, alto)
# Ancho=85px CRÍTICO para capturar '10' completo

# ROI del palo (debajo del valor)
ROI_CORNER_SUIT = (5, 50, 40, 40)  # Región cuadrada 40x40
```

#### Valores HSV del Tapete Verde
```python
LOWER_COLOR_FONDO = np.array([35, 153, 0])
UPPER_COLOR_FONDO = np.array([105, 255, 255])
# ⚠️ Estos valores son específicos del tapete usado
# Recalibrar si se cambia de superficie
```

#### Template Matching
```python
TEMPLATE_MATCHING_METHOD = 'TM_CCOEFF_NORMED'  # Correlación normalizada
TEMPLATE_MATCH_THRESHOLD = 0.35  # Umbral mínimo de confianza (0.0-1.0)
```

#### Filtros de Contornos
```python
MIN_CONTOUR_AREA = 5000  # Área mínima para carta válida (px²)
EPSILON_FACTOR = 0.03    # Factor aprox. poligonal (3% del perímetro)
```

---

## 🧪 Pipeline de Procesamiento

### 1. Preprocesamiento (`preprocessing.py`)

```
Frame BGR → HSV → Blur Gaussiano → Segmentación (inRange) →
Inversión máscara → Morfología (Close+Open) → Detección contornos →
Filtrado por área → Aproximación poligonal (4 lados) →
Transformación de perspectiva → Carta normalizada 200x300px
```

**Funciones clave**:
- `preprocess_and_warp()`: Detecta y normaliza una carta
- `detect_multiple_cards()`: Detecta múltiples cartas
- `is_red_card()`: Detecta color rojo vs negro (ratio BGR)
- `order_points()`: Ordena vértices del cuadrilátero

---

### 2. Template Matching (`template_matching.py`)

```
ROI binarizada → Match con 13 templates de valores (multi-escala) →
Scoring TM_CCOEFF_NORMED → Mejor match valor

ROI binarizada → Filtrado por color (rojo→♥♦, negro→♠♣) →
Match con templates de palos → Mejor match palo
```

**Funciones clave**:
- `match_value_templates()`: Compara ROI con todos los valores
- `match_suit_templates()`: Compara ROI con palos (filtrado por color)
- `match_template_multiscale()`: Matching en escalas 0.7-1.3
- `get_best_match()`: Selecciona resultado con mayor confianza

---

### 3. Clasificación (`classification.py`)

```
Carta normalizada → Extracción ROI valor y palo →
Detección de color (rojo/negro) → Template matching →
Validación de confianza (>0.5) →
Validación color-palo coherente →
Corrección si inconsistencia → Etiqueta final: VALOR_PALO
```

**Validaciones implementadas**:
1. **Umbral de confianza**: Valor y palo deben tener score >0.35
2. **Coherencia color-palo**: 
   - Si detecta rojo pero palo es negro → Re-clasifica entre ♥/♦
   - Si detecta negro pero palo es rojo → Re-clasifica entre ♠/♣

**Funciones clave**:
- `classify_card()`: Pipeline completo de clasificación
- `classify_multiple_cards()`: Clasifica múltiples cartas
- `format_classification_text()`: Formatea resultado para UI

---

## 📊 Métricas de Rendimiento

| Métrica | Valor | Condiciones |
|---------|-------|-------------|
| **Tasa de éxito** | >95% | Iluminación controlada, cartas limpias |
| **FPS** | 25-30 | Procesamiento en tiempo real |
| **Latencia RTSP** | ~200ms | Acceptable para aplicación |
| **Tiempo clasificación** | ~30ms/carta | Incluye preprocesamiento + matching |
| **Cartas simultáneas** | Hasta 5 | Sin oclusiones |
| **Precisión valores** | ~98% | Errores raros en 6 vs Q |
| **Precisión palos** | ~97% | Confusión ocasional ♠ vs ♣ |

---

## 🔬 Técnicas de Visión Artificial Empleadas

### Operaciones de Imagen

| Técnica | Función OpenCV | Parámetros Clave |
|---------|----------------|------------------|
| **Conversión espacios de color** | `cv2.cvtColor()` | BGR → HSV |
| **Filtrado Gaussiano** | `cv2.GaussianBlur()` | kernel=5x5, σ=0 |
| **Segmentación por umbral** | `cv2.inRange()` | lower_hsv, upper_hsv |
| **Morfología matemática** | `cv2.morphologyEx()` | MORPH_CLOSE, MORPH_OPEN |
| **Detección de contornos** | `cv2.findContours()` | RETR_EXTERNAL |
| **Aproximación poligonal** | `cv2.approxPolyDP()` | ε=3% perímetro |
| **Transformación proyectiva** | `cv2.getPerspectiveTransform()`<br/>`cv2.warpPerspective()` | 4 puntos → rectángulo |
| **Template matching** | `cv2.matchTemplate()` | TM_CCOEFF_NORMED |
| **Binarización** | `cv2.threshold()` | threshold=150, THRESH_BINARY_INV |

---

## 🐛 Troubleshooting

### Problema: No se detecta el tapete correctamente

**Solución**: Recalibrar valores HSV
```bash
python scripts/1_calibrar_hsv.py
```
Ajusta hasta que el tapete quede completamente blanco en la máscara.

---

### Problema: Carta '10' se clasifica como '6'

**Causa**: ROI de valor demasiado estrecha, cortando el '1'

**Solución**: Verificar en `config/settings.py`:
```python
ROI_CORNER_VALUE = (0, 3, 85, 50)  # Ancho debe ser ≥85px
```

---

### Problema: Confusión entre palos rojos (♥ vs ♦)

**Causa**: Templates de palos muy similares

**Solución**: 
1. Recrear templates con mayor contraste
```bash
python scripts/3b_recrear_solo_palos.py
```
2. Seleccionar ROI del palo central (más grande y claro)

---

### Problema: No se conecta al stream RTSP

**Verificar**:
1. Tablet y PC en misma red WiFi
2. App IP Webcam activa y mostrando URL
3. URL correcta en `config/settings.py`:
```python
RTSP_URL = 'rtsp://[IP_TABLET]:8080/h264.sdp'
```
4. Firewall no bloqueando puerto 8080

**Test manual**:
```bash
python test_conexion.py
```

---

## 🎓 Conceptos Aprendidos

### Visión Artificial Clásica
- Segmentación por color en espacio HSV
- Detección de contornos y aproximación poligonal
- Transformación de perspectiva (homografía)
- Template matching con correlación cruzada

### Procesamiento de Imágenes
- Operaciones morfológicas (erosión, dilatación, apertura, cierre)
- Filtrado espacial (Gaussiano)
- Binarización adaptativa
- Análisis de ROI (Regiones de Interés)

### Diseño de Software
- Arquitectura modular y separación de responsabilidades
- Configuración centralizada
- Desarrollo iterativo con herramientas de debugging
- Documentación de código

---

## 🚧 Limitaciones Conocidas

1. **Dependencia de iluminación**: Calibración necesaria para cada entorno
2. **Oclusiones parciales**: No maneja cartas tapadas
3. **Desgaste de cartas**: Bordes doblados afectan detección de cuadrilátero
4. **Símbolos similares**: ♠ y ♣ pueden confundirse en ángulos extremos
5. **Tapete específico**: Valores HSV calibrados para cartulina verde usada

---

## 🔮 Mejoras Futuras

- [ ] Auto-calibración adaptativa de HSV basada en histograma
- [ ] Detección de oclusiones parciales
- [ ] Tracking temporal de cartas entre frames
- [ ] Multi-threading para mayor FPS
- [ ] Soporte para mazos españoles/tarot
- [ ] Compensación automática de iluminación

---

## 📄 Licencia

Proyecto educativo desarrollado para el curso de Visión Artificial.

---

## 👥 Autor

**Natalia Cruz Babbar**  
Proyecto de Visión Artificial - 2025

---

## 📚 Referencias

- [OpenCV Documentation](https://docs.opencv.org/)
- [Template Matching Tutorial](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)
- [Contour Detection](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
- [Perspective Transformation](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)

---