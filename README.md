# 🔤 Sistema OCR - Reconocimiento de Caracteres Manuscritos

**Trabajo Final de Inteligencia Artificial - Curso 2024-2025**  
**Universidad Europea del Atlántico**

Sistema de OCR implementado desde cero para reconocer texto manuscrito sin usar bibliotecas especializadas de OCR.

---

## 📋 Resumen del Proyecto

### Objetivo
Desarrollar un sistema OCR capaz de reconocer caracteres manuscritos individuales y palabras completas sin usar bibliotecas especializadas de OCR (como Tesseract, EasyOCR, Google Vision API).

### Funcionalidades Implementadas

#### ✅ Obligatorias
- **Reconocimiento de caracteres manuscritos**: Letras (A-Z, a-z) y números (0-9)
- **Reconocimiento de palabras**: Segmentación automática por proyección vertical y predicción letra por letra
- **Arquitectura CNN propia**: Red neuronal convolucional entrenada desde cero con PyTorch

#### ⚠️ Parcialmente Implementadas
- **Caracteres españoles (Ñ, ñ)**: Reconocimiento mediante post-procesamiento con diccionario

---

## 🏗️ Arquitectura del Sistema

### Pipeline Completo

```
Imagen de entrada
      ↓
Preprocesamiento (binarización, normalización)
      ↓
Segmentación de palabras (operaciones morfológicas)
      ↓
Segmentación de letras (proyección vertical)
      ↓
Clasificación CNN (62 clases)
      ↓
Post-procesamiento (corrección Ñ/ñ)
      ↓
Texto digital
```

### Modelo CNN

```
Input (1, 28, 28)
     ↓
Conv2d(1→32) + ReLU + BatchNorm + MaxPool + Dropout(0.3)
     ↓
Conv2d(32→64) + ReLU + BatchNorm + MaxPool + Dropout(0.3)
     ↓
Conv2d(64→96) + ReLU + BatchNorm + MaxPool + Dropout(0.3)
     ↓
Flatten → 864
     ↓
Linear(864→320) + ReLU + Dropout(0.5)
     ↓
Linear(320→160) + ReLU + Dropout(0.5)
     ↓
Output: 62 clases
```

**Hiperparámetros:**
- Optimizador: Adam (lr=0.001)
- Función de pérdida: CrossEntropyLoss
- Batch size: 128
- Épocas: 15 (early stopping en época 8-10)
- Total de parámetros: ~412,734

---

## 📊 Datasets Utilizados

### 1. Dataset Custom (Manuscrito Propio)
- **Fuente**: Caracteres manuscritos de compañeros de clase
- **Clases**: 37 (números + mayúsculas, algunas con minúsculas mezcladas)
- **Total**: ~2,417 imágenes
- **División**: 80% entrenamiento, 20% validación
- **Nota**: En Windows, mayúsculas y minúsculas se fusionaron en mismas carpetas

### 2. EMNIST (Extended MNIST)
- **Fuente**: Dataset público de caracteres manuscritos
- **Clases**: 62 (0-9, A-Z, a-z)
- **Total**: ~697,932 entrenamiento, 116,323 test
- **Uso**: Base principal del modelo, combinado con custom (peso 2x)

### Dataset Combinado
- **Total train**: ~701,798 imágenes
- **Total test**: ~117,291 imágenes
- **Proporción**: EMNIST + Custom×2 (para dar más peso a escritura propia)

### Preprocesamiento
- Conversión a escala de grises
- Redimensionamiento a 28×28 píxeles
- Binarización (Otsu)
- Normalización ToTensor() [0, 1]
- **Sin Normalize() adicional** (causaba degradación de accuracy)

---

## 📈 Resultados

### Precisión Alcanzada
- **Caracteres individuales**: 70-80% accuracy
- **Palabras simples**: Funcional con ~70% letras correctas
- **Confusiones comunes**: l↔1, i↔1, O↔0, a↔o (inherentes a manuscritos)

### Evolución del Entrenamiento

| Época | Train Loss | Val Loss | Observación |
|-------|-----------|----------|-------------|
| 1 | 0.75 | 0.44 | Aprendizaje inicial |
| 2 | 0.39 | 0.37 | Mejora rápida |
| 8 | 0.30 | 0.35 | **Mejor modelo** |
| 15 | 0.25 | 0.41 | Overfitting detectado |

**Conclusión:** Early stopping en época 8-10 es óptimo (val_loss mínimo: 0.35)

### Mejoras Implementadas

✅ **Early Stopping**
- Implementado con patience=5
- Guarda mejor modelo automáticamente
- Ahorra ~30 minutos de entrenamiento

✅ **Data Augmentation**
- Rotación: ±10°
- Escalado: 0.9x - 1.1x
- Traslación: ±10%
- Perspectiva: 20% distorsión (50% prob)
- Brillo/contraste: ±20%/±15%

✅ **Segmentación Mejorada**
- Proyección vertical (reemplaza contornos)
- Filtrado de trazos horizontales (líneas de 't')
- División automática de letras fusionadas
- Recorte vertical para eliminar espacios

✅ **Post-procesamiento Ñ/ñ**
- Diccionario de palabras comunes: España, mañana, niño, año, señor
- Corrección basada en patrones (Ñ→R, ñ→r)

---

## 🚀 Uso del Sistema

### Instalación

```bash
# 1. Clonar repositorio
git clone https://github.com/natiicruuz/-25-26-IYA051---Inteligencia-Artificial.git

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Predicción de Caracteres

```bash
python inference/predict_char.py \
    --image test_images/char.png \
    --model models/weights/final_custom_emnist_augmented_*.pth
```

### Predicción de Palabras

```bash
# Básico
python inference/predict_word.py \
    --image test_images/word.png \
    --model models/weights/final_custom_emnist_augmented_*.pth

# Con debugging (muestra segmentación)
python inference/predict_word.py \
    --image test_images/word.png \
    --model modelo.pth \
    --debug

# Ajustar sensibilidad (menor = más sensible)
python inference/predict_word.py \
    --image test_images/word.png \
    --model modelo.pth \
    --threshold 0.05
```

### Entrenamiento Desde Cero

```bash
# Con custom + EMNIST + early stopping + augmentation
python scripts/train_custom_emnist.py \
    --epochs 15 \
    --patience 5 \
    --custom ./data/normalized
```

---

## 📁 Estructura del Proyecto

```
-25-26-IYA051---Inteligencia-Artificial/
├── models/                 # Arquitectura del modelo
│   ├── config.py           # Configuración (62/64 clases, mappings)
│   ├── cnn.py              # Red CNN con 3 capas conv + 3 FC
│   └── ocr_model.py        # Wrapper para inferencia
│
├── datasets/               # Carga de datos
│   ├── normalized_dataset.py  # Dataset custom normalizado
│   ├── combined_dataset.py    # Custom + EMNIST
│   └── augmented_dataset.py   # Wrapper para augmentation
│
├── training/                # Scripts de entrenamiento
│   └── train_custom_emnist.py  # Entrenamiento combinado (62 clases)
│
├── inference/              # Scripts de predicción
│   ├── predict_char.py     # Predicción de caracteres
│   └── predict_word.py     # Predicción de palabras
│
├── segmentation/           # Segmentación
│   ├── word_segmentation.py    # Detección de palabras
│   └── letter_segmentation.py  # Proyección vertical
│
├── utils/                  # Utilidades
│   ├── image_utils.py      # Resize con padding, conversiones
│   └── viz.py              # Visualización de resultados
│
├── data/                   # Datos
│   ├── raw/                # Dataset original (mayúsculas, minúsculas, números)
│   ├── normalized/         # Dataset normalizado (28x28, binarizado)
│   └── emnist/             # EMNIST descargado automáticamente
│
├── models/weights/         # Modelos entrenados
│   └── final_custom_emnist_augmented_*.pth
│
├── test_images/            # Imágenes de prueba
├── checkpoints/            # Checkpoints durante entrenamiento
└── requirements.txt        # Dependencias
```

---

## 🔧 Tecnologías Utilizadas

### Bibliotecas Principales
- **PyTorch 2.6.0**: Framework de deep learning
- **OpenCV 4.8.0**: Procesamiento de imágenes (binarización, morfología)
- **NumPy 2.1.0**: Operaciones numéricas
- **Matplotlib 3.7.0**: Visualización
- **Pillow 10.0.0**: Manejo de imágenes (PIL)
- **SciPy 1.13.0**: Filtros gaussianos (normalización)
- **tqdm 4.65.0**: Barras de progreso

### ⚠️ Restricciones Cumplidas
- ❌ **NO se usó**: Tesseract, EasyOCR, Google Vision API, Keras OCR, o similares
- ✅ **Solo se usó**: PyTorch (framework general), OpenCV (procesamiento básico), NumPy
- ✅ **Implementación propia**: Segmentación, arquitectura CNN, entrenamiento, inferencia

---

## 🐛 Problemas Conocidos y Soluciones

### 1. Confusión l ↔ 1, i ↔ 1
**Causa:** Ambos son líneas verticales casi idénticas en manuscrito  
**Solución actual:** Post-procesamiento con contexto (diccionario)  
**Mejora futura:** LSTM/Transformer para contexto a nivel de palabra

### 2. Ñ/ñ detectada como R/r
**Causa:** Dataset custom insuficiente (~38 imágenes de Ñ/ñ vs 11,257 de otras)  
**Solución actual:** Diccionario post-procesamiento ('Espana' → 'España')  
**Mejora futura:** Aumentar Ñ/ñ a 10,000+ imágenes sintéticas

### 3. Segmentación falla con letras muy juntas
**Causa:** Proyección vertical no encuentra valles entre letras fusionadas  
**Solución actual:** División automática si aspect_ratio > 2.5  
**Mejora futura:** Segmentación basada en aprendizaje profundo

### 4. Windows fusiona mayúsculas/minúsculas
**Causa:** Sistema de archivos case-insensitive (A/ = a/)  
**Solución actual:** Dataset normalizado sin sufijos (37 clases mezcladas)  
**Mejora futura:** Usar sufijos (_upper, _lower) para separar

---

## 📊 Decisiones de Diseño

### ¿Por qué 62 clases y no 64?

**Decisión:** Entrenar con EMNIST (62 clases: 0-9, A-Z, a-z) + custom (37 clases)

**Razones:**
1. **Volumen de datos**: EMNIST aporta 697,932 imágenes vs 2,417 custom
2. **Desbalance masivo**: Ñ/ñ tienen ~38 imágenes cada una vs 11,257 promedio en otras
3. **Tiempo limitado**: Re-entrenar con 64 clases requiere 3-4 horas adicionales
4. **Solución pragmática**: Post-procesamiento para Ñ/ñ es suficiente para el examen

**Alternativa (no implementada):** Aumentar Ñ/ñ a 10,000 imágenes sintéticas + re-entrenar

### ¿Por qué proyección vertical y no contornos?

**Decisión:** Segmentación por proyección vertical con detección de valles

**Razones:**
1. **Robustez**: Funciona con letras muy juntas o ligeramente solapadas
2. **Simplicidad**: Algoritmo determinista sin hiperparámetros complejos
3. **Rendimiento**: Detecta correctamente 7/7 letras en pruebas (vs 4-6 con contornos)

**Desventajas:** Falla con escritura cursiva muy conectada

### ¿Por qué early stopping en época 8-10?

**Decisión:** Implementar early stopping con patience=5

**Razones:**
1. **Evidencia empírica**: Val_loss mínimo en época 8 (0.35)
2. **Overfitting**: Después de época 10, val_loss sube mientras train_loss baja
3. **Eficiencia**: Ahorra ~30 minutos de entrenamiento sin pérdida de accuracy

**Resultado:** Modelo óptimo guardado automáticamente

### ¿Por qué data augmentation?

**Decisión:** Aplicar rotación, escalado, traslación, perspectiva, brillo/contraste

**Razones:**
1. **Dataset pequeño**: Custom tiene solo 2,417 imágenes
2. **Variabilidad**: Simula diferentes estilos de escritura y condiciones de captura
3. **Generalización**: Reduce overfitting al exponer el modelo a más variaciones

**Resultado:** Mejora de accuracy de ~65% a ~70%

---

## 📝 Limitaciones del Trabajo

### 1. Limitaciones Técnicas

#### Dataset Insuficiente
- **Custom**: Solo 2,417 imágenes (objetivo: 20,000+)
- **Ñ/ñ**: Solo ~38 imágenes cada una (objetivo: 10,000+)
- **Impacto**: Limita capacidad de generalización, especialmente en caracteres poco frecuentes

#### Precisión Limitada
- **Actual**: 70-80% en caracteres individuales
- **Objetivo profesional**: >95%
- **Causas**: Dataset pequeño, arquitectura básica, sin post-procesamiento avanzado

#### Overfitting Moderado
- **Evidencia**: Val_loss sube después de época 8 mientras train_loss baja
- **Causa**: Modelo memoriza training set en lugar de generalizar
- **Solución parcial**: Early stopping, dropout 0.3-0.5, data augmentation

### 2. Limitaciones de Funcionalidad

#### Sin Reconocimiento de Contexto
- **Actual**: Predicción letra por letra sin contexto
- **Problema**: 'l' vs '1', 'i' vs '1' indistinguibles sin contexto
- **Ejemplo**: "hola" puede predecirse como "ho1a"

#### Sin Corrección Ortográfica Avanzada
- **Actual**: Diccionario simple de 10-15 palabras comunes
- **Problema**: No corrige palabras fuera del diccionario
- **Ejemplo**: "natalia" predicho como "natal1a" no se corrige

#### Segmentación Falla con Cursiva
- **Actual**: Proyección vertical asume separación entre letras
- **Problema**: Cursiva conecta letras, no hay valles
- **Resultado**: Fusiona múltiples letras en un segmento

### 3. Limitaciones de Arquitectura

#### Modelo Básico (CNN 3 capas)
- **Actual**: 412,734 parámetros, 3 capas convolucionales
- **Mejor práctica**: ResNet, DenseNet, Vision Transformer (millones de parámetros)
- **Impacto**: Menor capacidad de aprender patrones complejos

#### Sin Memoria de Secuencia
- **Actual**: Cada letra se predice independientemente
- **Mejor práctica**: LSTM/Transformer que recuerdan contexto
- **Ejemplo**: LSTM sabe que después de 'q' suele venir 'u'

#### Sin Attention Mechanism
- **Actual**: Red trata todas las regiones de la imagen por igual
- **Mejor práctica**: Attention se enfoca en partes relevantes
- **Impacto**: Menos robusto a ruido y variaciones

### 4. Limitaciones de Entorno

#### Solo Windows
- **Dataset normalizado**: Mezcladas mayúsculas/minúsculas por filesystem case-insensitive
- **Solución**: Requiere re-normalizar con sufijos para separar correctamente

#### Sin GPU dedicada
- **Entrenamiento**: 2-3 horas en CPU (vs 20-30 min en GPU)
- **Limitación**: Impide experimentar rápidamente con arquitecturas más grandes

#### Recursos computacionales limitados
- **RAM**: 8-16GB limita batch size y tamaño de modelo
- **Almacenamiento**: Dataset grande (EMNIST) requiere 1-2GB

---

## 🎯 Mejoras Futuras

1. **Learning rate scheduler**: Pendiente (ReduceLROnPlateau)
2. **Diccionario español completo**: Pendiente (500+ palabras)
3. **Arquitecturas avanzadas**: Probar ResNet18, MobileNetV2
4. **LSTM para contexto**: Añadir capa recurrente después de CNN
5. **Aumentar Ñ/ñ**: Generar 10,000 variaciones sintéticas
6. **Post-procesamiento**: Beam search + modelo de lenguaje (n-gramas)
7. **Transformer end-to-end**: TrOCR o similar
8. **Detección de layout**: Reconocer párrafos, columnas, títulos
9. **Reconocimiento de fórmulas matemáticas**: Ampliar a LaTeX
10. **API REST**: Servicio web para reconocimiento en tiempo real

---

## ✅ Conclusiones

### Logros Principales
1. ✅ **Sistema OCR funcional** sin bibliotecas especializadas
2. ✅ **Arquitectura CNN** diseñada, implementada y entrenada desde cero
3. ✅ **Segmentación automática** de palabras y letras sin librerías
4. ✅ **Precisión 70-80%** en caracteres individuales (suficiente para contexto académico)
5. ✅ **Soporte parcial Ñ/ñ** mediante post-procesamiento

### Lecciones Aprendidas
1. **Debugging sistemático**: Bugs de dataset, normalización, mapeo requirieron análisis cuidadoso
2. **Trade-offs**: Elegir entre precisión (64 clases con Ñ/ñ) vs eficiencia (62 clases + post-proc)
3. **Importancia de datos**: Más datos > mejor arquitectura (EMNIST 697k vs custom 2k)
4. **Regularización esencial**: Dropout, BatchNorm, early stopping previenen overfitting

### Reflexión Final

Este proyecto demuestra que es posible construir un sistema OCR funcional desde cero, pero también evidencia la complejidad real del problema. La precisión de 70-80% refleja los desafíos inherentes del reconocimiento manuscrito: alta variabilidad en escritura humana, necesidad de grandes volúmenes de datos, y arquitecturas sofisticadas.

El proceso de desarrollo, debugging y optimización fue tan valioso como la implementación inicial, proporcionando una comprensión profunda del pipeline completo de OCR y las decisiones de diseño involucradas en sistemas de ML reales.

**El sistema cumple los objetivos académicos y proporciona una base sólida para futuras mejoras.**

---

## 📚 Referencias

1. **EMNIST Dataset**: Cohen, G., et al. (2017). "EMNIST: Extending MNIST to handwritten letters"
2. **PyTorch**: Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library"
3. **Dropout**: Srivastava, N., et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
4. **Batch Normalization**: Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training"
5. **Adam Optimizer**: Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization"
---

<div align="center">
  <p>
    <strong>Sistema OCR - Reconocimiento de Caracteres Manuscritos</strong><br>
    Trabajo Final de IA 2024-2025
  </p>
  <p>
     <strong>Natalia Cruz Babbar</strong><br>
     Ingeniería Informática<br>
     Universidad Europea del Atlántico
     </p>
</div>