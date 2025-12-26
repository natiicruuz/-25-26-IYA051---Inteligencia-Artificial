# 📝 Sistema OCR para Caracteres Manuscritos

Sistema de Reconocimiento Óptico de Caracteres (OCR) implementado desde cero usando PyTorch para detectar y transcribir texto manuscrito en español (incluyendo Ñ/ñ).

## 🎯 Características

### Funcionalidades Principales
- ✅ **Reconocimiento de texto manuscrito** (letras mayúsculas, minúsculas y números)
- ✅ **Soporte para español** (incluye Ñ y ñ)
- ✅ **Segmentación automática** de palabras y letras
- ✅ **Pipeline completo** de entrenamiento e inferencia

### Funcionalidades Extras
- 📊 **Detección de tablas** con exportación a Markdown
- 🖼️ **Extracción de figuras/imágenes** del documento
- 📈 **Visualización** de métricas de entrenamiento
- 💾 **Guardado de predicciones** en formato texto

## 📁 Estructura del Proyecto

```
ocr_proyecto/
├── data/                          # Datos del proyecto
│   ├── raw/                       # Datos sin procesar
│   ├── processed/                 # Datos procesados
│   └── labels/                    # Etiquetas
│
├── models/                        # Arquitectura del modelo
│   ├── config.py                  # Configuración y mapeos
│   ├── cnn.py                     # Red neuronal convolucional
│   ├── ocr_model.py              # Wrapper del modelo
│   └── weights/                   # Modelos entrenados
│
├── datasets/                      # Carga de datasets
│   ├── handwritten_dataset.py    # Dataset manuscrito custom
│   └── combined_dataset.py       # Combinación custom + EMNIST
│
├── segmentation/                  # Segmentación de imágenes
│   ├── word_segmentation.py      # Detección de palabras
│   └── letter_segmentation.py    # Extracción de letras
│
├── training/                      # Scripts de entrenamiento
│   ├── train_handwritten.py      # Entrenar con dataset custom
│   └── train_combined.py         # Entrenar con custom + EMNIST
│
├── inference/                     # Scripts de predicción
│   ├── predict_char.py           # Predecir un carácter
│   ├── predict_word.py           # Predecir una palabra
│   └── predict_document.py       # Predecir documento completo
│
├── extras/                        # Funcionalidades adicionales
│   ├── detect_tables.py          # Detección de tablas
│   └── detect_figures.py         # Detección de figuras
│
├── utils/                         # Utilidades
│   ├── image_utils.py            # Procesamiento de imágenes
│   └── viz.py                    # Visualización
│
├── main.py                        # Script principal
└── requirements.txt               # Dependencias
```

## 🚀 Instalación

### 1. Clonar/descargar el proyecto

```bash
cd ocr_proyecto
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Activar en Windows
venv\Scripts\activate

# Activar en Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 📚 Uso

### Entrenamiento

#### Opción 1: Solo dataset manuscrito custom (90 épocas)

```bash
python training/train_handwritten.py \
    --dataset /ruta/al/dataset \
    --epochs 90 \
    --batch-size 128 \
    --save-dir ./models/weights
```

**Estructura esperada del dataset:**
```
dataset/
├── mayúsculas/
│   ├── A/
│   │   ├── A_Nombre_Apellido.png
│   └── ...
├── minúsculas/
│   ├── a/
│   │   ├── a_Nombre_Apellido.png
│   └── ...
└── números/
    ├── 0/
    │   ├── 0_Nombre_Apellido.png
    └── ...
```

#### Opción 2: Dataset combinado (custom + EMNIST, 8 épocas)

```bash
python training/train_combined.py \
    --custom-dataset /ruta/al/dataset \
    --emnist-root ./data \
    --epochs 8 \
    --batch-size 128
```

### Inferencia

#### 1. Predecir un carácter individual

```bash
python inference/predict_char.py \
    --image imagen_caracter.png \
    --model models/weights/modelo.pth
```

#### 2. Predecir una palabra

```bash
python inference/predict_word.py \
    --image imagen_palabra.png \
    --model models/weights/modelo.pth
```

#### 3. Predecir documento completo

```bash
python inference/predict_document.py \
    --image documento.png \
    --model models/weights/modelo.pth \
    --output documento_texto.txt
```

### Pipeline Completo (main.py)

#### OCR básico (solo texto)

```bash
python main.py \
    --image documento.png \
    --model models/weights/modelo.pth
```

#### OCR completo (texto + tablas + figuras)

```bash
python main.py \
    --image documento.png \
    --model models/weights/modelo.pth \
    --extract-tables \
    --extract-figures \
    --output ./resultados
```

### Funcionalidades Extras

#### Detectar tablas

```bash
python extras/detect_tables.py \
    --image documento.png \
    --output-dir ./tablas_extraidas \
    --visualize
```

#### Detectar figuras

```bash
python extras/detect_figures.py \
    --image documento.png \
    --output-dir ./figuras_extraidas \
    --visualize
```

## 🏗️ Arquitectura del Modelo

### Red Neuronal Convolucional (OCRCNN)

```
Input: (1, 28, 28)
    ↓
[Bloque Conv 1] → 32 filtros → ReLU → BatchNorm → MaxPool → Dropout(0.3)
    ↓
[Bloque Conv 2] → 64 filtros → ReLU → BatchNorm → MaxPool → Dropout(0.3)
    ↓
[Bloque Conv 3] → 96 filtros → ReLU → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Flatten → (864,)
    ↓
[FC 1] → 320 → ReLU → Dropout(0.5)
    ↓
[FC 2] → 160 → ReLU → Dropout(0.5)
    ↓
[Output] → num_classes (62 o 64)
```

**Parámetros:**
- Total de parámetros: ~350,000
- Optimizador: Adam (lr=0.001)
- Función de pérdida: CrossEntropyLoss

## 📊 Datasets

### Dataset Custom
- **Clases**: 64 (0-9, A-Z, Ñ, a-z, ñ)
- **Formato**: Imágenes PNG organizadas por carpetas
- **Preprocesamiento**: Grayscale, Resize(28x28), Normalización

### EMNIST (Extended MNIST)
- **Clases**: 62 (0-9, A-Z, a-z, sin Ñ/ñ)
- **Fuente**: Torchvision datasets
- **Preprocesamiento**: Rotación, flip, inversión de colores

## 🎯 Resultados Esperados

Con un entrenamiento adecuado, el modelo debería alcanzar:
- **Precisión en custom dataset**: ~85-95%
- **Precisión en EMNIST**: ~90-95%
- **Precisión en dataset combinado**: ~88-93%

## 🔧 Personalización

### Modificar arquitectura del modelo

Edita `models/cnn.py`:

```python
# Ejemplo: Añadir más filtros
nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
```

### Cambiar transformaciones

Edita las transformaciones en los scripts de entrenamiento o en `utils/image_utils.py`.

### Añadir nuevas clases

Modifica los mapeos en `models/config.py`:

```python
self._custom_mapping = [
    '0', '1', ..., 'z', '!', '?'  # Añadir símbolos
]
```

## 📝 Notas Importantes

1. **Restricción del proyecto**: No se usan librerías de OCR de alto nivel (Tesseract, EasyOCR, etc.)
2. **Solo manuscrito**: El proyecto se centra en texto manuscrito (no texto impreso)
3. **Español**: Soporte completo para caracteres españoles (Ñ, ñ)
4. **GPU**: El entrenamiento es más rápido con GPU (CUDA)

## 🐛 Solución de Problemas

### Error: "No module named 'torch'"
```bash
pip install torch torchvision
```

### Error: "CUDA out of memory"
Reduce el `batch_size`:
```bash
python training/train_handwritten.py --batch-size 64
```

### Baja precisión en predicciones
- Verifica que las imágenes tengan buena calidad
- Asegúrate de que el modelo esté bien entrenado (>80 épocas)
- Prueba con el dataset combinado (custom + EMNIST)

## 📄 Licencia

Este proyecto es un trabajo académico para la asignatura de Inteligencia Artificial.

## 👥 Autor

Desarrollado como proyecto final de IA - 2025

---

## 🎓 Trabajo Académico

**Asignatura**: Inteligencia Artificial  
**Año**: 2025  
**Requisitos cumplidos**:
- ✅ Reconocimiento de texto manuscrito
- ✅ Segmentación de palabras y letras
- ✅ Detección de tablas (extra)
- ✅ Detección de figuras (extra)
- ✅ Sin uso de librerías OCR de alto nivel