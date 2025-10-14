Reto del examen Examen Parcial


Reto de Visión Artificial: Reconocimiento de Cartas mediante Procesamiento Clásico de Imágenes


# Objetivo General
Desarrollar un sistema de reconocimiento automático de cartas de juego utilizando técnicas clásicas de visión artificial, sin emplear redes neuronales ni aprendizaje profundo. 

El objetivo es identificar una o varias cartas presentes en una escena controlada.

# Requisitos y Condiciones
1. Captura de Imagen
Libertad total de hardware: el participante puede elegir libremente la cámara o sensor de captura.
Libertad total de formatos de captura: el participante puede elegir libremente formatos de imgaen o video.
Control de iluminación opcional: se permite emplear fuentes de luz controladas o naturales, siempre que se documente el tipo de iluminación usada.
Se pueden utilizar elementos auxiliares para aislar la imagen de interferencias o ruido del exterior.

3. Escenario
Se deberá utilizar un TAPETE VERDE o superficie uniforme que cubra completamente el fondo del plano de la imagen.

4. Cartas
El sistema debe reconocer cartas de una baraja estándar de póker (baraja francesa o naipes).

Requisito: Número de cartas en la escena:
Mínimo: 1 carta visible completamente.
Opcional (+puntos): varias cartas en la misma imagen sin oclusiones.
Opcional (+puntos): varias cartas en escena con oclusiones parciales.

4. Restricciones técnicas
No se permite el uso de redes neuronales, modelos entrenados, clasificadores SVM entrenados, ni librerías de visión con funciones de reconocimiento basado en aprendizaje.
Solo se pueden usar operaciones clásicas de procesamiento de imagen, tales como:
- Transformaciones de color (RGB, HSV, LAB, etc.)
- Umbralización y segmentación.
- Operaciones morfológicas.
- Detección de bordes, contornos y esquinas.
- Filtros espaciales y convoluciones.
- Transformaciones geométricas.
- Coincidencia mediante correlación o plantillas fijas.
- Operaciones matriciales directas.
- Otras [preguntar al profesor]

Entregables
- Breve memoria técnica en PDF describiendo:
    - Hardware: características técnicas y requisitos. Justificación de uso.
    - Software: características técnicas y requisitos. Justificación de uso.
    - Hoja de ruta del desarrollo: Descripción del proceso de desarrollo.
    - Solución:
      - Diagrama de decisión (o equivalente) para la clasificación de las cartas
      - Secuencialización de las operaciones realizadas sobre las imágenes con explicación de las configuraciones de cada una de las funciones utilizadas y el porqué de cada parámetro.
    - Otras tareas realizadas
- Código fuente documentado.
- Incluir si se utiliza: Link de Base de datos para pruebas.


# Sistema de Puntuación (Total: 10 + 2 ptos)
* Reconocimiento básico (obligatorio): Identificación correcta de una carta en la escena (número y palo).	Se presentan 10 cartas a reconocer, cada una de ellas en distintas posiciones y orientaciones. 
* Múltiples cartas	Identificación simultánea de varias cartas en la misma imagen.	+1 pts. Hasta 3 intentos. Puntúa el mejor.
* Oclusiones parciales	Capacidad para reconocer cartas parcialmente tapadas.	+1 pts. Hasta 3 intentos. Puntúa el mejor.
* Diseño y documentación	Claridad en el código, explicación técnica, justificación de decisiones. hasta -5 puntos de no presentarse en tiempo y formato adecuados.
