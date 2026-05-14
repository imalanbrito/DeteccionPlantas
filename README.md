
🌱 Detección de Brócoli mediante Visión Artificial
📖 Descripción del Proyecto
Este proyecto implementa un sistema de visión artificial diseñado para integrarse en un tractor inteligente. Su objetivo principal es identificar y clasificar plantas en campos de cultivo en tiempo real, diferenciando entre:

Brócoli sano

Brócoli enfermo

Maleza (cualquier otro elemento detectado)

El sistema está dividido en dos fases principales: el preprocesamiento y entrenamiento del modelo en un entorno de cuaderno (Jupyter/Colab) y el despliegue del motor de inferencia local para el prototipo.

🏗️ Arquitectura del Sistema
Fase 1: Entrenamiento del Modelo (Deteccion_V1.ipynb)
Este módulo se encarga de la ingesta de datos, preparación y el entrenamiento de la red neuronal convolucional.

Ingesta de Datos: Conexión mediante la API de Roboflow para descargar el dataset etiquetado de cultivos (deteccion-brocoli-yolo).

Segmentación: Uso de split-folders para dividir las imágenes de manera representativa:

70% Entrenamiento

20% Validación

10% Prueba

Entrenamiento: Se aplica transferencia de aprendizaje utilizando un modelo preentrenado de YOLOv5 (yolov5n.pt). El modelo se entrena durante 100 épocas con un tamaño de lote (batch) de 16 y una resolución de 640 píxeles.

Validación: El script incluye herramientas basadas en pandas y matplotlib para graficar la pérdida (loss) y la precisión (mAP), asegurando la calidad del modelo final (best.pt).

Fase 2: Despliegue e Inferencia Local (detectar_lap.py)
Este script es el motor de ejecución en tiempo real, diseñado para correr en el dispositivo Edge (laptop o Raspberry Pi) conectado a las cámaras del tractor.

Captura de Video: Utiliza OpenCV (cv2) para acceder a la cámara web (VideoCapture(0)).

Inferencia Continua: El modelo evalúa cada fotograma de manera asíncrona mediante el modo stream de Ultralytics, aplicando un umbral de confianza del 50% (conf=0.5).

Renderizado: Las detecciones (cajas delimitadoras y etiquetas) se superponen automáticamente en el video en vivo.

⚙️ Requisitos y Dependencias
Para ejecutar este proyecto, asegúrate de tener instalado Python y las siguientes librerías:

Bash
pip install ultralytics opencv-python split-folders roboflow pandas matplotlib pyyaml pillow
🚀 Guía de Uso
1. Reentrenar el Modelo (Opcional)
Si deseas agregar más imágenes al dataset o modificar parámetros, ejecuta el cuaderno en Google Colab o Jupyter:

Configura tu API Key de Roboflow en la celda de descarga.

Ejecuta las celdas secuencialmente para descargar, dividir y entrenar.

Al finalizar, descarga el archivo de pesos generados: runs/detect/proyecto_brocoli/entrenamiento_brocoli6/weights/best.pt.

2. Ejecutar el Prototipo en Tiempo Real
Asegúrate de que tu cámara esté conectada y que el archivo best.pt se encuentre en el mismo directorio que el script principal.

Ejecuta el siguiente comando en tu terminal:

Bash
python detectar_lap.py
Controles: Presiona la tecla q en tu teclado para detener la cámara y cerrar el programa de forma segura.

📂 Estructura de Archivos Principal
detectar_lap.py: Script principal para la detección en tiempo real con la cámara.

Deteccion_V1.ipynb: Cuaderno de experimentación, métricas y entrenamiento del modelo.

best.pt: Archivo PyTorch con los pesos finales del modelo entrenado.

data.yaml: Archivo de configuración generado durante la descarga del dataset para guiar el entrenamiento.
