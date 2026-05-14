# PROYECTO: DETECCIÓN DE BRÓCOLI MEDIANTE VISIÓN ARTIFICIAL

## 📖 DESCRIPCIÓN DEL PROYECTO
Sistema de visión artificial para tractores inteligentes que clasifica brócoli (sano/enfermo) y maleza utilizando YOLOv5 y OpenCV sobre hardware embebido.

## 🏗️ ARQUITECTURA DEL SISTEMA

### FASE 1: ENTRENAMIENTO DEL MODELO (DETECCION_V1.IPYNB)
* **Dataset:** Descarga desde Roboflow (API).
* **Segmentación:** División 70/20/10 mediante `split-folders`.
* **Entrenamiento:** 100 épocas con `yolov5n.pt` y resolución de imganenes del dataset a 640px.

### FASE 2: DESPLIEGUE E INFERENCIA LOCAL (DETECTAR_LAP.PY)
* **Inferencia:** Carga de `best.pt` para predicción en tiempo real.
* **Captura:** OpenCV para video en vivo con umbral de confianza de 0.5.

## ⚙️ REQUISITOS Y DEPENDENCIAS
* Python 3.x
* Librerías: `ultralytics`, `opencv-python`, `split-folders`, `roboflow`.

## 🚀 GUÍA DE USO
1. Colocar `best.pt` en la raíz.
2. Ejecutar `python detectar_lap.py`.
3. Presionar 'q' para salir.

## 📂 ESTRUCTURA DE ARCHIVOS PRINCIPAL
* `detectar_lap.py`: Script de inferencia local.
* `Deteccion_V1.ipynb`: Pipeline de entrenamiento.
* `best.pt`: Pesos del modelo entrenado.
