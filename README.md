# Monitor de Cultivo Inteligente: Detección de Brócoli mediante IA

## Descripción del Proyecto
Este sistema es un prototipo de visión artificial diseñado para integrarse en tractores inteligentes o sistemas de monitoreo en invernaderos. Su objetivo principal es analizar el cultivo en tiempo real y clasificar lo que capta la cámara en tres categorías:

* **Brócoli Sano**
* **Brócoli Enfermo**
* **Maleza**

El núcleo del sistema está impulsado por un modelo de aprendizaje profundo (**YOLOv5**) entrenado específicamente con un conjunto de datos personalizado, y una interfaz gráfica amigable desarrollada con **OpenCV** y **Tkinter**.

---

## Características Principales

- **Análisis en Tiempo Real:** Procesa el video de la cámara web para detectar y enmarcar las plantas al instante.
- **Métricas en Pantalla:** Muestra el conteo exacto de las plantas detectadas por categoría y el tiempo de procesamiento (velocidad de inferencia) en milisegundos.
- **Interfaz Amigable:** Cuenta con un panel de control intuitivo que permite a cualquier usuario operar el sistema sin necesidad de interactuar con el código fuente.
- **Pipeline de Entrenamiento Incluido:** Se adjunta el cuaderno de Jupyter utilizado para descargar, procesar y entrenar el modelo original.

---

# Guía de Inicio Rápido (Para usuarios sin experiencia)

## 1. Requisitos Previos

Para que este programa funcione en tu computadora, necesitas tener instalado lo siguiente:

* **Python 3.8 o superior:** Puedes descargarlo de forma gratuita desde [python.org](https://www.python.org/downloads/).

> **Importante para Windows:** Durante la instalación de Python, asegúrate de marcar la casilla que dice *"Add Python to PATH"* (Agregar Python al PATH) antes de presionar **Install Now**.

* **Una cámara web** funcional y conectada a tu equipo.

---

## 2. Instalación de Dependencias

El sistema utiliza algunas librerías externas de Python para funcionar.

Abre tu terminal (Símbolo del sistema en Windows o Terminal en Mac/Linux) y ejecuta el siguiente comando:

```bash
pip install ultralytics opencv-python pillow
```

---

## 3. Ejecución del Programa

Asegúrate de haber descargado todos los archivos de este proyecto y guardarlos juntos en una sola carpeta.

Tienes dos opciones para usar el sistema:

### Opción A: Interfaz Gráfica Completa (Recomendado)

Abre tu terminal, navega hasta la carpeta donde guardaste los archivos y ejecuta:

```bash
python interfaz_brocoli.py
```

Esto abrirá el panel de control principal con visualización de cámara en vivo y métricas.

---

### Opción B: Ventana de Prueba Rápida (Modo Desarrollador)

Si solo deseas probar el reconocimiento sin la interfaz visual:

```bash
python detectar_lap.py
```

Para salir de este modo:

1. Haz clic sobre la ventana de video.
2. Presiona la tecla `q` en tu teclado.

---

# Estructura del Proyecto

Si deseas explorar el código, entender cómo funciona o reentrenar el modelo, esta es la función de cada archivo:

## `interfaz_brocoli.py`

Script principal.

Contiene el código de la Interfaz Gráfica de Usuario (GUI) que conecta el modelo de IA con una ventana amigable usando **Tkinter**.

---

## `detectar_lap.py`

Script ligero de inferencia.

Ideal para comprobar rápidamente que la cámara y el modelo se comunican correctamente.

---

## `best.pt`

Archivo crítico que contiene los pesos entrenados del modelo YOLOv5.

Debe permanecer siempre en la misma carpeta que los scripts de Python.

---

## `Deteccion_V1.ipynb`

Cuaderno de Jupyter optimizado para Google Colab que documenta todo el proceso científico:

* Conexión a la API de Roboflow para descargar el dataset.
* División automática de imágenes:
  * 70% entrenamiento
  * 20% validación
  * 10% pruebas
* Entrenamiento del modelo base durante 100 épocas.
* Evaluación de métricas y validación.

---

## `dataset/`

Directorio sugerido donde se almacenan las imágenes de entrenamiento y validación separadas por clases:

* `brocoli_sano`
* `brocoli_enfermo`
* `maleza`

---

# Tecnologías Utilizadas

| Categoría | Tecnología |
|---|---|
| Lenguaje | Python |
| Visión Computacional | OpenCV, PIL (Pillow) |
| Inteligencia Artificial | Ultralytics YOLOv5 |
| Interfaz Gráfica | Tkinter |
| Gestión de Datos | Pandas, Matplotlib, split-folders, Roboflow API |

---

# Objetivo del Proyecto

Este proyecto busca demostrar cómo la inteligencia artificial y la visión computacional pueden aplicarse en la agricultura inteligente para:

* Automatizar el monitoreo de cultivos.
* Detectar enfermedades tempranamente.
* Reducir el trabajo manual.
* Optimizar recursos agrícolas.
* Facilitar la toma de decisiones en tiempo real.

---