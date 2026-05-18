"""
Script básico para la detección de enfermedades en brócoli y maleza.
Utiliza la cámara web del equipo para analizar el video en tiempo real
usando un modelo de Inteligencia Artificial (YOLO).
"""

import cv2
from ultralytics import YOLO

# 1. Cargar el modelo de IA entrenado. 
# El archivo "best.pt" contiene el "cerebro" de la IA que ya aprendió a reconocer el brócoli.
model = YOLO("best.pt")

# 2. Configurar la cámara web. 
# El '0' indica que usaremos la cámara principal de la laptop.
cap = cv2.VideoCapture(0)

print("Iniciando cámara... Presiona la tecla 'q' en tu teclado para salir del prototipo.")

# 3. Bucle principal de video. 
# Un video es solo una secuencia de imágenes (frames). Este ciclo analiza una imagen tras otra.
while cap.isOpened():
    # Leer el frame (imagen actual) de la cámara
    exito, frame = cap.read()
    
    # Si no se pudo leer la imagen (ej. la cámara se desconectó), salimos del ciclo
    if not exito:
        print("Error: No se pudo acceder a la cámara.")
        break

    # 4. Realizar la detección (Inferencia)
    # Le pasamos la imagen al modelo. 
    # conf=0.5 significa que solo mostrará detecciones de las que esté 50% o más seguro.
    # stream=True optimiza el uso de memoria para video en vivo.
    resultados = model(frame, conf=0.5, stream=True)

    # 5. Dibujar los resultados en la imagen
    for resultado in resultados:
        # La función plot() dibuja automáticamente los cuadros y nombres (brócoli sano, enfermo, maleza)
        imagen_anotada = resultado.plot()  

    # 6. Mostrar la ventana con el video en vivo y las detecciones
    cv2.imshow("Prototipo - Detección de Plantas", imagen_anotada)

    # 7. Condición de salida
    # Espera 1 milisegundo a ver si el usuario presiona una tecla. Si es la 'q', se rompe el ciclo.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Limpieza: Apagar la cámara y cerrar las ventanas de video
cap.release()
cv2.destroyAllWindows()