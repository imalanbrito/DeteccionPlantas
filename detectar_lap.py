import cv2
from ultralytics import YOLO

# 1. Cargar el modelo entrenado (.pt)
# Asegúrate de que el archivo best.pt esté en la misma carpeta que este script
model = YOLO("best.pt")

# 2. Configurar la cámara web de la laptop
cap = cv2.VideoCapture(0)

print("Presiona 'q' para salir del prototipo.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Realizar la detección (Inferencia)
    # conf=0.5 establece el umbral de confianza al 50%
    results = model(frame, conf=0.5, stream=True)

    # 4. Dibujar los resultados en el video
    for r in results:
        annotated_frame = r.plot()  # Dibuja cuadros y etiquetas automáticamente

    # 5. Mostrar la ventana
    cv2.imshow("Prototipo Deteccion de Plantas ", annotated_frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()