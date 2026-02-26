import cv2
import numpy as np
import tensorflow as tf

# 1. Cargar el modelo y etiquetas
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
labels = ["Brocoli Sano", "Brocoli Enfermo", "Maleza"] # Deben ir en el mismo orden que en Teachable Machine

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 2. Preprocesar imagen (1280x720 -> 224x224 para el modelo)
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # 3. Predicción
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    index = np.argmax(output_data[0])
    
    # 4. Mostrar resultado
    texto = f"{labels[index]}: {output_data[0][index]*100:.1f}%"
    cv2.putText(frame, texto, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Deteccion de Plantas', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()