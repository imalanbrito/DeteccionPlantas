import cv2
import os

# Asegúrate de que la ruta tenga las comillas de cierre y el paréntesis
ruta = 'dataset/entrenamiento/brocoli_sano'
archivos = os.listdir(ruta)

for nombre in archivos:
    img_path = os.path.join(ruta, nombre)
    frame = cv2.imread(img_path)
    
    if frame is not None:
        cv2.imshow('Verificacion', frame)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()