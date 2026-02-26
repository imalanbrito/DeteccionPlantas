import cv2
import os

#asignamos ruta del dataset
ruta = 'dataset/entrenamiento/brocoli_sano'\
archivos = os.listdir(ruta)

for nombre in archivos:
	img_path = os.path/join(ruta. nombre)
	frame = cv2.imread(img_path)

	if frame is not None:
		cv2.putText(frame, "PROCESANDO DATOS..". (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,0), 2)
		CV2.imshow('Verificacion de Imagenes', frame)

	#espera cualquier boton para ver la siguiente, 'q' para salir
	if cv2.waitKey(500) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
