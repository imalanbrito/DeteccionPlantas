import os
from PIL import Image

# Configuración
carpetas = ['dataset/entrenamiento/brocoli_sano', 'dataset/entrenamiento/brocoli_enfermo', 'dataset/entrenamiento/maleza']
resolucion = (1280, 720)

for ruta in carpetas:
    if not os.path.exists(ruta): continue
    print(f"Procesando: {ruta}")
    for archivo in os.listdir(ruta):
        if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(ruta, archivo)
            img = Image.open(img_path)
            # Redimensionar manteniendo calidad
            img_resized = img.resize(resolucion, Image.Resampling.LANCZOS)
            img_resized.save(img_path)
            print(f"Reajustado: {archivo}")

print("¡Listo! Todas las imágenes son ahora de 1280x720.")