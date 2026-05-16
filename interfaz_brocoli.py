import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

class InterfazBrocoli:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Detección de Brócoli - Invernadero")
        self.ventana.configure(bg="#2c3e50")

        # --- 1. Cargar el modelo ---
        try:
            self.model = YOLO("best.pt")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo: {e}")
            self.ventana.destroy()
            return

        # --- 2. Inicializar la cámara ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se detectó cámara web.")
            self.ventana.destroy()
            return

        # --- 3. Elementos de la Interfaz ---
        
        # Título
        titulo = tk.Label(ventana, text="Monitor de Cultivo en Tiempo Real", font=("Arial", 16, "bold"), bg="#2c3e50", fg="white")
        titulo.pack(pady=10)

        # Lienzo para el video
        self.canvas = tk.Canvas(ventana, width=640, height=480, bg="black", highlightthickness=2, highlightbackground="#34495e")
        self.canvas.pack()

        # Panel de métricas (donde irán los datos de YOLO)
        frame_metricas = tk.Frame(ventana, bg="#34495e", padx=10, pady=10)
        frame_metricas.pack(fill=tk.X, padx=20, pady=10)

        self.lbl_detecciones = tk.Label(frame_metricas, text="Detecciones: Esperando...", font=("Consolas", 11), bg="#34495e", fg="#00FF00", anchor="w")
        self.lbl_detecciones.pack(fill=tk.X)

        self.lbl_velocidad = tk.Label(frame_metricas, text="Velocidad: Esperando...", font=("Consolas", 11), bg="#34495e", fg="#00FF00", anchor="w")
        self.lbl_velocidad.pack(fill=tk.X)

        # Botón para salir
        self.btn_salir = tk.Button(ventana, text="Detener Sistema", command=self.cerrar_aplicacion, bg="#e74c3c", fg="white", font=("Arial", 12, "bold"), relief=tk.FLAT)
        self.btn_salir.pack(pady=10)

        # --- 4. Iniciar el bucle ---
        self.actualizar_video()

    def actualizar_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Realizar inferencia
            results = self.model(frame, conf=0.5, stream=False, verbose=False) # stream=False es mejor para obtener los datos completos del frame
            
            for r in results:
                # 1. Dibujar resultados en la imagen
                annotated_frame = r.plot()
                
                # 2. Extraer datos de rendimiento (Velocidad)
                speed = r.speed # Diccionario con 'preprocess', 'inference', 'postprocess'
                speed_text = f"Speed: {speed['preprocess']:.1f}ms preprocess, {speed['inference']:.1f}ms inference, {speed['postprocess']:.1f}ms postprocess"
                
                # 3. Extraer resumen de detecciones
                # r.names contiene el diccionario de clases {0: 'brocoli_sano', 1: 'maleza'...}
                # r.boxes.cls contiene las clases detectadas en este frame
                clases_detectadas = [int(cls) for cls in r.boxes.cls.tolist()]
                
                if len(clases_detectadas) == 0:
                    det_text = "0: 480x640 (no detections)"
                else:
                    # Contar cuántas de cada clase hay
                    conteo = {}
                    for cls_id in clases_detectadas:
                        nombre_clase = r.names[cls_id]
                        conteo[nombre_clase] = conteo.get(nombre_clase, 0) + 1
                    
                    # Formatear el texto "0: 480x640 2 brocoli_sano, 1 maleza"
                    resumen_clases = ", ".join([f"{count} {name}" for name, count in conteo.items()])
                    det_text = f"0: 480x640 {resumen_clases}"

                # 4. Actualizar las etiquetas en la interfaz
                self.lbl_detecciones.config(text=det_text)
                self.lbl_velocidad.config(text=speed_text)
            
            # 5. Convertir imagen para mostrar en Tkinter
            img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            self.photo = ImageTk.PhotoImage(image=img_pil)
            
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Repetir
        self.ventana.after(10, self.actualizar_video)

    def cerrar_aplicacion(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.ventana.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("700x700")
    app = InterfazBrocoli(root)
    root.mainloop()