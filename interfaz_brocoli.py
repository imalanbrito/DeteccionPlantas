"""
Interfaz gráfica de usuario (GUI) LUMACAD.
Diseño moderno redimensionable, paleta de colores verde/blanco 
y resolución original de cámara (640x480).
"""

import cv2
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image
from ultralytics import YOLO

# --- Configuración global de la apariencia ---
ctk.set_appearance_mode("light")  # Fondo principal claro/blanco
ctk.set_default_color_theme("green")  # Acentos, botones y barras en verde

class InterfazBrocoli(ctk.CTk):
    """
    Clase principal que construye y maneja la ventana de la aplicación LUMACAD.
    """
    def __init__(self):
        super().__init__()

        # --- Configuración de la ventana ---
        self.title("LUMACAD Deteccion de plantas")
        self.geometry("800x750") # Tamaño inicial amplio
        self.resizable(True, True) # Permite ajustar el tamaño de la ventana libremente

        # --- 1. Cargar el modelo de IA ---
        try:
            self.model = YOLO("best.pt")
        except Exception as e:
            messagebox.showerror("Error crítico", f"No se encontró 'best.pt': {e}")
            self.destroy()
            return

        # --- 2. Inicializar la cámara ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se detectó cámara web.")
            self.destroy()
            return

        # --- 3. Construcción de Elementos Visuales ---
        
        # Título superior
        self.lbl_titulo = ctk.CTkLabel(self, text="LUMACAD Detección de Plantas", 
                                       font=ctk.CTkFont(family="Arial", size=28, weight="bold"),
                                       text_color="#1E8449") # Tono verde oscuro institucional
        self.lbl_titulo.pack(pady=(20, 15))

        # Tarjeta (Frame) para el video con fondo blanco puro y borde verde sutil
        self.frame_video = ctk.CTkFrame(self, fg_color="white", corner_radius=10, 
                                        border_width=2, border_color="#2ECC71")
        self.frame_video.pack(padx=20, pady=10)

        # Etiqueta que contendrá los cuadros de video (Resolución original 640x480)
        self.lbl_video = ctk.CTkLabel(self.frame_video, text="")
        self.lbl_video.pack(padx=10, pady=10)

        # Contenedor para alinear las métricas lado a lado
        self.frame_metricas = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_metricas.pack(padx=20, pady=10, fill="x", expand=True)

        # Tarjeta para la sección de detecciones (Fondo verde muy claro)
        self.frame_det = ctk.CTkFrame(self.frame_metricas, fg_color="#EAF8E8", corner_radius=10) 
        self.frame_det.pack(side="left", padx=(0, 10), fill="both", expand=True)
        
        self.lbl_titulo_det = ctk.CTkLabel(self.frame_det, text="Detecciones Actuales", 
                                           font=ctk.CTkFont(size=16, weight="bold"), text_color="#27AE60")
        self.lbl_titulo_det.pack(pady=(10, 0))
        
        self.lbl_detecciones = ctk.CTkLabel(self.frame_det, text="Iniciando cámara...", 
                                            font=ctk.CTkFont(size=14), text_color="#333333")
        self.lbl_detecciones.pack(pady=(5, 10))

        # Tarjeta para la sección de rendimiento (Fondo verde muy claro)
        self.frame_vel = ctk.CTkFrame(self.frame_metricas, fg_color="#EAF8E8", corner_radius=10)
        self.frame_vel.pack(side="right", padx=(10, 0), fill="both", expand=True)

        self.lbl_titulo_vel = ctk.CTkLabel(self.frame_vel, text="Rendimiento del Sistema", 
                                           font=ctk.CTkFont(size=16, weight="bold"), text_color="#27AE60")
        self.lbl_titulo_vel.pack(pady=(10, 0))

        self.lbl_velocidad = ctk.CTkLabel(self.frame_vel, text="Calculando...", 
                                          font=ctk.CTkFont(size=14), text_color="#333333")
        self.lbl_velocidad.pack(pady=(5, 10))

        # Botón principal redondeado en la parte inferior
        self.btn_salir = ctk.CTkButton(self, text="Detener Sistema", command=self.cerrar_aplicacion,
                                       fg_color="#E74C3C", hover_color="#C0392B", # Tonos rojos para detener
                                       font=ctk.CTkFont(size=16, weight="bold"),
                                       height=45, corner_radius=20)
        self.btn_salir.pack(side="bottom", pady=20, padx=20, fill="x")

        # --- 4. Arrancar el motor de video ---
        self.actualizar_video()

    def actualizar_video(self):
        exito, frame = self.cap.read()
        
        if exito:
            # 1. Realizar inferencia
            resultados = self.model(frame, conf=0.5, stream=False, verbose=False) 
            
            for r in resultados:
                frame_anotado = r.plot()
                
                # 2. Datos de rendimiento
                speed = r.speed 
                texto_velocidad = f"Pre: {speed['preprocess']:.1f}ms | IA: {speed['inference']:.1f}ms | Post: {speed['postprocess']:.1f}ms"
                
                # 3. Conteo de clases
                clases_detectadas = [int(cls) for cls in r.boxes.cls.tolist()]
                
                if len(clases_detectadas) == 0:
                    texto_detecciones = "Área libre de objetivos"
                else:
                    conteo = {}
                    for id_clase in clases_detectadas:
                        nombre_clase = r.names[id_clase].replace('_', ' ').title()
                        conteo[nombre_clase] = conteo.get(nombre_clase, 0) + 1
                    
                    resumen = " • ".join([f"{cantidad} {nombre}" for nombre, cantidad in conteo.items()])
                    texto_detecciones = resumen

                # 4. Actualizar textos en la interfaz
                self.lbl_detecciones.configure(text=texto_detecciones)
                self.lbl_velocidad.configure(text=texto_velocidad)
            
            # 5. Convertir imagen para CustomTkinter
            img_rgb = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)
            
            # Mantener la resolución original solicitada (640x480)
            img_resized = cv2.resize(img_rgb, (640, 480)) 
            img_pil = Image.fromarray(img_resized)
            
            # Cargar la imagen en la interfaz con el tamaño exacto
            self.ctk_image = ctk.CTkImage(light_image=img_pil, size=(640, 480))
            self.lbl_video.configure(image=self.ctk_image)

        # 6. Bucle de actualización (10ms)
        self.after(10, self.actualizar_video)

    def cerrar_aplicacion(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = InterfazBrocoli()
    app.mainloop()