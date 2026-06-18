import cv2
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================
VIDEO_PATH = "videos_prueba/IMG_2.MOV" 
CARPETA_DATASET = "Dataset"
OBJETIVO_IMAGENES = 250

def main():
    # Crea la carpeta Dataset si no existe
    os.makedirs(CARPETA_DATASET, exist_ok=True)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video abierto. Total de frames detectados: {total_frames}")
    
    # Calcula el salto exacto para obtener el objetivo de imágenes a lo largo de todo el video
    salto_frames = max(1, total_frames // OBJETIVO_IMAGENES)
    print(f"Se extraerá 1 frame cada {salto_frames} fotogramas.")
    print("Iniciando extracción...")

    count = 0
    guardados = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Fin del video
            
        # Extraer fotograma si coincide con el salto y no hemos llegado al límite
        if count % salto_frames == 0 and guardados < OBJETIVO_IMAGENES:
            nombre_archivo = os.path.join(CARPETA_DATASET, f"futbolito_{guardados:04d}.jpg")
            cv2.imwrite(nombre_archivo, frame)
            guardados += 1
            
        count += 1

    cap.release()
    print(f"¡Listo! Se guardaron {guardados} imágenes en la carpeta '{CARPETA_DATASET}'.")

if __name__ == "__main__":
    main()