import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import os
import glob
import sys
import json

# ==========================================
# 1. CONFIGURACIÓN Y RUTAS BASE
# ==========================================
VIDEO_PATH = "videos_prueba/IMG_1.mp4" 
CARPETA_SALIDA = "resultados_videos"
MODELO_YOLO = "models/yolov8x.pt"

# ==========================================
# 2. CONFIGURACIÓN MATEMÁTICA DEL RADAR 2D
# ==========================================
# A. Puntos originales del video
SOURCE_PTS = np.array([
    [121, 159],   # Arriba a la Izquierda
    [1078, 145],  # Arriba a la Derecha
    [1078, 1888], # Abajo a la Derecha
    [0, 1624]     # Abajo a la Izquierda
], dtype=np.float32)

# B. Dimensiones virtuales del Radar en nuestro Dashboard (en píxeles)
RADAR_W = 500
RADAR_H = 800

# C. Puntos de destino
TARGET_PTS = np.array([
    [0, 0],               # Arriba a la Izquierda
    [RADAR_W, 0],         # Arriba a la Derecha
    [RADAR_W, RADAR_H],   # Abajo a la Derecha
    [0, RADAR_H]          # Abajo a la Izquierda
], dtype=np.float32)

# D. Calcular la Matriz de Transformación (El núcleo matemático)
MATRIZ_H = cv2.getPerspectiveTransform(SOURCE_PTS, TARGET_PTS)

# ==========================================
# 3. FUNCIONES AUXILIARES
# ==========================================
def generar_paths_salida(carpeta, nombre_base="dashboard"):
    os.makedirs(carpeta, exist_ok=True)
    archivos = glob.glob(os.path.join(carpeta, f"{nombre_base}_*.mp4"))
    
    max_id = 0
    for arch in archivos:
        nombre = os.path.basename(arch)
        try:
            num = int(nombre.replace(f"{nombre_base}_", "").replace(".mp4", ""))
            if num > max_id: max_id = num
        except ValueError: pass
            
    nuevo_id = max_id + 1
    paths = {
        "video": os.path.join(carpeta, f"{nombre_base}_{nuevo_id}.mp4"),
        "json":  os.path.join(carpeta, f"{nombre_base}_{nuevo_id}.json"),
        "out":   os.path.join(carpeta, f"{nombre_base}_{nuevo_id}.out")
    }
    return paths, nuevo_id

def crear_cancha_radar(w, h):
    """Dibuja el fondo verde y las líneas del minimapa 2D"""
    cancha = np.zeros((h, w, 3), dtype=np.uint8)
    cancha[:] = (40, 130, 40) # Color verde oscuro
    cv2.rectangle(cancha, (0, 0), (w, h), (255, 255, 255), 4) # Borde
    cv2.line(cancha, (0, h//2), (w, h//2), (255, 255, 255), 4) # Línea central
    cv2.circle(cancha, (w//2, h//2), 50, (255, 255, 255), 4) # Círculo central
    return cancha

# ==========================================
# 4. FUNCIÓN PRINCIPAL
# ==========================================
def main():
    paths, run_id = generar_paths_salida(CARPETA_SALIDA, "dashboard")
    sys.stdout = open(paths["out"], 'w')
    sys.stderr = sys.stdout
    
    print(f"--- Iniciando Ejecución ID: {run_id} ---")
    model = YOLO(MODELO_YOLO)
    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
    
    dashboard_width = 1920
    dashboard_height = 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(paths["video"], fourcc, video_info.fps, (dashboard_width, dashboard_height))
    
    datos_partido = {"run_id": run_id, "fps": video_info.fps, "frames": []}
    frame_count = 0
    
    for frame in sv.get_video_frames_generator(VIDEO_PATH):
        frame_count += 1
        
        # --- A. INFERENCIA CON YOLO ---
        resultados = model(frame, verbose=False)[0]
        detecciones = sv.Detections.from_ultralytics(resultados)
        
        # --- B. LIENZO Y VIDEO ORIGINAL ---
        dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
        alto_orig, ancho_orig = frame.shape[:2]
        escala = dashboard_height / alto_orig
        nuevo_ancho = int(ancho_orig * escala)
        frame_proporcional = cv2.resize(frame, (nuevo_ancho, dashboard_height))
        dashboard[0:dashboard_height, 0:nuevo_ancho] = frame_proporcional
        
        # --- C. RADAR TÁCTICO 2D ---
        radar_canvas = crear_cancha_radar(RADAR_W, RADAR_H)
        
        # Procesar cada objeto detectado
        for bbox in detecciones.xyxy:
            x1, y1, x2, y2 = bbox
            # 1. Tomamos el punto medio inferior de la caja (donde toca el suelo)
            punto_suelo_x = (x1 + x2) / 2
            punto_suelo_y = y2
            
            # 2. Aplicamos la Matriz de Homografía a ese punto
            punto_original = np.array([[[punto_suelo_x, punto_suelo_y]]], dtype=np.float32)
            punto_transformado = cv2.perspectiveTransform(punto_original, MATRIZ_H)
            
            radar_x = int(punto_transformado[0][0][0])
            radar_y = int(punto_transformado[0][0][1])
            
            # 3. Dibujamos el punto en el radar si está dentro de los límites
            if 0 <= radar_x <= RADAR_W and 0 <= radar_y <= RADAR_H:
                cv2.circle(radar_canvas, (radar_x, radar_y), 12, (0, 0, 255), -1) # Punto rojo
                cv2.circle(radar_canvas, (radar_x, radar_y), 12, (255, 255, 255), 2) # Borde blanco
        
        # Pegar el Radar en la columna central del Dashboard
        x_offset = nuevo_ancho + 100
        y_offset = 100
        dashboard[y_offset:y_offset+RADAR_H, x_offset:x_offset+RADAR_W] = radar_canvas
        
        out.write(dashboard)
            
    out.release()
    with open(paths["json"], 'w') as f:
        json.dump(datos_partido, f, indent=4)
        
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f"Ejecución {run_id} completada. Radar 2D integrado.")

if __name__ == "__main__":
    main()