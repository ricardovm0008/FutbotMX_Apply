import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import os
import glob
import sys
import json
from collections import deque

# ==========================================
# 1. CONFIGURACIÓN Y RUTAS BASE
# ==========================================
VIDEO_PATH = "videos_prueba/IMG_1.mp4" 
CARPETA_SALIDA = "resultados_videos"
MODELO_YOLO = "models/yolov8x.pt"

# ==========================================
# 2. CONFIGURACIÓN MATEMÁTICA DEL RADAR 2D
# ==========================================
SOURCE_PTS = np.array([
    [121, 159],   
    [1078, 145],  
    [1078, 1888], 
    [0, 1624]     
], dtype=np.float32)

RADAR_W = 500
RADAR_H = 800

TARGET_PTS = np.array([
    [0, 0],               
    [RADAR_W, 0],         
    [RADAR_W, RADAR_H],   
    [0, RADAR_H]          
], dtype=np.float32)

MATRIZ_H = cv2.getPerspectiveTransform(SOURCE_PTS, TARGET_PTS)

# ==========================================
# 3. FUNCIONES AUXILIARES
# ==========================================
def generar_paths_salida(carpeta, nombre_base="dashboard"):
    """Genera identificadores únicos para no sobreescribir ejecuciones"""
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
    """Dibuja una cancha de fútbol robótico estilizada"""
    cancha = np.zeros((h, w, 3), dtype=np.uint8)
    cancha[:] = (45, 135, 45) # Verde pasto

    color_linea = (255, 255, 255)
    grosor = 3
    margen = 30

    # Borde, línea central y círculo
    cv2.rectangle(cancha, (margen, margen), (w - margen, h - margen), color_linea, grosor)
    cv2.line(cancha, (margen, h//2), (w - margen, h//2), color_linea, grosor)
    cv2.circle(cancha, (w//2, h//2), 60, color_linea, grosor)
    cv2.circle(cancha, (w//2, h//2), 6, color_linea, -1)

    # Áreas
    ancho_area = 200
    alto_area = 100
    x_area = (w - ancho_area) // 2
    cv2.rectangle(cancha, (x_area, margen), (x_area + ancho_area, margen + alto_area), color_linea, grosor) 
    cv2.rectangle(cancha, (x_area, h - margen - alto_area), (x_area + ancho_area, h - margen), color_linea, grosor) 

    # Porterías
    ancho_porteria = 120
    alto_porteria = 15
    x_port = (w - ancho_porteria) // 2
    cv2.rectangle(cancha, (x_port, margen - alto_porteria), (x_port + ancho_porteria, margen), (0, 100, 255), -1)
    cv2.rectangle(cancha, (x_port, h - margen), (x_port + ancho_porteria, h - margen + alto_porteria), (0, 100, 255), -1)

    return cancha

# ==========================================
# 4. FUNCIÓN PRINCIPAL
# ==========================================
def main():
    paths, run_id = generar_paths_salida(CARPETA_SALIDA, "dashboard")
    
    # Redirigir logs al archivo .out para limpiar la terminal
    sys.stdout = open(paths["out"], 'w')
    sys.stderr = sys.stdout
    
    print(f"--- Iniciando Ejecución ID: {run_id} ---")
    
    # Cargar YOLO en la GPU
    model = YOLO(MODELO_YOLO)
    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
    
    # Anotadores visuales estilizados
    corner_annotator = sv.BoxCornerAnnotator(thickness=4, color=sv.Color(r=255, g=140, b=0))
    label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=2, color=sv.Color(r=255, g=140, b=0))
    
    # Configuración del Video Final
    dashboard_width = 1920
    dashboard_height = 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(paths["video"], fourcc, video_info.fps, (dashboard_width, dashboard_height))
    
    trayectoria_pelota = deque(maxlen=45)
    datos_partido = {"run_id": run_id, "fps": video_info.fps, "frames": []}
    frame_count = 0
    
    for frame in sv.get_video_frames_generator(VIDEO_PATH):
        frame_count += 1
        
        # --- A. INFERENCIA YOLO ---
        resultados_yolo = model(frame, verbose=False)[0]
        
        # --- B. DETECCIÓN DE LA PELOTA (COLOR + FORMA) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        naranja_bajo = np.array([5, 120, 120])
        naranja_alto = np.array([25, 255, 255])
        
        mascara_naranja = cv2.inRange(hsv, naranja_bajo, naranja_alto)
        contornos, _ = cv2.findContours(mascara_naranja, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        radar_canvas = crear_cancha_radar(RADAR_W, RADAR_H)
        pelota_encontrada = False
        
        if contornos:
            # Ordenar contornos de mayor a menor tamaño
            contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
            
            for contorno in contornos:
                area = cv2.contourArea(contorno)
                
                # Descartar ruido pequeño
                if area > 50:
                    perimetro = cv2.arcLength(contorno, True)
                    if perimetro == 0:
                        continue
                        
                    # Filtro de Circularidad (descarta manos/brazos)
                    circularidad = 4 * np.pi * (area / (perimetro * perimetro))
                    
                    if 0.75 < circularidad <= 1.2:
                        x, y, w, h = cv2.boundingRect(contorno)
                        
                        # Convertir a formato supervision
                        caja_pelota = np.array([[x, y, x + w, y + h]])
                        deteccion_pelota = sv.Detections(
                            xyxy=caja_pelota,
                            class_id=np.array([0]),
                            confidence=np.array([circularidad])
                        )
                        
                        # Aplicar anotadores al video
                        etiquetas = [f"Pelota ({circularidad:.2f})"]
                        frame = corner_annotator.annotate(scene=frame, detections=deteccion_pelota)
                        frame = label_annotator.annotate(scene=frame, detections=deteccion_pelota, labels=etiquetas)
                        
                        # Extraer coordenadas para el radar
                        pelota_encontrada = True
                        punto_centro_x = x + (w / 2)
                        punto_centro_y = y + (h / 2)
                        break 
        
        # --- C. RADAR TÁCTICO 2D ---
        if pelota_encontrada:
            punto_original = np.array([[[punto_centro_x, punto_centro_y]]], dtype=np.float32)
            punto_transformado = cv2.perspectiveTransform(punto_original, MATRIZ_H)
            
            radar_x = int(punto_transformado[0][0][0])
            radar_y = int(punto_transformado[0][0][1])
            
            if 0 <= radar_x <= RADAR_W and 0 <= radar_y <= RADAR_H:
                trayectoria_pelota.append((radar_x, radar_y))
                
        # Dibujar estela naranja
        for i in range(1, len(trayectoria_pelota)):
            grosor_estela = int(np.sqrt(45 / float(len(trayectoria_pelota) - i + 1)) * 2)
            cv2.line(radar_canvas, trayectoria_pelota[i - 1], trayectoria_pelota[i], (0, 140, 255), grosor_estela)
            
        # Dibujar posición actual
        if len(trayectoria_pelota) > 0:
            cv2.circle(radar_canvas, trayectoria_pelota[-1], 10, (0, 100, 255), -1) 
            cv2.circle(radar_canvas, trayectoria_pelota[-1], 10, (255, 255, 255), 2)
        
        # --- D. ENSAMBLAJE DEL DASHBOARD ---
        dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
        
        alto_orig, ancho_orig = frame.shape[:2]
        escala = dashboard_height / alto_orig
        nuevo_ancho = int(ancho_orig * escala)
        frame_proporcional = cv2.resize(frame, (nuevo_ancho, dashboard_height))
        
        dashboard[0:dashboard_height, 0:nuevo_ancho] = frame_proporcional
        
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
    print(f"Ejecución {run_id} completada.")

if __name__ == "__main__":
    main()