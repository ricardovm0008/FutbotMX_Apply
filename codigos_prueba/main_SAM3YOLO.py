import cv2
import numpy as np
from ultralytics import YOLO, SAM
import supervision as sv
import os
import glob
import sys
import json
from collections import deque

# ==========================================
# 1. CONFIGURACIÓN Y RUTAS BASE
# ==========================================
VIDEO_PATH = "videos_prueba/IMG_2.MOV"
CARPETA_SALIDA = "resultados_videos"
MODELO_YOLO = "runs/detect/modelo_propio/futbolito_v1-3/weights/best.pt"
MODELO_SAM = "models/sam3.pt"

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
    cancha = np.zeros((h, w, 3), dtype=np.uint8)
    cancha[:] = (45, 135, 45)  # Verde pasto

    color_linea = (255, 255, 255)
    grosor = 3
    margen = 30

    cv2.rectangle(cancha, (margen, margen), (w - margen, h - margen), color_linea, grosor)
    cv2.line(cancha, (margen, h//2), (w - margen, h//2), color_linea, grosor)
    cv2.circle(cancha, (w//2, h//2), 60, color_linea, grosor)
    cv2.circle(cancha, (w//2, h//2), 6, color_linea, -1)

    ancho_area = 200
    alto_area = 100
    x_area = (w - ancho_area) // 2
    cv2.rectangle(cancha, (x_area, margen), (x_area + ancho_area, margen + alto_area), color_linea, grosor)
    cv2.rectangle(cancha, (x_area, h - margen - alto_area), (x_area + ancho_area, h - margen), color_linea, grosor)

    ancho_porteria = 120
    alto_porteria = 15
    x_port = (w - ancho_porteria) // 2
    
    cv2.rectangle(cancha, (x_port, margen - alto_porteria), (x_port + ancho_porteria, margen), (0, 100, 255), -1)
    cv2.rectangle(cancha, (x_port, h - margen), (x_port + ancho_porteria, h - margen + alto_porteria), (0, 100, 255), -1)
    
    cv2.rectangle(cancha, (x_port, margen - alto_porteria), (x_port + ancho_porteria, margen), (0, 255, 255), 1)
    cv2.rectangle(cancha, (x_port, h - margen), (x_port + ancho_porteria, h - margen + alto_porteria), (0, 255, 255), 1)

    return cancha

# ==========================================
# 4. FUNCIÓN PRINCIPAL
# ==========================================
def main():
    paths, run_id = generar_paths_salida(CARPETA_SALIDA, "dashboard")
    sys.stdout = open(paths["out"], 'w')
    sys.stderr = sys.stdout
    print(f"--- Iniciando Ejecución ID: {run_id} ---")

    # Inicializamos ambos modelos en la GPU
    model_yolo = YOLO(MODELO_YOLO)
    model_sam = SAM(MODELO_SAM)
    
    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)

    # CAMBIO DE ANOTADOR: Usamos MaskAnnotator para las siluetas de SAM 3
    mask_annotator = sv.MaskAnnotator(opacity=0.5) 
    label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=1)

    dashboard_height = 1080
    ancho_orig = video_info.width
    alto_orig = video_info.height
    escala = dashboard_height / alto_orig
    nuevo_ancho = int(ancho_orig * escala)          
    x_offset_radar = nuevo_ancho + 100              
    dashboard_width = x_offset_radar + RADAR_W      
    print(f"Tamaño del dashboard: {dashboard_width}x{dashboard_height}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(paths["video"], fourcc, video_info.fps, (dashboard_width, dashboard_height))

    trayectoria_pelota = deque(maxlen=45)
    historial_equipos = {}          
    marcador = {"Equipo A": 0, "Equipo B": 0}
    cooldown_gol = 0 
    FPS = int(video_info.fps)
    
    ZONA_GOL_ARRIBA_Y = 30 
    ZONA_GOL_ABAJO_Y = RADAR_H - 30 
    X_GOL_MIN = 190
    X_GOL_MAX = 310

    datos_partido = {
        "run_id": run_id,
        "fps": video_info.fps,
        "frames": []
    }
    contador_frame = 0

    for frame in sv.get_video_frames_generator(VIDEO_PATH):
        # 1. RASTREO CON YOLOv8
        resultados = model_yolo.track(frame, persist=True, verbose=False)[0]
        detecciones = sv.Detections.from_ultralytics(resultados)

        # 2. PROCESAMIENTO DE SILUETAS CON SAM 3
        # Pasamos las cajas detectadas por YOLO como guías para que SAM extraiga las máscaras exactas
        if len(detecciones) > 0 and detecciones.xyxy.size > 0:
            sam_results = model_sam(frame, bboxes=detecciones.xyxy, verbose=False)[0]
            if sam_results.masks is not None:
                # Inyectamos las máscaras de píxeles dentro de nuestro objeto de detecciones
                detecciones.mask = sam_results.masks.data.cpu().numpy().astype(bool)

        radar_canvas = crear_cancha_radar(RADAR_W, RADAR_H)
        frame_data = {
            "frame": contador_frame,
            "marcador": dict(marcador),
            "pelota": None,
            "robots": []
        }

        # Procesar datos y radar táctico
        for i in range(len(detecciones)):
            class_id = detecciones.class_id[i]
            conf = float(detecciones.confidence[i])
            track_id = int(detecciones.tracker_id[i]) if detecciones.tracker_id is not None else -1

            x1, y1, x2, y2 = detecciones.xyxy[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            punto_orig = np.array([[[cx, cy]]], dtype=np.float32)
            punto_trans = cv2.perspectiveTransform(punto_orig, MATRIZ_H)
            
            if punto_trans is not None:
                rx = int(punto_trans[0][0][0])
                ry = int(punto_trans[0][0][1])
            else:
                rx, ry = -1, -1

            # Calcular área precisa en píxeles usando la máscara de SAM 3
            area_segmentada = 0
            if detecciones.mask is not None:
                area_segmentada = int(np.sum(detecciones.mask[i]))

            # --- Robots ---
            if class_id == 2:
                if track_id != -1:
                    if track_id not in historial_equipos:
                        if ry < RADAR_H / 2:
                            historial_equipos[track_id] = "Equipo A"
                        else:
                            historial_equipos[track_id] = "Equipo B"
                    equipo = historial_equipos[track_id]

                    if 0 <= rx <= RADAR_W and 0 <= ry <= RADAR_H:
                        color = (240, 130, 40) if equipo == "Equipo A" else (50, 50, 240)  
                        cv2.circle(radar_canvas, (rx, ry), 16, color, -1)
                        cv2.circle(radar_canvas, (rx, ry), 16, (255, 255, 255), 2)
                        cv2.putText(radar_canvas, str(track_id), (rx - 6, ry + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

                    # ENTRADA JSON ENRIQUECIDA CON DATOS DE SAM 3
                    frame_data["robots"].append({
                        "track_id": track_id,
                        "equipo": equipo,
                        "x_radar": rx if (0 <= rx <= RADAR_W) else None,
                        "y_radar": ry if (0 <= ry <= RADAR_H) else None,
                        "confianza": conf,
                        "area_píxeles_sam": area_segmentada,  # Métrica exacta de tamaño de la silueta
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })

            # --- Pelota ---
            elif class_id == 0:
                if 0 <= rx <= RADAR_W and 0 <= ry <= RADAR_H:
                    trayectoria_pelota.append((rx, ry))
                    frame_data["pelota"] = {
                        "x_radar": rx,
                        "y_radar": ry,
                        "confianza": conf,
                        "area_píxeles_sam": area_segmentada
                    }
                    
                    if cooldown_gol == 0:
                        if X_GOL_MIN <= rx <= X_GOL_MAX:
                            if ry <= ZONA_GOL_ARRIBA_Y:
                                marcador["Equipo B"] += 1
                                cooldown_gol = FPS * 4 
                            elif ry >= ZONA_GOL_ABAJO_Y:
                                marcador["Equipo A"] += 1
                                cooldown_gol = FPS * 4

        if cooldown_gol > 0:
            cooldown_gol -= 1

        for i in range(1, len(trayectoria_pelota)):
            grosor = int(np.sqrt(45 / float(len(trayectoria_pelota) - i + 1)) * 2)
            cv2.line(radar_canvas, trayectoria_pelota[i - 1], trayectoria_pelota[i], (0, 140, 255), grosor)
        if trayectoria_pelota:
            cv2.circle(radar_canvas, trayectoria_pelota[-1], 9, (0, 100, 255), -1)
            cv2.circle(radar_canvas, trayectoria_pelota[-1], 9, (255, 255, 255), 2)

        # Paleta de Colores en formato Supervision (RGB)
        COLOR_EQUIPO_A = sv.Color(r=240, g=130, b=40)  
        COLOR_EQUIPO_B = sv.Color(r=50, g=50, b=240)    
        COLOR_PELOTA   = sv.Color(r=0, g=165, b=255)    
        COLOR_PORTERIA = sv.Color(r=255, g=255, b=0) # Amarillo vibrante solicitado

        # RENDERIZADO SECUENCIAL DE MÁSCARAS (SAM 3)
        for i in range(len(detecciones)):
            single_det = detecciones[i] 
            c_id = single_det.class_id[0]
            t_id = int(single_det.tracker_id[0]) if single_det.tracker_id is not None else -1

            if c_id == 2:  
                if t_id in historial_equipos:
                    eq = historial_equipos[t_id]
                    color_actual = COLOR_EQUIPO_A if eq == "Equipo A" else COLOR_EQUIPO_B
                    label_text = f"{eq} #{t_id}"
                else:
                    color_actual = sv.Color.WHITE
                    label_text = "Robot"
            elif c_id == 0:  
                color_actual = COLOR_PELOTA
                label_text = "Pelota"
            elif c_id == 1:  
                color_actual = COLOR_PORTERIA
                label_text = "Porteria"
            else:
                color_actual = sv.Color.WHITE
                label_text = ""

            # Configuramos color y dibujamos la máscara exacta sobre el frame original
            mask_annotator.color = color_actual
            label_annotator.color = color_actual
            label_annotator.text_color = sv.Color.WHITE

            frame = mask_annotator.annotate(scene=frame, detections=single_det)
            frame = label_annotator.annotate(scene=frame, detections=single_det, labels=[label_text])

        # Ensamble del Lienzo
        dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
        frame_escalado = cv2.resize(frame, (nuevo_ancho, dashboard_height))
        dashboard[0:dashboard_height, 0:nuevo_ancho] = frame_escalado
        
        y_radar = 150 
        dashboard[y_radar:y_radar + RADAR_H, x_offset_radar:x_offset_radar + RADAR_W] = radar_canvas

        # ===================================================
        # SOLUCIÓN AL TRASLAPE DEL MARCADOR DIGITAL (REDISEÑO)
        # ===================================================
        alto_marcador = 80
        y_marcador = 35
        
        # Base oscura del marcador
        cv2.rectangle(dashboard, (x_offset_radar, y_marcador), (x_offset_radar + RADAR_W, y_marcador + alto_marcador), (20, 20, 20), -1)
        cv2.rectangle(dashboard, (x_offset_radar, y_marcador), (x_offset_radar + RADAR_W, y_marcador + alto_marcador), (100, 100, 100), 2)
        
        # 1. Ajuste Nombres: Fuente compacta y pegada a los extremos (Offset seguro de 15px)
        cv2.putText(dashboard, "EQUIPO A", (x_offset_radar + 15, y_marcador + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 130, 40), 2)
        cv2.putText(dashboard, "EQUIPO B", (x_offset_radar + RADAR_W - 125, y_marcador + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 240), 2)
        
        # 2. Ajuste Score: Reducción de espacios a un estándar "X - X" y cálculo dinámico de centrado
        texto_puntos = f"{marcador['Equipo A']} - {marcador['Equipo B']}"
        text_size = cv2.getTextSize(texto_puntos, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)[0]
        tx = x_offset_radar + (RADAR_W - text_size[0]) // 2  # Centrado matemático milimétrico
        
        color_puntos = (0, 255, 0) if (cooldown_gol > 0 and (cooldown_gol // 10) % 2 == 0) else (255, 255, 255)
        cv2.putText(dashboard, texto_puntos, (tx, y_marcador + 53), cv2.FONT_HERSHEY_DUPLEX, 1.1, color_puntos, 2)

        out.write(dashboard)
        datos_partido["frames"].append(frame_data)
        contador_frame += 1

    out.release()
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f"Ejecución completada. Eventos narrados y robots rastreados visualmente.")

if __name__ == "__main__":
    main()
    with open(paths["json"], 'w') as f:
        json.dump(datos_partido, f, indent=4)

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f"Ejecución {run_id} completada exitosamente. SAM 3 integrado y marcador corregido.")

if __name__ == "__main__":
    main()