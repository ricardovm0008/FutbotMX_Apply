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
MODELO_YOLO = "runs/detect/modelo_propio/futbolito_v1-3/weights/best.pt"

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
    """Dibuja la cancha táctica con líneas y porterías"""
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

    # Inicialización base de los Anotadores Visuales
    corner_annotator = sv.BoxCornerAnnotator(thickness=4)
    label_annotator = sv.LabelAnnotator(text_scale=0.8, text_thickness=2)

    # Configuración dinámica del lienzo
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
    datos_partido = {
        "run_id": run_id,
        "fps": video_info.fps,
        "frames": []
    }
    contador_frame = 0

    for frame in sv.get_video_frames_generator(VIDEO_PATH):
        # Mapeo persistente de trayectorias
        resultados = model.track(frame, persist=True, verbose=False)[0]
        detecciones = sv.Detections.from_ultralytics(resultados)

        radar_canvas = crear_cancha_radar(RADAR_W, RADAR_H)
        frame_data = {
            "frame": contador_frame,
            "pelota": None,
            "robots": []
        }

        # ----------------==========================
        # A. PROCESAMIENTO TÁCTICO (RADAR Y DATA)
        # ----------------==========================
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

            # --- Clase 2: Robots ---
            if class_id == 2:
                if track_id != -1:
                    if track_id not in historial_equipos:
                        if ry < RADAR_H / 2:
                            historial_equipos[track_id] = "Equipo A"
                        else:
                            historial_equipos[track_id] = "Equipo B"
                    equipo = historial_equipos[track_id]

                    if 0 <= rx <= RADAR_W and 0 <= ry <= RADAR_H:
                        color = (240, 130, 40) if equipo == "Equipo A" else (50, 50, 240)  # BGR
                        cv2.circle(radar_canvas, (rx, ry), 16, color, -1)
                        cv2.circle(radar_canvas, (rx, ry), 16, (255, 255, 255), 2)
                        cv2.putText(radar_canvas, str(track_id), (rx - 6, ry + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

                    frame_data["robots"].append({
                        "track_id": track_id,
                        "equipo": equipo,
                        "x_radar": rx if (0 <= rx <= RADAR_W) else None,
                        "y_radar": ry if (0 <= ry <= RADAR_H) else None,
                        "confianza": conf,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })

            # --- Clase 0: Pelota ---
            elif class_id == 0:
                if 0 <= rx <= RADAR_W and 0 <= ry <= RADAR_H:
                    trayectoria_pelota.append((rx, ry))
                    frame_data["pelota"] = {
                        "x_radar": rx,
                        "y_radar": ry,
                        "confianza": conf
                    }

        # Pintar estela histórica de la trayectoria
        for i in range(1, len(trayectoria_pelota)):
            grosor = int(np.sqrt(45 / float(len(trayectoria_pelota) - i + 1)) * 2)
            cv2.line(radar_canvas, trayectoria_pelota[i - 1], trayectoria_pelota[i], (0, 140, 255), grosor)
        if trayectoria_pelota:
            cv2.circle(radar_canvas, trayectoria_pelota[-1], 9, (0, 100, 255), -1)
            cv2.circle(radar_canvas, trayectoria_pelota[-1], 9, (255, 255, 255), 2)

        # ----------------==========================
        # B. RENDERIZADO DINÁMICO DE ANOTADORES (FIX)
        # ----------------==========================
        COLOR_EQUIPO_A = sv.Color(r=240, g=130, b=40)   # Azul cian en RGB/Supervision
        COLOR_EQUIPO_B = sv.Color(r=50, g=50, b=240)    # Rojo en RGB/Supervision
        COLOR_PELOTA   = sv.Color(r=0, g=165, b=255)    # Naranja
        COLOR_PORTERIA = sv.Color(r=57, g=255, b=20)  # Verde

        for i in range(len(detecciones)):
            single_det = detecciones[i]  # Aislamos la detección exacta
            c_id = single_det.class_id[0]
            t_id = int(single_det.tracker_id[0]) if single_det.tracker_id is not None else -1

            # Configurar color y etiqueta según correspondencia
            if c_id == 2:  # Robot
                if t_id in historial_equipos:
                    eq = historial_equipos[t_id]
                    color_actual = COLOR_EQUIPO_A if eq == "Equipo A" else COLOR_EQUIPO_B
                    label_text = f"{eq} #{t_id}"
                else:
                    color_actual = sv.Color.WHITE
                    label_text = "Robot"
            elif c_id == 0:  # Pelota
                color_actual = COLOR_PELOTA
                label_text = "Pelota"
            elif c_id == 1:  # Portería
                color_actual = COLOR_PORTERIA
                label_text = "Porteria"
            else:
                color_actual = sv.Color.WHITE
                label_text = ""

            # Modificar la propiedad de los objetos anotadores dinámicamente
            corner_annotator.color = color_actual
            label_annotator.color = color_actual
            label_annotator.text_color = sv.Color.WHITE

            # Quemar la anotación individual secuencialmente sobre el mismo frame
            frame = corner_annotator.annotate(scene=frame, detections=single_det)
            frame = label_annotator.annotate(scene=frame, detections=single_det, labels=[label_text])

        # ----------------==========================
        # C. ENSAMBLAJE FINAL DEL DASHBOARD
        # ----------------==========================
        dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
        
        # Vídeo escalado pegado al extremo izquierdo
        frame_escalado = cv2.resize(frame, (nuevo_ancho, dashboard_height))
        dashboard[0:dashboard_height, 0:nuevo_ancho] = frame_escalado
        
        # Inyección del radar táctico centrado verticalmente
        y_radar = 100
        dashboard[y_radar:y_radar + RADAR_H, x_offset_radar:x_offset_radar + RADAR_W] = radar_canvas

        out.write(dashboard)
        datos_partido["frames"].append(frame_data)
        contador_frame += 1

    out.release()

    # Guardar JSON final estructurado
    with open(paths["json"], 'w') as f:
        json.dump(datos_partido, f, indent=4)

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f"Ejecución {run_id} completada sin errores. Archivos guardados en '{CARPETA_SALIDA}'.")

if __name__ == "__main__":
    main()