import cv2
import numpy as np
from ultralytics import YOLO, SAM
import supervision as sv
import os
import glob
import sys
import json
import math                  
from collections import deque

# ==========================================
# 1. CONFIGURACIÓN Y RUTAS BASE
# ==========================================
VIDEO_PATH = "videos_prueba/IMG_3.MOV"
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
    """Genera paths únicos con ID autoincremental. Robusto a nombres malformados."""
    os.makedirs(carpeta, exist_ok=True)
    archivos = glob.glob(os.path.join(carpeta, f"{nombre_base}_*.mp4"))
    max_id = 0
    for arch in archivos:
        nombre = os.path.basename(arch)
        try:
            num = int(nombre.replace(f"{nombre_base}_", "").replace(".mp4", ""))
            if num > max_id:
                max_id = num
        except ValueError:
            pass
    nuevo_id = max_id + 1
    paths = {
        "video": os.path.join(carpeta, f"{nombre_base}_{nuevo_id}.mp4"),
        "json":  os.path.join(carpeta, f"{nombre_base}_{nuevo_id}.json"),
        "out":   os.path.join(carpeta, f"{nombre_base}_{nuevo_id}.out")
    }
    return paths, nuevo_id

def crear_cancha_radar(w, h):
    """Dibuja el campo de fútbol 2D sobre el radar táctico."""
    cancha = np.zeros((h, w, 3), dtype=np.uint8)
    cancha[:] = (45, 135, 45)  # Verde pasto

    color_linea = (255, 255, 255)
    grosor = 3
    margen = 30

    # Borde exterior y línea media
    cv2.rectangle(cancha, (margen, margen), (w - margen, h - margen), color_linea, grosor)
    cv2.line(cancha, (margen, h // 2), (w - margen, h // 2), color_linea, grosor)

    # Círculo central
    cv2.circle(cancha, (w // 2, h // 2), 60, color_linea, grosor)
    cv2.circle(cancha, (w // 2, h // 2), 6, color_linea, -1)

    # Áreas
    ancho_area = 200
    alto_area = 100
    x_area = (w - ancho_area) // 2
    cv2.rectangle(cancha, (x_area, margen), (x_area + ancho_area, margen + alto_area), color_linea, grosor)
    cv2.rectangle(cancha, (x_area, h - margen - alto_area), (x_area + ancho_area, h - margen), color_linea, grosor)

    # Porterías (relleno azul + borde cyan)
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

    # Modelos en GPU
    model_yolo = YOLO(MODELO_YOLO)
    model_sam = SAM(MODELO_SAM)

    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)

    # Anotadores de Supervision (máscaras SAM 3)
    mask_annotator  = sv.MaskAnnotator(opacity=0.5)
    label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=1)

    # Dimensiones del dashboard
    dashboard_height = 1080
    escala           = dashboard_height / video_info.height
    nuevo_ancho      = int(video_info.width * escala)
    x_offset_radar   = nuevo_ancho + 100
    dashboard_width  = x_offset_radar + RADAR_W
    print(f"Tamaño del dashboard: {dashboard_width}x{dashboard_height}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(paths["video"], fourcc, video_info.fps, (dashboard_width, dashboard_height))

    # ==========================================
    # VARIABLES DE HISTORIAL Y ESTADO
    # ==========================================
    trayectoria_pelota  = deque(maxlen=45)
    trayectorias_robots = {}            # estela visual por robot {track_id: deque}
    historial_equipos   = {}            # Asignación persistente de equipo por robot
    marcador            = {"Equipo A": 0, "Equipo B": 0}
    cooldown_gol        = 0
    cooldown_colision   = 0             # evita spam de mensajes de colisión
    FPS                 = int(video_info.fps)

    ZONA_GOL_ARRIBA_Y = 30
    ZONA_GOL_ABAJO_Y  = RADAR_H - 30
    X_GOL_MIN         = 190
    X_GOL_MAX         = 310

    # NUEVO: Motor de Narración
    ultimo_robot_poseedor  = None
    ultimo_equipo_poseedor = None
    mensajes_narrador      = deque(maxlen=4)
    mensajes_narrador.append("¡Inicia el partido!")

    # Estructura de salida JSON
    datos_partido  = {"run_id": run_id, "fps": video_info.fps, "frames": []}
    contador_frame = 0

    # ==========================================
    # LOOP PRINCIPAL
    # ==========================================
    for frame in sv.get_video_frames_generator(VIDEO_PATH):

        # --------------------------------------------------
        # FASE 1 — Detección YOLO + Segmentación SAM 3
        # --------------------------------------------------
        resultados = model_yolo.track(frame, persist=True, verbose=False)[0]
        detecciones = sv.Detections.from_ultralytics(resultados)

        if len(detecciones) > 0 and detecciones.xyxy.size > 0:
            sam_results = model_sam(frame, bboxes=detecciones.xyxy, verbose=False)[0]
            if sam_results.masks is not None:
                detecciones.mask = sam_results.masks.data.cpu().numpy().astype(bool)

        radar_canvas = crear_cancha_radar(RADAR_W, RADAR_H)
        frame_data = {
            "frame":    contador_frame,
            "marcador": dict(marcador),
            "pelota":   None,
            "robots":   []
        }

        # --------------------------------------------------
        # FASE 2 — Extracción de coordenadas (pase previo)
        #           Necesario para que el motor de eventos
        #           tenga TODOS los datos antes de dibujar.
        # --------------------------------------------------
        datos_robots_frame = []   # Lista temporal por frame
        datos_pelota_frame = None

        for i in range(len(detecciones)):
            class_id  = detecciones.class_id[i]
            conf      = float(detecciones.confidence[i])
            track_id  = int(detecciones.tracker_id[i]) if detecciones.tracker_id is not None else -1

            x1, y1, x2, y2 = detecciones.xyxy[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            punto_trans = cv2.perspectiveTransform(
                np.array([[[cx, cy]]], dtype=np.float32), MATRIZ_H
            )
            rx = int(punto_trans[0][0][0]) if punto_trans is not None else -1
            ry = int(punto_trans[0][0][1]) if punto_trans is not None else -1

            # Área precisa usando máscara SAM 3
            area_segmentada = int(np.sum(detecciones.mask[i])) if detecciones.mask is not None else 0

            # --- Robots ---
            if class_id == 2 and track_id != -1 and rx != -1:
                if track_id not in historial_equipos:
                    historial_equipos[track_id] = "Equipo A" if ry < RADAR_H / 2 else "Equipo B"

                #  actualizar estela del robot
                if track_id not in trayectorias_robots:
                    trayectorias_robots[track_id] = deque(maxlen=25)
                trayectorias_robots[track_id].append((rx, ry))

                datos_robots_frame.append({
                    "id":       track_id,
                    "equipo":   historial_equipos[track_id],
                    "rx":       rx,
                    "ry":       ry,
                    "conf":     conf,
                    "area_sam": area_segmentada,
                    "bbox":     [float(x1), float(y1), float(x2), float(y2)]
                })

            # --- Pelota ---
            elif class_id == 0 and rx != -1:
                if 0 <= rx <= RADAR_W and 0 <= ry <= RADAR_H:
                    trayectoria_pelota.append((rx, ry))
                    datos_pelota_frame = {
                        "rx":       rx,
                        "ry":       ry,
                        "conf":     conf,
                        "area_sam": area_segmentada
                    }

        # --------------------------------------------------
        # FASE 3 — MOTOR DE EVENTOS MATEMÁTICO
        # --------------------------------------------------

        # A) Colisiones entre robots
        if cooldown_colision == 0 and len(datos_robots_frame) > 1:
            for i in range(len(datos_robots_frame)):
                for j in range(i + 1, len(datos_robots_frame)):
                    r1, r2 = datos_robots_frame[i], datos_robots_frame[j]
                    distancia = math.hypot(r1["rx"] - r2["rx"], r1["ry"] - r2["ry"])
                    if distancia < 35:  # umbral en píxeles del radar
                        mensajes_narrador.append(f"¡Colision! {r1['equipo']} y {r2['equipo']}")
                        cooldown_colision = FPS * 2
                        break
        if cooldown_colision > 0:
            cooldown_colision -= 1

        # B) Posesión, Pases e Intercepciones
        if datos_pelota_frame is not None:
            poseedor_actual = None
            for robot in datos_robots_frame:
                dist_pelota = math.hypot(
                    robot["rx"] - datos_pelota_frame["rx"],
                    robot["ry"] - datos_pelota_frame["ry"]
                )
                if dist_pelota < 40:  # umbral de control del balón
                    poseedor_actual = robot
                    break

            if poseedor_actual is not None:
                if (ultimo_robot_poseedor is not None and
                        ultimo_robot_poseedor != poseedor_actual["id"]):
                    if ultimo_equipo_poseedor == poseedor_actual["equipo"]:
                        mensajes_narrador.append(f"Pase completado en {poseedor_actual['equipo']}")
                    else:
                        mensajes_narrador.append(f"¡Intercepcion del {poseedor_actual['equipo']}!")
                ultimo_robot_poseedor  = poseedor_actual["id"]
                ultimo_equipo_poseedor = poseedor_actual["equipo"]

        # C) Goles
        if cooldown_gol == 0 and datos_pelota_frame is not None:
            px, py = datos_pelota_frame["rx"], datos_pelota_frame["ry"]
            if X_GOL_MIN <= px <= X_GOL_MAX:
                if py <= ZONA_GOL_ARRIBA_Y:
                    marcador["Equipo B"] += 1
                    mensajes_narrador.append("¡GOOOOL DEL EQUIPO B!")
                    cooldown_gol = FPS * 4
                    ultimo_robot_poseedor = None
                elif py >= ZONA_GOL_ABAJO_Y:
                    marcador["Equipo A"] += 1
                    mensajes_narrador.append("¡GOOOOL DEL EQUIPO A!")
                    cooldown_gol = FPS * 4
                    ultimo_robot_poseedor = None
        if cooldown_gol > 0:
            cooldown_gol -= 1

        # --------------------------------------------------
        # FASE 4 — Dibujar en el Radar Táctico
        # --------------------------------------------------

        # NUEVO: Estelas de robots (color de equipo, grosor creciente)
        for r_id, estela in trayectorias_robots.items():
            if len(estela) > 1 and r_id in historial_equipos:
                color_estela = (200, 100, 30) if historial_equipos[r_id] == "Equipo A" else (30, 30, 200)
                for k in range(1, len(estela)):
                    grosor = max(1, int(k / 8))
                    cv2.line(radar_canvas, estela[k - 1], estela[k], color_estela, grosor)

        # Posiciones actuales de robots + acumulación JSON
        for robot in datos_robots_frame:
            rx, ry = robot["rx"], robot["ry"]
            if 0 <= rx <= RADAR_W and 0 <= ry <= RADAR_H:
                color = (240, 130, 40) if robot["equipo"] == "Equipo A" else (50, 50, 240)
                cv2.circle(radar_canvas, (rx, ry), 16, color, -1)
                cv2.circle(radar_canvas, (rx, ry), 16, (255, 255, 255), 2)
                cv2.putText(radar_canvas, str(robot["id"]),
                            (rx - 6, ry + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

            frame_data["robots"].append({
                "track_id":         robot["id"],
                "equipo":           robot["equipo"],
                "x_radar":          rx if 0 <= rx <= RADAR_W else None,
                "y_radar":          ry if 0 <= ry <= RADAR_H else None,
                "confianza":        robot["conf"],
                "area_píxeles_sam": robot["area_sam"],   # Métrica de SAM 3
                "bbox":             robot["bbox"]
            })

        # Estela de la pelota (naranja brillante)
        for k in range(1, len(trayectoria_pelota)):
            grosor = int(np.sqrt(45 / float(len(trayectoria_pelota) - k + 1)) * 2)
            cv2.line(radar_canvas, trayectoria_pelota[k - 1], trayectoria_pelota[k], (0, 140, 255), grosor)
        if trayectoria_pelota:
            cv2.circle(radar_canvas, trayectoria_pelota[-1], 9, (0, 100, 255), -1)
            cv2.circle(radar_canvas, trayectoria_pelota[-1], 9, (255, 255, 255), 2)

        # Acumulación JSON de pelota
        if datos_pelota_frame is not None:
            frame_data["pelota"] = {
                "x_radar":          datos_pelota_frame["rx"],
                "y_radar":          datos_pelota_frame["ry"],
                "confianza":        datos_pelota_frame["conf"],
                "area_píxeles_sam": datos_pelota_frame["area_sam"]
            }

        # --------------------------------------------------
        # FASE 5 — Renderizado de Máscaras SAM 3 sobre el frame
        # --------------------------------------------------
        COLOR_EQUIPO_A = sv.Color(r=240, g=130, b=40)
        COLOR_EQUIPO_B = sv.Color(r=50,  g=50,  b=240)
        COLOR_PELOTA   = sv.Color(r=0,   g=165, b=255)
        COLOR_PORTERIA = sv.Color(r=255, g=255, b=0)

        for i in range(len(detecciones)):
            single_det = detecciones[i]
            c_id = single_det.class_id[0]
            t_id = int(single_det.tracker_id[0]) if single_det.tracker_id is not None else -1

            if c_id == 2:
                if t_id in historial_equipos:
                    eq = historial_equipos[t_id]
                    color_actual = COLOR_EQUIPO_A if eq == "Equipo A" else COLOR_EQUIPO_B
                    label_text   = f"{eq} #{t_id}"
                else:
                    color_actual = sv.Color.WHITE
                    label_text   = "Robot"
            elif c_id == 0:
                color_actual = COLOR_PELOTA
                label_text   = "Pelota"
            elif c_id == 1:
                color_actual = COLOR_PORTERIA
                label_text   = "Porteria"
            else:
                color_actual = sv.Color.WHITE
                label_text   = ""

            mask_annotator.color        = color_actual
            label_annotator.color       = color_actual
            label_annotator.text_color  = sv.Color.WHITE
            frame = mask_annotator.annotate(scene=frame, detections=single_det)
            frame = label_annotator.annotate(scene=frame, detections=single_det, labels=[label_text])

        # --------------------------------------------------
        # FASE 6 — Ensamblaje del Dashboard
        # --------------------------------------------------
        dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)

        # Panel izquierdo: video anotado
        dashboard[0:dashboard_height, 0:nuevo_ancho] = cv2.resize(frame, (nuevo_ancho, dashboard_height))

        # NUEVO: Caja de subtítulos del Narrador (parte inferior del panel de video)
        cv2.rectangle(dashboard,
                      (0, dashboard_height - 150),
                      (nuevo_ancho, dashboard_height),
                      (0, 0, 0), -1)
        cv2.putText(dashboard, "EVENTOS DEL PARTIDO:",
                    (30, dashboard_height - 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_texto = dashboard_height - 70
        for msg in reversed(mensajes_narrador):
            color_texto = (100, 255, 100) if "GOOOOL" in msg else (255, 255, 255)
            cv2.putText(dashboard, f"> {msg}",
                        (30, y_texto),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2)
            y_texto += 30

        # Panel derecho: Radar Táctico
        y_radar = 150
        dashboard[y_radar:y_radar + RADAR_H, x_offset_radar:x_offset_radar + RADAR_W] = radar_canvas

        # Marcador Estilizado (diseño Código 1: posición variable)
        alto_marcador = 80
        y_marcador    = 35
        cv2.rectangle(dashboard,
                      (x_offset_radar, y_marcador),
                      (x_offset_radar + RADAR_W, y_marcador + alto_marcador),
                      (20, 20, 20), -1)
        cv2.rectangle(dashboard,
                      (x_offset_radar, y_marcador),
                      (x_offset_radar + RADAR_W, y_marcador + alto_marcador),
                      (100, 100, 100), 2)
        cv2.putText(dashboard, "EQUIPO A",
                    (x_offset_radar + 15, y_marcador + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 130, 40), 2)
        cv2.putText(dashboard, "EQUIPO B",
                    (x_offset_radar + RADAR_W - 125, y_marcador + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 240), 2)

        texto_puntos = f"{marcador['Equipo A']} - {marcador['Equipo B']}"
        text_size    = cv2.getTextSize(texto_puntos, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)[0]
        tx           = x_offset_radar + (RADAR_W - text_size[0]) // 2
        color_puntos = (0, 255, 0) if (cooldown_gol > 0 and (cooldown_gol // 10) % 2 == 0) else (255, 255, 255)
        cv2.putText(dashboard, texto_puntos,
                    (tx, y_marcador + 53),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, color_puntos, 2)

        # Escribir frame y acumular JSON
        out_writer.write(dashboard)
        datos_partido["frames"].append(frame_data)
        contador_frame += 1

    # --------------------------------------------------
    # CIERRE Y GUARDADO DE RESULTADOS
    # --------------------------------------------------
    out_writer.release()

    with open(paths["json"], 'w', encoding='utf-8') as f:
        json.dump(datos_partido, f, indent=4, ensure_ascii=False)

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f"Ejecución {run_id} completada. Motor de eventos, trayectorias y narrador integrados.")


if __name__ == "__main__":
    main()