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
    [-95, 300],
    [904, 113],
    [1012, 1682],
    [66, 1838]
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
    cancha = np.zeros((h, w, 3), dtype=np.uint8)
    cancha[:] = (45, 135, 45)

    color_linea = (255, 255, 255)
    grosor = 3
    margen = 30

    cv2.rectangle(cancha, (margen, margen), (w - margen, h - margen), color_linea, grosor)
    cv2.line(cancha, (margen, h // 2), (w - margen, h // 2), color_linea, grosor)
    cv2.circle(cancha, (w // 2, h // 2), 60, color_linea, grosor)
    cv2.circle(cancha, (w // 2, h // 2), 6, color_linea, -1)

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

    model_yolo = YOLO(MODELO_YOLO)
    model_sam = SAM(MODELO_SAM)

    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)

    mask_annotator  = sv.MaskAnnotator(opacity=0.5)
    label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=1)

    dashboard_height = 1080
    escala           = dashboard_height / video_info.height
    nuevo_ancho      = int(video_info.width * escala)
    x_offset_radar   = nuevo_ancho + 100

    THIRD_PANEL_WIDTH = 300
    RIGHT_MARGIN = 40
    x_offset_third = x_offset_radar + RADAR_W + RIGHT_MARGIN
    dashboard_width = x_offset_third + THIRD_PANEL_WIDTH

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(paths["video"], fourcc, video_info.fps, (dashboard_width, dashboard_height))

    # ==========================================
    # VARIABLES DE HISTORIAL Y ESTADO
    # ==========================================
    trayectoria_pelota  = deque(maxlen=45)
    trayectorias_robots = {}
    historial_equipos   = {}
    marcador            = {"Equipo A": 0, "Equipo B": 0}

    cooldown_gol_top    = 0
    cooldown_gol_bottom = 0
    cooldown_colision   = 0
    FPS                 = int(video_info.fps)

    ZONA_GOL_ARRIBA_Y = 50
    ZONA_GOL_ABAJO_Y  = RADAR_H - 50
    X_GOL_MIN         = 230
    X_GOL_MAX         = 350

    orphan_robots = {}

    ultimo_robot_poseedor  = None
    ultimo_equipo_poseedor = None
    mensajes_narrador      = deque(maxlen=4)
    mensajes_narrador.append("¡Inicia el partido!")

    # Mapa de calor y estadísticas
    heatmap_accum = np.zeros((RADAR_H, RADAR_W), dtype=np.float32)

    robot_stats = {}

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
        # FASE 2 — Extracción de coordenadas
        # --------------------------------------------------
        datos_robots_frame = []
        datos_pelota_frame = None
        nuevos_ids = set()

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

            area_segmentada = int(np.sum(detecciones.mask[i])) if detecciones.mask is not None else 0

            # --- Robots ---
            if class_id == 2 and track_id != -1 and rx != -1:
                nuevos_ids.add(track_id)

                if track_id not in historial_equipos:
                    equipo_asignado = None
                    for oid, oinfo in orphan_robots.items():
                        dist = math.hypot(rx - oinfo['pos'][0], ry - oinfo['pos'][1])
                        if dist < 50:
                            equipo_asignado = oinfo['team']
                            del orphan_robots[oid]
                            break
                    if equipo_asignado is None:
                        equipo_asignado = "Equipo A" if ry < RADAR_H / 2 else "Equipo B"
                    historial_equipos[track_id] = equipo_asignado

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
        # ACTUALIZAR HUÉRFANOS
        # --------------------------------------------------
        for oid in list(orphan_robots.keys()):
            orphan_robots[oid]['age'] += 1
            if orphan_robots[oid]['age'] > FPS * 2:
                del orphan_robots[oid]

        for tid in historial_equipos:
            if tid not in nuevos_ids and tid not in orphan_robots:
                last_pos = None
                if tid in robot_stats and robot_stats[tid]['prev_pos'] is not None:
                    last_pos = robot_stats[tid]['prev_pos']
                elif tid in trayectorias_robots and trayectorias_robots[tid]:
                    last_pos = trayectorias_robots[tid][-1]
                if last_pos is not None:
                    orphan_robots[tid] = {
                        'pos': last_pos,
                        'team': historial_equipos[tid],
                        'age': 0
                    }

        # --------------------------------------------------
        # ACTUALIZAR ESTADÍSTICAS DE ROBOTS
        # --------------------------------------------------
        for robot in datos_robots_frame:
            tid = robot["id"]
            rx, ry = robot["rx"], robot["ry"]

            if tid not in robot_stats:
                robot_stats[tid] = {"touches": 0, "goals": 0, "distance": 0.0, "prev_pos": None}

            prev = robot_stats[tid]["prev_pos"]
            if prev is not None:
                dx = rx - prev[0]
                dy = ry - prev[1]
                robot_stats[tid]["distance"] += math.hypot(dx, dy)
            robot_stats[tid]["prev_pos"] = (rx, ry)

        # --------------------------------------------------
        # FASE 3 — MOTOR DE EVENTOS
        # --------------------------------------------------
        if cooldown_colision == 0 and len(datos_robots_frame) > 1:
            for i in range(len(datos_robots_frame)):
                for j in range(i + 1, len(datos_robots_frame)):
                    r1, r2 = datos_robots_frame[i], datos_robots_frame[j]
                    distancia = math.hypot(r1["rx"] - r2["rx"], r1["ry"] - r2["ry"])
                    if distancia < 35:
                        mensajes_narrador.append(f"¡Colision! {r1['equipo']} y {r2['equipo']}")
                        cooldown_colision = FPS * 2
                        break
        if cooldown_colision > 0:
            cooldown_colision -= 1

        if datos_pelota_frame is not None:
            poseedor_actual = None
            for robot in datos_robots_frame:
                dist_pelota = math.hypot(
                    robot["rx"] - datos_pelota_frame["rx"],
                    robot["ry"] - datos_pelota_frame["ry"]
                )
                if dist_pelota < 40:
                    poseedor_actual = robot
                    break

            if poseedor_actual is not None:
                if ultimo_robot_poseedor != poseedor_actual["id"]:
                    robot_stats[poseedor_actual["id"]]["touches"] += 1
                    if ultimo_robot_poseedor is not None:
                        if ultimo_equipo_poseedor == poseedor_actual["equipo"]:
                            mensajes_narrador.append(f"Pase completado en {poseedor_actual['equipo']}")
                        else:
                            mensajes_narrador.append(f"¡Intercepcion del {poseedor_actual['equipo']}!")
                ultimo_robot_poseedor = poseedor_actual["id"]
                ultimo_equipo_poseedor = poseedor_actual["equipo"]

        # Goles (Lógica basada en Zonas)
        if datos_pelota_frame is not None:
            px, py = datos_pelota_frame["rx"], datos_pelota_frame["ry"]
            
            # 1. Definir si la pelota está DENTRO de las cajas de gol
            en_zona_top = (py <= ZONA_GOL_ARRIBA_Y) and (X_GOL_MIN <= px <= X_GOL_MAX)
            en_zona_bottom = (py >= ZONA_GOL_ABAJO_Y) and (X_GOL_MIN <= px <= X_GOL_MAX)

            # 2. Evaluar Gol Equipo B (Arriba)
            if en_zona_top and cooldown_gol_top == 0:
                marcador["Equipo B"] += 1
                mensajes_narrador.append("¡GOOOOL DEL EQUIPO B!")
                if ultimo_robot_poseedor is not None and ultimo_robot_poseedor in robot_stats:
                    robot_stats[ultimo_robot_poseedor]["goals"] += 1
                cooldown_gol_top = FPS * 4
                ultimo_robot_poseedor = None

            # 3. Evaluar Gol Equipo A (Abajo)
            elif en_zona_bottom and cooldown_gol_bottom == 0:
                marcador["Equipo A"] += 1
                mensajes_narrador.append("¡GOOOOL DEL EQUIPO A!")
                if ultimo_robot_poseedor is not None and ultimo_robot_poseedor in robot_stats:
                    robot_stats[ultimo_robot_poseedor]["goals"] += 1
                cooldown_gol_bottom = FPS * 4
                ultimo_robot_poseedor = None

        # Descontar el tiempo de espera (cooldown) siempre, en cada frame
        if cooldown_gol_top > 0:
            cooldown_gol_top -= 1
        if cooldown_gol_bottom > 0:
            cooldown_gol_bottom -= 1

        # --------------------------------------------------
        # FASE 4 — DIBUJAR RADAR TÁCTICO
        # --------------------------------------------------
        for r_id, estela in trayectorias_robots.items():
            if len(estela) > 1 and r_id in historial_equipos:
                color_estela = (200, 100, 30) if historial_equipos[r_id] == "Equipo A" else (30, 30, 200)
                for k in range(1, len(estela)):
                    grosor = max(1, int(k / 8))
                    cv2.line(radar_canvas, estela[k - 1], estela[k], color_estela, grosor)

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
                "area_píxeles_sam": robot["area_sam"],
                "bbox":             robot["bbox"]
            })

        for k in range(1, len(trayectoria_pelota)):
            grosor = int(np.sqrt(45 / float(len(trayectoria_pelota) - k + 1)) * 2)
            cv2.line(radar_canvas, trayectoria_pelota[k - 1], trayectoria_pelota[k], (0, 140, 255), grosor)
        if trayectoria_pelota:
            cv2.circle(radar_canvas, trayectoria_pelota[-1], 9, (0, 100, 255), -1)
            cv2.circle(radar_canvas, trayectoria_pelota[-1], 9, (255, 255, 255), 2)

        if datos_pelota_frame is not None:
            frame_data["pelota"] = {
                "x_radar":          datos_pelota_frame["rx"],
                "y_radar":          datos_pelota_frame["ry"],
                "confianza":        datos_pelota_frame["conf"],
                "area_píxeles_sam": datos_pelota_frame["area_sam"]
            }
            # Sumar al mapa de calor permanente
            px, py = int(datos_pelota_frame["rx"]), int(datos_pelota_frame["ry"])
            if 0 <= px < RADAR_W and 0 <= py < RADAR_H:
                cv2.circle(heatmap_accum, (px, py), 12, 1.0, -1)

        # --------------------------------------------------
        # FASE 5 — MÁSCARAS SAM 3 SOBRE EL VÍDEO
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
        # FASE 6 — DASHBOARD TRÍPTICO
        # --------------------------------------------------
        dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)

        # Panel izquierdo: video anotado
        dashboard[0:dashboard_height, 0:nuevo_ancho] = cv2.resize(frame, (nuevo_ancho, dashboard_height))

        # Narrador
        cv2.rectangle(dashboard, (0, dashboard_height - 150), (nuevo_ancho, dashboard_height), (0,0,0), -1)
        cv2.putText(dashboard, "EVENTOS DEL PARTIDO:", (30, dashboard_height - 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        y_texto = dashboard_height - 70
        for msg in reversed(mensajes_narrador):
            color_texto = (100, 255, 100) if "GOOOOL" in msg else (255, 255, 255)
            cv2.putText(dashboard, f"> {msg}", (30, y_texto),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2)
            y_texto += 30

        # Panel central: Radar táctico
        y_radar = 150
        dashboard[y_radar:y_radar + RADAR_H, x_offset_radar:x_offset_radar + RADAR_W] = radar_canvas

        # Marcador
        alto_marcador = 80
        y_marcador    = 35
        cv2.rectangle(dashboard, (x_offset_radar, y_marcador),
                      (x_offset_radar + RADAR_W, y_marcador + alto_marcador), (20,20,20), -1)
        cv2.rectangle(dashboard, (x_offset_radar, y_marcador),
                      (x_offset_radar + RADAR_W, y_marcador + alto_marcador), (100,100,100), 2)
        cv2.putText(dashboard, "EQUIPO A", (x_offset_radar + 15, y_marcador + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,130,40), 2)
        cv2.putText(dashboard, "EQUIPO B", (x_offset_radar + RADAR_W - 125, y_marcador + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,240), 2)

        texto_puntos = f"{marcador['Equipo A']} - {marcador['Equipo B']}"
        text_size = cv2.getTextSize(texto_puntos, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)[0]
        tx = x_offset_radar + (RADAR_W - text_size[0]) // 2
        parpadeo = (cooldown_gol_top > 0 or cooldown_gol_bottom > 0) and (contador_frame // 10) % 2 == 0
        color_puntos = (0, 255, 0) if parpadeo else (255, 255, 255)
        cv2.putText(dashboard, texto_puntos, (tx, y_marcador + 53),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, color_puntos, 2)

        # --------------------------------------------------
        # PANEL DERECHO: Estadísticas + Mini campo vertical (mapa de calor acumulativo)
        # --------------------------------------------------
        third_panel = np.zeros((dashboard_height, THIRD_PANEL_WIDTH, 3), dtype=np.uint8)
        third_panel[:] = (30, 30, 30)

        # --- Estadísticas ---
        cv2.putText(third_panel, "ESTADISTICAS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        max_touches = max((s["touches"] for s in robot_stats.values()), default=1)
        max_goals   = max((s["goals"] for s in robot_stats.values()), default=1)
        max_dist    = max((s["distance"] for s in robot_stats.values()), default=1.0)

        y_base = 60
        bar_width = THIRD_PANEL_WIDTH - 80
        bar_height = 10
        gap = 25

        sorted_robots = sorted(robot_stats.items(), key=lambda x: x[1]["touches"], reverse=True)
        for i, (tid, stats) in enumerate(sorted_robots):
            y = y_base + i * gap
            if y > dashboard_height // 2 - 20:
                break

            cv2.putText(third_panel, f"R{tid}", (5, y+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

            touch_len = int((stats["touches"] / max_touches) * bar_width) if max_touches else 0
            cv2.rectangle(third_panel, (30, y), (30+touch_len, y+bar_height), (0,255,0), -1)
            cv2.putText(third_panel, f"T:{stats['touches']}", (30+touch_len+5, y+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

            goal_len = int((stats["goals"] / max_goals) * bar_width) if max_goals else 0
            cv2.rectangle(third_panel, (30, y+12), (30+goal_len, y+22), (0,0,255), -1)
            cv2.putText(third_panel, f"G:{stats['goals']}", (30+goal_len+5, y+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)

            dist_px = int((stats["distance"] / max_dist) * bar_width) if max_dist else 0
            cv2.rectangle(third_panel, (30, y+24), (30+dist_px, y+34), (255,0,0), -1)
            cv2.putText(third_panel, f"D:{int(stats['distance'])}", (30+dist_px+5, y+34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)

        # --- Mini campo VERTICAL con mapa de calor acumulativo ---
        mini_w = 250
        mini_h = int(mini_w * RADAR_H / RADAR_W)   # 250 * 800/500 = 400 pixeles de alto
        mini_x = (THIRD_PANEL_WIDTH - mini_w) // 2
        mini_y = dashboard_height - mini_h - 40

        # Seleccionamos la región del panel donde irá la cancha
        roi = third_panel[mini_y:mini_y+mini_h, mini_x:mini_x+mini_w]
        
        # 1. Pintamos el fondo de la cancha de color VERDE original
        roi[:] = (45, 135, 45)

        # 2. Dibujamos las líneas de la cancha ANTES del calor
        cv2.rectangle(roi, (0, 0), (mini_w, mini_h), (255, 255, 255), 1)
        cv2.line(roi, (0, mini_h//2), (mini_w, mini_h//2), (255, 255, 255), 1)
        
        porteria_w = 40
        porteria_h = 10
        cv2.rectangle(roi, (mini_w//2 - porteria_w//2, 0), (mini_w//2 + porteria_w//2, porteria_h), (0, 100, 255), -1)
        cv2.rectangle(roi, (mini_w//2 - porteria_w//2, mini_h - porteria_h), (mini_w//2 + porteria_w//2, mini_h), (0, 100, 255), -1)

        # 3. Procesar y superponer el mapa de calor de la pelota
        if np.max(heatmap_accum) > 0:
            # Difuminar para crear efecto de "nube" o mancha térmica
            heat_blur = cv2.GaussianBlur(heatmap_accum, (45, 45), 0)
            
            # Normalizar: el lugar con más acumulación será el rojo más fuerte (255)
            heat_norm = (heat_blur / heat_blur.max() * 255).astype(np.uint8)
            
            # Redimensionar al tamaño del mini campo (¡Ya no lo rotamos!)
            heat_resized = cv2.resize(heat_norm, (mini_w, mini_h), interpolation=cv2.INTER_LINEAR)
            
            # Aplicar mapa JET (escalas de azul, verde, amarillo y rojo)
            heat_color = cv2.applyColorMap(heat_resized, cv2.COLORMAP_JET)
            
            # Máscara y Fusión:
            # Solo queremos pintar los tonos cálidos (donde la intensidad > 10). 
            # Lo que sea 0 (donde no pasó la pelota) se queda con el césped verde intacto.
            alpha = 0.7 # 0.7 = 70% calor visible, 30% cancha verde debajo
            condicion_calor = heat_resized > 10
            
            for c in range(3): # Recorrer los canales B, G, R
                roi[:, :, c] = np.where(
                    condicion_calor, 
                    (heat_color[:, :, c] * alpha + roi[:, :, c] * (1 - alpha)).astype(np.uint8), 
                    roi[:, :, c]
                )

        # --> AQUÍ ESTÁ LA LÍNEA CRÍTICA PARA PEGAR EL PANEL DERECHO <--
        dashboard[:, x_offset_third:x_offset_third+THIRD_PANEL_WIDTH] = third_panel

        # --------------------------------------------------
        out_writer.write(dashboard)
        datos_partido["frames"].append(frame_data)
        contador_frame += 1

    # --------------------------------------------------
    # CIERRE
    # --------------------------------------------------
    out_writer.release()

    with open(paths["json"], 'w', encoding='utf-8') as f:
        json.dump(datos_partido, f, indent=4, ensure_ascii=False)

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f"Ejecución {run_id} completada. Dashboard tríptico con mapa de calor dinámico.")

if __name__ == "__main__":
    main()