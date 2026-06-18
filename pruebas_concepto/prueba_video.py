import torch
import os
from sam3.model_builder import build_sam3_video_predictor

print("=== INICIANDO DETECCIÓN Y TRACKING EN VIDEO ===")

# Configurar rutas
video_name = "IMG_9913.MOV"
video_path = f"videos_prueba/{video_name}"

# Verificar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

print("Cargando el predictor de video SAM 3...")
video_predictor = build_sam3_video_predictor()
# Asegurar que el predictor use la GPU
if hasattr(video_predictor, 'to'):
    video_predictor.to(device)

print(f"Iniciando sesión de video para: {video_path}")
# 1. Iniciar la sesión de tracking con el video
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = response["session_id"]
print(f"Sesión creada con ID: {session_id}")

print("Enviando prompt conceptual al frame 0...")
# 2. Añadir el prompt de texto en el primer cuadro (frame 0) para enganchar los robots
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0, 
        text="robot",
    )
)

# 3. Obtener las salidas del modelo (máscaras y trayectorias iniciales)
output = response["outputs"]

print("\n--- RESULTADOS INICIALES DEL VIDEO ---")
# Aquí SAM 3 nos devuelve un diccionario con las predicciones propagadas
if "predictions" in output:
    print(f"Se generaron predicciones de tracking para el video.")
    # Imprimimos las llaves para entender qué estructura de datos nos da Meta
    print("Campos disponibles en la salida:", output.keys())
else:
    print("El modelo procesó el frame inicial correctamente.")

print("=== PROCESO DE VIDEO TERMINADO ===")