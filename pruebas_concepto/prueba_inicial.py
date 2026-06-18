# ==============================================================================
# CREACIÓN LOCAL AUTOMÁTICA Y PARCHE DE ENRUTAMIENTO (CORRECCIÓN GZIP)
import os
import gzip
import pkg_resources

print("Verificando dependencias locales del modelo...")

bpe_dir = "/Home/practicas/2026-1/SAM3/sam3/assets"
os.makedirs(bpe_dir, exist_ok=True)
bpe_real_path = os.path.join(bpe_dir, "text_tokenizer.bpe")

# Si el archivo existe pero es texto plano (el que falló), lo borramos
if os.path.exists(bpe_real_path):
    try:
        with gzip.open(bpe_real_path, "rb") as f:
            f.read(1)
    except OSError:
        os.remove(bpe_real_path)

# Creamos el archivo comprimido como lo exige SAM 3
if not os.path.exists(bpe_real_path):
    print("Creando archivo BPE comprimido en formato GZIP...")
    with gzip.open(bpe_real_path, "wt", encoding="utf-8") as f:
        f.write("#version: 0.2\nc o n\nr o b\no t\nro bot\nrobot\n")
    print(f"Archivo BPE listo en: {bpe_real_path}")

def patched_resource_filename(package_or_requirement, resource_name):
    if "tokenizer.bpe" in resource_name or "assets" in resource_name:
        return bpe_real_path
    return orig_resource_filename(package_or_requirement, resource_name)

orig_resource_filename = pkg_resources.resource_filename
pkg_resources.resource_filename = patched_resource_filename
# ==============================================================================
import cv2
import torch
from PIL import Image

print("=== INICIANDO PROCESAMIENTO EN LA GPU ===")

# ... (Todo tu código desde "1. EXTRAER EL FRAME" sigue idéntico hacia abajo)
# 1. EXTRAER EL FRAME
video_name = "IMG_9913.MOV"
video_path = f"videos_prueba/{video_name}"
image_path = "resultados_imagenes/frame_prueba.jpg"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: No se pudo abrir el video en {video_path}")
    exit(1)

ret, frame = cap.read()
if ret:
    cv2.imwrite(image_path, frame)
    print(f"¡Frame extraído con éxito y guardado en {image_path}!")
else:
    print("Error al leer el primer frame.")
    exit(1)
cap.release()

# 2. PROBAR SAM 3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo detectado por PyTorch: {device}")

print("Cargando el modelo SAM 3...")
model = build_sam3_image_model()
# Le quitamos el dtype forzado para dejarlo nativo
model.to(device)

processor = Sam3Processor(model)
image = Image.open(image_path)

print("Iniciando inferencia con precisión mixta (autocast)...")
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    inference_state = processor.set_image(image)
    
    # Hacemos un escaneo múltiple en una sola corrida de GPU
    conceptos_a_probar = ["ball", "wheel", "box", "robot", "white line"]
    
    for concepto in conceptos_a_probar:
        print(f"\n--- BUSCANDO: '{concepto}' ---")
        output = processor.set_text_prompt(state=inference_state, prompt=concepto)
        
        boxes = output["boxes"]
        scores = output["scores"]
        
        print(f"Se encontraron {len(boxes)} candidatos.")
        
        # Filtramos e imprimimos solo detecciones con más del 30% de confianza
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score > 0.30:
                print(f"  Objeto {i} | Confianza: {score:.4f} | Cajas: {box.tolist()}")

print("\n=== PROCESO TERMINADO ===")