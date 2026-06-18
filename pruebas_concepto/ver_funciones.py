# ==============================================================================
# PARCHE DE RUTAS Y DEPENDENCIAS (FORMATO SEGURO GZIP)
import os, gzip, pkg_resources

bpe_dir = "/Home/practicas/2026-1/SAM3/sam3/assets"
os.makedirs(bpe_dir, exist_ok=True)
bpe_path = os.path.join(bpe_dir, "text_tokenizer.bpe")

if not os.path.exists(bpe_path):
    with gzip.open(bpe_path, "wt", encoding="utf-8") as f:
        f.write("#version: 0.2\n")

def patched_resource_filename(req, res): return bpe_path
pkg_resources.resource_filename = patched_resource_filename
# ==============================================================================

import torch
from PIL import Image
import cv2
import numpy as np

# Crear carpeta de salida
os.makedirs("resultados_imagenes", exist_ok=True)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ---------- CONFIGURACIÓN ----------
image_path = "resultados_imagenes/cat.jpg"  # Tu imagen
prompt_texto = "cat"  # Cambia por "cat", "dog", "person", etc.

print(f"Cargando modelo SAM 3...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_sam3_image_model()
model.to(device)
processor = Sam3Processor(model)

print(f"Imagen: {image_path}")
print(f"Prompt: '{prompt_texto}'")

# Cargar imagen
image = Image.open(image_path).convert("RGB")
img_anotada = cv2.imread(image_path)

# Inferencia
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt_texto)

# Extraer resultados
masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]

# Convertir a numpy (manejo de bfloat16)
if isinstance(masks, torch.Tensor):
    masks = masks.float().cpu().numpy()
if isinstance(boxes, torch.Tensor):
    boxes = boxes.float().cpu().numpy()
if isinstance(scores, torch.Tensor):
    scores = scores.float().cpu().numpy()

num_objetos = len(masks) if isinstance(masks, list) else masks.shape[0]
print(f"\n✅ Objetos detectados: {num_objetos}")

# Dibujar resultados
for i in range(num_objetos):
    mask = np.squeeze(masks[i])
    box = boxes[i]
    score = scores[i]
    
    if score < 0.3:
        continue
    
    x1, y1, x2, y2 = map(int, box)
    print(f"  [{i+1}] Score: {score:.3f} | Caja: [{x1}, {y1}, {x2}, {y2}]")
    
    # Dibujar rectángulo y etiqueta
    cv2.rectangle(img_anotada, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(img_anotada, f"{prompt_texto} ({score:.2f})", 
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Guardar resultado
ruta_salida = f"resultados_imagenes/deteccion_{prompt_texto.replace(' ', '_')}.jpg"
cv2.imwrite(ruta_salida, img_anotada)
print(f"\n📁 Imagen guardada en: {ruta_salida}")