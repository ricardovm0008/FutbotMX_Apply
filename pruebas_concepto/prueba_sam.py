import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Importaciones de la librería de Meta
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def show_mask(mask, ax):
    """Función auxiliar para pintar la máscara generada sobre la imagen"""
    color = np.array([30/255, 144/255, 255/255, 0.6]) 
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

print("Cargando modelo en memoria (Estado base para Autocast)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# CORRECCIÓN: Cargamos el modelo normal, SIN forzar el dtype.
model = build_sam3_image_model().to(device)
processor = Sam3Processor(model)

image_path = "resultados_imagenes/manzana.jpg" 
try:
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"Error: No se encontró '{image_path}'. Asegúrate de guardarla en la ruta correcta.")
    exit()

prompt_text = "apple"
print(f"Generando embeddings y buscando el concepto: '{prompt_text}'...")

# El bloque autocast se encarga de las conversiones matemáticas sobre la marcha
with torch.autocast(device_type=device, dtype=torch.bfloat16):
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)

masks = output["masks"]
scores = output["scores"]

# Lógica de Visualización y Guardado
if len(scores) > 0 and scores[0].item() > 0.5:
    confianza = scores[0].item() * 100
    print(f"¡Objeto detectado! Confianza estadística: {confianza:.2f}%")
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Aseguramos que la máscara vuelva a un formato legible para matplotlib
    best_mask = masks[0].cpu().to(torch.float32).numpy()
    
    show_mask(best_mask, plt.gca())
    
    plt.title(f"Detección de: '{prompt_text}' | Nivel de Confianza: {confianza:.1f}%", fontsize=14)
    plt.axis('off')
    
    ruta_salida = "resultados_imagenes/manzana_segmentada.jpg"
    plt.savefig(ruta_salida, bbox_inches='tight')
    print(f"Imagen guardada exitosamente en: {ruta_salida}")
    
    plt.close()
else:
    print("El modelo no encontró el objeto con suficiente confianza.")