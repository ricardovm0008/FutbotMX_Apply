import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# Importaciones de la librería de Meta
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# MODIFICACIÓN 1: La función ahora recibe el color como parámetro
def show_mask(mask, ax, color):
    """Pinta la máscara generada sobre la imagen usando el color especificado"""
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

print("Cargando modelo en memoria (Estado base para Autocast)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = build_sam3_image_model().to(device)
processor = Sam3Processor(model)

image_path = "resultados_imagenes/frame_pruebaUP.png" 
try:
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"Error: No se encontró '{image_path}'.")
    exit()

# MODIFICACIÓN 2: Diccionario que asocia el prompt con un color RGBA específico
# El formato es [Rojo, Verde, Azul, Transparencia]. Rango de 0 a 1.
conceptos_a_buscar = {
    "ball":               np.array([255/255, 0/255, 0/255, 0.6]),    # Rojo
    "green soccer field": np.array([128/255, 0/255, 128/255, 0.6]),    # Verde (más transparente para que no tape todo)
    "small robot":        np.array([0/255, 0/255, 255/255, 0.6]),    # Azul
    "white lines":        np.array([255/255, 255/255, 0/255, 0.6])
}

print("Generando embeddings de la imagen...")

# MODIFICACIÓN 3: Preparamos el lienzo ÚNICO antes del ciclo
plt.figure(figsize=(12, 12))
plt.imshow(image)
ax = plt.gca()

# Lista para guardar los cuadros de color que irán en la leyenda
elementos_leyenda = []

with torch.autocast(device_type=device, dtype=torch.bfloat16):
    inference_state = processor.set_image(image)
    
    # Iteramos sobre el diccionario (prompt y su color)
    for prompt_text, color in conceptos_a_buscar.items():
        print(f"Buscando: '{prompt_text}'...")
        output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)

        masks = output["masks"]
        scores = output["scores"]

        if len(scores) > 0 and scores[0].item() > 0.5:
            confianza = scores[0].item() * 100
            print(f"  -> ¡Detectado! Confianza: {confianza:.2f}%")
            
            # Extraemos y pintamos la máscara en el MISMO lienzo, usando su color
            best_mask = masks[0].cpu().to(torch.float32).numpy()
            show_mask(best_mask, ax, color)
            
            # Creamos el parche de color para la leyenda con su porcentaje de confianza
            parche = mpatches.Patch(color=color, label=f"{prompt_text} ({confianza:.1f}%)")
            elementos_leyenda.append(parche)
        else:
            print(f"  -> No se encontró con suficiente confianza.")

# MODIFICACIÓN 4: Añadimos la leyenda, ajustes visuales y guardamos la imagen final
print("\nGuardando imagen combinada...")

# Colocamos la leyenda en la esquina superior derecha, fuera de la imagen si es posible
plt.legend(handles=elementos_leyenda, loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=12)
plt.title("Análisis de Segmentación - Robotics Soccer", fontsize=16)
plt.axis('off')

ruta_salida = "resultados_imagenes/analisis_completo_partidoUP.jpg"
plt.savefig(ruta_salida, bbox_inches='tight')
plt.close()

print(f"¡Listo! La imagen combinada está en: {ruta_salida}")