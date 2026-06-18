import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Importaciones de la librería de Meta
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def show_mask(mask, ax):
    """Función auxiliar para pintar la máscara generada sobre la imagen"""
    # Color azul semitransparente
    color = np.array([30/255, 144/255, 255/255, 0.6]) 
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

print("Cargando modelo en memoria (Estado base para Autocast)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = build_sam3_image_model().to(device)
processor = Sam3Processor(model)

# imagen a usar
image_path = "resultados_imagenes/frame_prueba.jpg" 
try:
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"Error: No se encontró '{image_path}'.")
    exit()

# Lista de conceptos a buscar
conceptos_a_buscar = [
    "ball",         # La pelota
    "green soccer field",  # La cancha
    "small robot"          # Los robots de los equipos
]

print("Generando embeddings de la imagen (Esto se hace una sola vez)...")
with torch.autocast(device_type=device, dtype=torch.bfloat16):
    inference_state = processor.set_image(image)
    
    #Iteramos sobre cada concepto en la lista
    for prompt_text in conceptos_a_buscar:
        print(f"\n--- Buscando el concepto: '{prompt_text}' ---")
        output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)

        masks = output["masks"]
        scores = output["scores"]

        if len(scores) > 0 and scores[0].item() > 0.5:
            confianza = scores[0].item() * 100
            print(f"¡Objeto detectado! Confianza: {confianza:.2f}%")
            
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            
            best_mask = masks[0].cpu().to(torch.float32).numpy()
            show_mask(best_mask, plt.gca())
            
            plt.title(f"Detección: '{prompt_text}' | Confianza: {confianza:.1f}%", fontsize=14)
            plt.axis('off')
            
            # Formatear el nombre del archivo de salida (cambia espacios por guiones bajos)
            nombre_limpio = prompt_text.replace(" ", "_")
            ruta_salida = f"resultados_imagenes/frame_{nombre_limpio}.jpg"
            
            plt.savefig(ruta_salida, bbox_inches='tight')
            print(f"Imagen guardada exitosamente en: {ruta_salida}")
            
            plt.close()
        else:
            print(f"No se encontró '{prompt_text}' con suficiente confianza en esta imagen.")

print("\n¡Análisis completado!")