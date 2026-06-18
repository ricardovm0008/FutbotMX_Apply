import os
from ultralytics import YOLO

def main():
    dataset_path = os.path.abspath("dataset_futbolito")
    classes_file = os.path.join(dataset_path, "classes.txt")
    
    if not os.path.exists(classes_file):
        print("Error: No se encontró el archivo classes.txt. Verifica que descomprimiste bien el zip.")
        return

    # 1. Leer tus etiquetas automáticamente
    with open(classes_file, 'r') as f:
        clases = [line.strip() for line in f.readlines()]

    # 2. Crear el archivo de configuración data.yaml
    yaml_content = f"""
path: {dataset_path}
train: images
val: images
nc: {len(clases)}
names: {clases}
"""
    yaml_path = os.path.join(dataset_path, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Configuración lista. Clases detectadas: {clases}")
    print("Iniciando entrenamiento en la GPU...")

    # 3. Lanzar el entrenamiento
    # Usamos yolov8x.pt porque tienes hardware de sobra para el modelo más potente
    model = YOLO("yolov8x.pt") 

    resultados = model.train(
        data=yaml_path,
        epochs=50,             # 50 pasadas sobre 225 imágenes
        imgsz=640,             # Resolución de entrenamiento
        batch=16,              # Paquetes de 16 imágenes a la vez
        device=0,              # Le indicamos que use la GPU principal
        project="modelo_propio",
        name="futbolito_v1"
    )
    
    print("¡Entrenamiento finalizado!")

if __name__ == "__main__":
    main()