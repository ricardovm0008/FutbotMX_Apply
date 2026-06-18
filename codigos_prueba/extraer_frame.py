import cv2

VIDEO_PATH = "videos_prueba/IMG_1.mp4"
OUTPUT_PATH = "resultados_imagenes/primer_frame.jpg"

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()

if ret:
    cv2.imwrite(OUTPUT_PATH, frame)
    print(f"¡Imagen guardada con éxito en {OUTPUT_PATH}!")
else:
    print("Error al leer el video.")
    
cap.release()