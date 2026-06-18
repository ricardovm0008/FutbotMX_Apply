# Motor de Analítica de Video para FutbotMX (YOLO + SAM 3)

Este repositorio aloja el código fuente de un sistema de visión computacional diseñado para automatizar el análisis táctico de partidos de futbolito robótico.

Mediante la integración de modelos de inteligencia artificial de estado del arte, el script procesa video en bruto para extraer telemetría precisa en tiempo real. Utiliza YOLOv8 para la detección y seguimiento continuo de los jugadores y el balón, y Segment Anything Model (SAM 3) para aislar con exactitud el área en píxeles de cada objeto.

El núcleo matemático del proyecto reside en su transformación de perspectiva, que toma las coordenadas de la cámara y las proyecta en un plano 2D. Esto permite alimentar un motor de eventos capaz de calcular la posesión, detectar colisiones, validar goles por cruce de zonas espaciales y generar un mapa de calor térmico acumulativo de las áreas de mayor tránsito.

## Tabla de Contenidos

- [Características Principales](#características-principales)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos e Instalación](#requisitos-e-instalación)
- [Configuración](#configuración)
- [Arquitectura y Flujo de Procesamiento](#arquitectura-y-flujo-de-procesamiento)
- [Salidas Generadas](#salidas-generadas)
- [Personalización y Calibración](#personalización-y-calibración)
- [Problemas Encontrados y Soluciones](#problemas-encontrados-y-soluciones)
- [Limitaciones y Mejoras Futuras](#limitaciones-y-mejoras-futuras)
- [Créditos y Autoría](#créditos-y-autoría)

## Características Principales

| Característica | Descripción |
|---|---|
| Detección Multiclase | YOLOv8 entrenado a medida para detectar pelota (clase 0), portería (clase 1) y robot (clase 2). |
| Segmentación de Precisión | SAM 3 genera máscaras semitransparentes exactas sobre el vídeo original para cada objeto detectado. |
| Tracking Avanzado | Seguimiento continuo de robots y pelota utilizando el tracker integrado de YOLO. |
| Transformación 2D | Homografía que mapea las coordenadas de la cámara a un plano cenital de la cancha (radar táctico). |
| Asignación de Equipos | Identificación automática basada en la posición inicial en la cancha, con sistema de herencia para resolver oclusiones. |
| Motor de Eventos | Detección en tiempo real de goles, colisiones, posesión del balón, pases e intercepciones. |
| Dashboard Tríptico | Interfaz visual 1080p que unifica el video original, el radar táctico y un panel de estadísticas con mapa de calor. |
| Exportación de Datos | Generación de video final MP4, registros estructurados en JSON y logs de ejecución en texto plano. |

## Estructura del Proyecto

```
SAM3/
├── Directorios
│   ├── assets/
│   ├── dataset_futbolito/
│   ├── models/
│   ├── pruebas_concepto/
│   ├── resultados_imagenes/
│   ├── resultados_videos/
│   ├── runs/
│   ├── sam3/
│   ├── src/
│   └── videos_prueba/
├── Scripts de Entrenamiento y Procesamiento (.py)
│   ├── entrenar.py
│   ├── extraer_dataset.py
│   └── extraer_frame.py
├── Scripts Principales y de Pruebas (.py)
│   ├── main2.py
│   ├── main_Dashboard.py
│   ├── main_exitoso1.py
│   ├── main_exitoso2.py
│   ├── main_exitoso3.py
│   ├── main_exitoso4.py
│   ├── main_PelotaTrack.py
│   └── main_SAM3YOLO.py
├── Scripts de Ejecución / Bash (.sh)
│   ├── run_sam3.sh
│   └── run_video.sh
└── Documentación
    └── README.md
```

## Requisitos e Instalación

Este proyecto requiere aceleración por hardware (GPU) para ejecutar los modelos de forma eficiente.

### Prerrequisitos del Sistema

- GPU NVIDIA con drivers actualizados (mínimo CUDA 11.8 recomendado).
- Anaconda o Miniconda instalado.
- Python 3.8 o superior (se recomienda 3.10 para compatibilidad con Ultralytics).
- Cuenta en Hugging Face con token de acceso (necesario para descargar SAM 3).

### Paso 1: Creación del entorno virtual

Abre una terminal y ejecuta:

```bash
conda create -n futbot_env python=3.10 -y
conda activate futbot_env
```

### Paso 2: Instalación de PyTorch

Elige el comando correspondiente a tu versión de CUDA.

Para CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Para CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verifica la instalación de CUDA ejecutando:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

El resultado debe ser `True`.

### Paso 3: Instalación de dependencias

```bash
pip install opencv-python numpy ultralytics supervision
```

### Paso 4: Autenticación en Hugging Face (Requisito para SAM 3)

El modelo SAM 3 requiere aceptar una licencia de uso.

1. Crea una cuenta en Hugging Face y solicita acceso al repositorio de `meta/segment-anything-3`.
2. Genera un token de acceso personal en la configuración de tu cuenta.
3. En la terminal, ejecuta el login y pega tu token:

```bash
huggingface-cli login
```

Descarga los pesos de SAM 3 (archivo `.pt`) y colócalos en la ruta definida en la variable `MODELO_SAM`. Asegúrate de tener el vídeo de entrada accesible en la ruta `VIDEO_PATH`.

## Configuración

Todas las constantes operativas se encuentran al inicio del script principal.

| Variable | Descripción |
|---|---|
| `VIDEO_PATH` | Ruta absoluta o relativa al vídeo de entrada. |
| `CARPETA_SALIDA` | Directorio destino para los resultados generados. |
| `MODELO_YOLO` | Ruta al modelo YOLO entrenado en formato `.pt`. |
| `MODELO_SAM` | Ruta a los pesos descargados del modelo SAM 3. |
| `SOURCE_PTS` | Coordenadas de los 4 vértices de la cancha en la imagen original. Requiere calibración por vídeo. |
| `RADAR_W`, `RADAR_H` | Dimensiones en píxeles del radar 2D (por defecto 500x800). |
| `TARGET_PTS` | Esquinas del radar correspondientes al mismo orden que `SOURCE_PTS`. |
| `MATRIZ_H` | Matriz de homografía calculada dinámicamente por OpenCV. |
| Zonas de Gol | Definidas por `X_GOL_MIN`, `X_GOL_MAX`, `ZONA_GOL_ARRIBA_Y` y `ZONA_GOL_ABAJO_Y`. |

## Arquitectura y Flujo de Procesamiento

### 1. Modelos de IA Utilizados

El sistema emplea YOLOv8 (personalizado) para la detección y el seguimiento de identidades mediante su funcionalidad `model.track`. Posteriormente, transfiere las cajas delimitadoras a SAM 3, el cual genera máscaras de segmentación binaria. Esto aporta precisión en la silueta y permite extraer el área segmentada para superposiciones en video.

### 2. Transformación de Perspectiva (Homografía)

Dado que la cámara graba en un ángulo oblicuo, se proyecta el campo en un plano cenital. Se utiliza la función `cv2.getPerspectiveTransform` para calcular la matriz de homografía `H` que mapea los puntos origen de la imagen `(x,y)` a las coordenadas del radar `(rx,ry)`. Cada centroide detectado pasa por `cv2.perspectiveTransform` para ser posicionado en el entorno 2D.

### 3. Bucle Principal de Procesamiento

Cada fotograma del video es procesado a través de seis fases secuenciales:

**Fase 1: Detección y Segmentación**
YOLO procesa el frame manteniendo las identidades. Si existen detecciones, las coordenadas se envían a SAM para generar las máscaras, las cuales se adjuntan a los objetos de detección.

**Fase 2: Extracción de Coordenadas en Radar**
Se calcula el centroide de cada detección, se aplica la homografía y se clasifica. Los robots se asignan al Equipo A o B dependiendo de su posición inicial en el eje Y del radar. Para resolver oclusiones, el sistema almacena robots temporalmente perdidos como "huérfanos" durante 2 segundos; si un nuevo seguimiento aparece cerca, hereda el equipo anterior.

**Fase 3: Motor de Eventos**

- **Colisiones:** Se calculan distancias entre robots. Una distancia menor a 35 píxeles activa un evento de colisión con un tiempo de enfriamiento de 2 segundos.
- **Posesión:** El robot a menos de 40 píxeles de la pelota es considerado el poseedor. El sistema compara el poseedor actual con el anterior para registrar "Pases completados" o "Intercepciones", sumando toques a las estadísticas.
- **Goles:** Si la pelota ingresa a las cajas espaciales definidas como porterías, se actualiza el marcador general, se suma el gol al robot poseedor y se activa una pausa de 4 segundos para evitar múltiples registros.

**Fase 4: Dibujado del Radar**
El lienzo cenital se actualiza dibujando estelas de movimiento históricas (cola de cometa), identificadores numéricos, la posición exacta de los jugadores y la trayectoria del balón.

**Fase 5: Superposición Visual**
Las máscaras generadas por SAM se aplican al video original utilizando colores específicos por equipo y clase, controlando la opacidad mediante librerías de anotación gráfica.

**Fase 6: Construcción del Dashboard**
Se renderiza un lienzo final que agrupa el video procesado, el radar interactivo y un panel de métricas. Las métricas incluyen barras horizontales de rendimiento individual (toques, goles, distancia) y un mapa de calor vertical generado mediante el desenfoque gaussiano de un acumulador de posiciones del balón.

## Salidas Generadas

Por cada ejecución exitosa, el sistema exporta tres archivos identificados con un ID incremental en la carpeta de salida:

- **Archivo MP4** (`dashboard_<id>.mp4`): El video final renderizado con el dashboard tríptico.
- **Archivo JSON** (`dashboard_<id>.json`): Estructura de datos completa detallando los fotogramas, marcador, posiciones espaciales de la pelota y estadísticas precisas de cada robot detectado.
- **Archivo LOG** (`dashboard_<id>.out`): Volcado de la consola incluyendo tiempos de inicio, registros del motor de eventos y posibles errores.

## Personalización y Calibración

Para adaptar el sistema a diferentes entornos de grabación, modifique las constantes en el script:

- **Ajuste de Homografía:** Actualice `SOURCE_PTS` con las coordenadas en píxeles de las cuatro esquinas de la cancha de su video específico.
- **Zonas de Gol:** Modifique los límites X e Y que definen los rectángulos de puntuación en el radar virtual.
- **Umbrales de Eventos:** Personalice la sensibilidad alterando la distancia de colisión (35), distancia de posesión (40), radio de herencia de huérfanos (50) y los tiempos de enfriamiento basados en los fotogramas por segundo (FPS).

## Problemas Encontrados y Soluciones

### Obtención del Modelo SAM 3

**Problema:** Descargar el modelo directamente devuelve un error de autorización, ya que Meta requiere aceptación explícita de licencia.

**Solución:** Se implementó el flujo de autenticación mediante cuenta de Hugging Face y token de acceso personal. Este paso es estricto y bloqueante para la instalación inicial.

### Costo Computacional de los Modelos

**Problema:** La ejecución simultánea de YOLO y SAM 3 es altamente demandante; el pipeline solo alcanzaba 2-3 FPS en hardware de gama media.

**Solución:** Se optimizó la carga limitando el uso de SAM únicamente a frames con detecciones válidas y agrupando todas las cajas delimitadoras en una sola llamada de inferencia. El sistema actual es ideal para análisis asíncrono (offline).

### Limitaciones del Dataset de Entrenamiento

**Problema:** Extraer y etiquetar datos manualmente a partir de videos largos con imágenes repetitivas resultaba ineficiente.

**Solución:** Se desarrolló un script automatizado para extraer fotogramas basado en diferenciales de movimiento. Esto generó un conjunto curado de imágenes que posteriormente se etiquetó en Label Studio para entrenar el modelo YOLO personalizado.

### Calibración de Grabaciones Imperfectas

**Problema:** Las variaciones en el ángulo de la cámara o canchas incompletas deformaban la proyección en el radar 2D.

**Solución:** El flujo actual requiere una calibración visual manual previa sobre un fotograma estático del video objetivo para definir los puntos exactos de anclaje de la homografía.

### Consistencia en Identidad (Oclusiones)

**Problema:** El tracker nativo perdía identificadores tras oclusiones prolongadas, reasignando equipos erróneamente por posición vertical.

**Solución:** Se implementó una lógica de memoria espacial. Los robots perdidos se declaran en estado de orfandad por 2 segundos. Las nuevas detecciones en ese radio heredan el identificador histórico y su equipo.

## Limitaciones y Mejoras Futuras

- El seguimiento de YOLO puede fallar en oclusiones muy largas que superen el temporizador de memoria espacial.
- El procesamiento con SAM 3 impide la ejecución fluida en tiempo real; se evalúa la implementación de saltos dinámicos de frames o la migración a versiones más ligeras (SAM 2).
- El mapa de calor visualiza la acumulación histórica total. Una mejora planificada es añadir un factor de decaimiento temporal.
- El motor de eventos no soporta reglas complejas como saques de banda o faltas; asume un entorno de juego ininterrumpido.
- La asignación de lados de la cancha es estática y actualmente no contempla el cambio de mitades en el medio tiempo.

## Créditos y Autoría

El proyecto fue construido utilizando herramientas y frameworks de código abierto fundamentales:

- **Ultralytics:** Detección de objetos y seguimiento.
- **Meta (Segment Anything):** Modelado de segmentación semántica de alta precisión.
- **Roboflow Supervision:** Gestión gráfica y utilidades de visión por computadora.
- **OpenCV:** Manipulación de video y transformaciones matriciales.

Link al Reel de instagram:
https://www.instagram.com/reel/DZvPu_lJYCTwfLOhJeGnPMmLBvcSwLVHySwntk0/?igsh=bTJrZmJ2ZWpoaTR4


**Desarrollado por:**
Ricardo Valera Martinez
Ingeniería en Ciencias de la Computación
Facultad de Ciencias de la Computación – BUAP
ricardo.valeram@alumno.buap.mx
