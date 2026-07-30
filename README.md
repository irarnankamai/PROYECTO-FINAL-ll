# Detector de Taxis Amarillos con HOG + SVM y YOLO

Sistema de visión por computador desarrollado para detectar taxis amarillos del entorno ecuatoriano en tiempo real.

El proyecto utiliza **OpenCV C++**, el descriptor **Histogram of Oriented Gradients (HOG)** y un clasificador **Support Vector Machine (SVM)** para realizar la detección inicial. Cuando se identifica un taxi amarillo, el sistema captura una imagen, graba un video de cinco segundos y envía la evidencia a una API desarrollada con **FastAPI**.

Posteriormente, la evidencia es procesada mediante **YOLO11n-seg** y los resultados son enviados automáticamente al usuario a través de un bot de **Telegram**.

---

## Autores

- Noé Leandro Ayavaca Guarango
- Irar Cristian Nankamai Tsuink

**Tutor:** Ing. Vladimir Robles Bykbaev  
**Institución:** Universidad Politécnica Salesiana  
**Sede:** Cuenca  
**Carrera:** Computación  
**Año:** 2026  

---

## Objetivo general

Desarrollar un sistema de visión por computador capaz de detectar taxis amarillos en tiempo real mediante OpenCV C++, utilizando el descriptor HOG y el clasificador SVM, con generación automática de evidencia y envío de notificaciones a través de un servidor FastAPI y un bot de Telegram.

---

## Funcionalidades principales

- Captura de video en tiempo real.
- Detección de taxis amarillos mediante HOG + SVM.
- Visualización del bounding box.
- Visualización del score de detección.
- Captura automática de imágenes.
- Grabación automática de videos de cinco segundos.
- Control de eventos para evitar envíos repetidos.
- Envío de evidencias mediante FastAPI.
- Procesamiento de imágenes y videos con YOLO11n-seg.
- Envío de alertas y evidencias mediante Telegram.
- Visualización de FPS, memoria RAM y estado del sistema.
- Compatibilidad con cámara integrada y cámara USB.

---

## Tecnologías utilizadas

### Aplicación principal

- C++
- OpenCV
- CMake
- HOG
- SVM
- libcurl

### Servidor y procesamiento

- Python
- FastAPI
- Uvicorn
- Ultralytics YOLO
- OpenCV Python
- Telegram Bot API

### Preparación del conjunto de datos

- Python
- Albumentations
- NumPy
- OpenCV

---

## Arquitectura del sistema

El funcionamiento general del sistema es el siguiente:

1. La aplicación captura video en tiempo real mediante OpenCV C++.
2. Los fotogramas son analizados mediante el detector HOG + SVM.
3. Si no se detecta un taxi, el sistema continúa monitoreando.
4. Si se detecta un taxi amarillo:
   - Se dibuja el bounding box.
   - Se muestra el score.
   - Se captura una imagen.
   - Se graba un video de cinco segundos.
5. La evidencia se envía al servidor FastAPI.
6. FastAPI procesa la imagen y el video mediante YOLO11n-seg.
7. El bot de Telegram envía la alerta y la evidencia.
8. El sistema continúa con el monitoreo en tiempo real.

---

## Estructura del proyecto

```text
ProyectoTaxiDetector/
├── build/
├── captures/
├── dataset/
│   ├── positivos/
│   │   ├── originales/
│   │   ├── train/
│   │   ├── validation/
│   │   ├── test/
│   │   ├── augmented/
│   │   └── recortes/
│   │       ├── train/
│   │       ├── validation/
│   │       ├── test/
│   │       └── augmented/
│   ├── negativos/
│   │   ├── originales/
│   │   ├── train/
│   │   ├── validation/
│   │   ├── test/
│   │   └── augmented/
│   └── hog/
│       ├── train/
│       ├── validation/
│       └── test/
├── include/
│   ├── detector_hog.hpp
│   ├── capturador.hpp
│   ├── cliente_api.hpp
│   └── grabador_video.cpp
├── models/
│   └── hog_svm_taxi.yml
├── resultados/
├── scripts/
│   ├── verificar_dataset.py
│   ├── dividir_dataset_generico.py
│   ├── aumentar_negativos.py
│   ├── aumentar_recortes_positivos.py
│   └── preparar_dataset_hog.py
├── src/
│   ├── detector_hog.cpp
│   ├── capturador.cpp
│   ├── cliente_api.cpp
│   ├── grabador_video.cpp
│   ├── detector_hog.cpp
│   ├── entrenar_hog_svm.cpp.cpp
│   ├── etiquetar_taxis.cpp
│   ├── detectar_taxi_imagen.cpp
│   └── detectar_taxi_camara.cpp
├── TelegramBot/
│   ├── input/
│   ├── output/
│   ├── api_server.py
│   ├── bot.py
│   ├── config.py
│   ├── video_handler.py
│   ├── video_processor.py
│   ├── yolo_processor.py
│   ├── yolo11n-seg.pt
│   └── requirements.txt
├── videos/
├── .env.local
├── activar_proyecto.sh
├── CMakeLists.txt
└── README.md
```

---

## Conjunto de datos

Para entrenar el detector se construyó un conjunto de datos balanceado compuesto por:

- 4.000 imágenes positivas.
- 4.000 imágenes negativas.
- 8.000 imágenes en total.

Las imágenes positivas contienen taxis amarillos, mientras que las imágenes negativas contienen otros vehículos, objetos y escenarios sin el vehículo objetivo.

### Imágenes positivas originales

Inicialmente se recolectaron 777 imágenes positivas válidas.

| Conjunto | Cantidad |
|---|---:|
| Entrenamiento | 543 |
| Validación | 117 |
| Prueba | 117 |
| Total original | 777 |

Las técnicas de Data Augmentation se aplicaron sobre el conjunto de entrenamiento hasta completar un total de 4.000 imágenes positivas.

### Imágenes negativas originales

Se utilizaron inicialmente 900 imágenes negativas.

| Conjunto | Cantidad |
|---|---:|
| Entrenamiento | 630 |
| Validación | 135 |
| Prueba | 135 |
| Total original | 900 |

Las técnicas de Data Augmentation se aplicaron sobre el conjunto de entrenamiento hasta completar un total de 4.000 imágenes negativas.

---

## Configuración del descriptor HOG

| Parámetro | Valor |
|---|---|
| Tamaño de ventana | 128 × 64 |
| Tamaño de bloque | 16 × 16 |
| Desplazamiento del bloque | 8 × 8 |
| Tamaño de celda | 8 × 8 |
| Número de bins | 9 |
| Longitud del descriptor | 3780 |

---

## Entrenamiento del modelo

El modelo HOG + SVM fue entrenado utilizando:

- 3.766 imágenes positivas de entrenamiento.
- 3.730 imágenes negativas de entrenamiento.
- 7.496 imágenes de entrenamiento en total.

El archivo generado durante el entrenamiento se encuentra en:

```text
models/hog_svm_taxi.yml
```

---

## Resultados del modelo HOG + SVM

### Métricas obtenidas

| Métrica | Resultado |
|---|---:|
| Accuracy | 99,21 % |
| Precisión | 99,15 % |
| Sensibilidad | 99,15 % |
| Especificidad | 99,26 % |
| F1-score | 99,15 % |

### Matriz de confusión

| Resultado | Cantidad |
|---|---:|
| Verdaderos positivos | 116 |
| Verdaderos negativos | 134 |
| Falsos positivos | 1 |
| Falsos negativos | 1 |

---

## Requisitos del sistema

### Sistema operativo

El proyecto fue desarrollado y probado principalmente en:

- Kubuntu/Linux.
- Visual Studio Code.

### Dependencias de C++

- Compilador compatible con C++17.
- OpenCV.
- CMake.
- libcurl.

### Dependencias de Python

- Python 3.
- FastAPI.
- Uvicorn.
- Ultralytics.
- OpenCV.
- NumPy.
- Requests.
- HTTPX.
- Python Multipart.
- Albumentations.

---

## Instalación en Kubuntu o Ubuntu

### 1. Instalar las dependencias del sistema

```bash
sudo apt update
sudo apt install build-essential cmake pkg-config libopencv-dev libcurl4-openssl-dev python3 python3-pip python3-venv
```

### 2. Ingresar en la carpeta del proyecto

```bash
cd /home/irar/Escritorio/ProyectoTaxiDetector
```

### 3. Crear el entorno virtual

```bash
python3 -m venv .venv
```

### 4. Activar el entorno virtual

```bash
source .venv/bin/activate
```

### 5. Actualizar pip

```bash
pip install --upgrade pip
```

### 6. Instalar las dependencias de Python

```bash
pip install -r TelegramBot/requirements.txt
```

---

## Configuración del bot de Telegram

En la carpeta principal del proyecto debe existir un archivo llamado:

```text
.env.local
```

Ejemplo:

```env
TELEGRAM_BOT_TOKEN=COLOCAR_TOKEN_DEL_BOT
TELEGRAM_CHAT_ID=COLOCAR_CHAT_ID
```

No se debe publicar el token real del bot ni el identificador privado del chat.

---

## Compilación del proyecto C++

Desde la carpeta principal del proyecto, ejecutar:

```bash
cmake -S . -B build
```

Luego compilar:

```bash
cmake --build build
```

También se puede utilizar compilación paralela:

```bash
cmake --build build --parallel
```

Los ejecutables se generarán en la carpeta:

```text
build/
```

---

## Ejecución del servidor FastAPI

Primero se debe activar el entorno virtual:

```bash
source .venv/bin/activate
```

Luego ejecutar:

```bash
python3 TelegramBot/api_server.py
```

La dirección local del servidor es:

```text
http://127.0.0.1:8000
```

Para comprobar que la API se encuentra activa:

```bash
curl http://127.0.0.1:8000/health
```

---

## Detección en una imagen

Sintaxis:

```bash
./build/detectar_taxi_imagen_v2 <ruta_imagen> [score_minimo]
```

Ejemplo:

```bash
./build/detectar_taxi_imagen_v2 dataset/positivos/test/taxi_amarillo_test_0001.jpg 0.8
```

El parámetro `score_minimo` es opcional.

Un valor bajo aumenta la sensibilidad, pero puede generar más falsos positivos. Un valor alto reduce los falsos positivos, pero también puede impedir la detección de algunos taxis.

---

## Detección mediante cámara

Sintaxis:

```bash
./build/detectar_taxi_camara <indice_camara> <intervalo_frames>
```

Ejemplo con cámara integrada:

```bash
./build/detectar_taxi_camara 0 5
```

Ejemplo con cámara USB:

```bash
./build/detectar_taxi_camara 2 5
```

Ejemplo con una detección cada 10 fotogramas:

```bash
./build/detectar_taxi_camara 0 10
```

### Parámetros

- `indice_camara`: índice asignado a la cámara por el sistema operativo.
- `intervalo_frames`: cantidad de fotogramas entre cada ejecución del detector.

---

## Identificar cámaras disponibles en Linux

Para listar las cámaras disponibles:

```bash
v4l2-ctl --list-devices
```

También se puede ejecutar:

```bash
ls /dev/video*
```

Durante las pruebas se utilizaron los siguientes dispositivos:

- Cámara integrada: `/dev/video0`.
- Cámara USB: `/dev/video2`.

Los índices pueden cambiar dependiendo del computador y del orden en que se conecten las cámaras.

---

## Controles de la aplicación

| Tecla | Acción |
|---|---|
| `Q` | Cerrar la aplicación |
| `ESC` | Cerrar la aplicación |

---

## Evidencia generada

Cuando se detecta un taxi amarillo, el sistema genera automáticamente una imagen y un video de cinco segundos.

### Imágenes

Las imágenes se almacenan en:

```text
captures/
```

### Videos

Los videos se almacenan en:

```text
videos/
```

### Resultados procesados

Los archivos procesados mediante YOLO se almacenan en:

```text
TelegramBot/output/
```

---

## Información enviada a Telegram

El bot puede enviar la siguiente información:

- Alerta de detección.
- Identificador del evento.
- Vehículo objetivo.
- Cámara utilizada.
- Fecha y hora.
- Imagen original.
- Imagen procesada mediante YOLO.
- Video original.
- Video procesado mediante YOLO.
- Cantidad de objetos detectados.
- Confianza promedio.
- Confianza máxima.
- FPS de procesamiento.
- Consumo de memoria RAM.
- Clases detectadas.

---

## Panel en tiempo real

Durante la ejecución, la aplicación muestra:

- FPS de la interfaz.
- Cantidad de taxis detectados.
- Consumo de memoria RAM.
- Tiempo de detección.
- FPS equivalente del detector.
- Estado actual del sistema.
- Cantidad de eventos.
- Cantidad de capturas.
- Cantidad de videos.
- Cantidad de eventos enviados.
- Estado de conexión con FastAPI.
- Latencia de la API.

Los estados principales son:

```text
ESPERANDO TAXI
EVENTO ACTIVO
GRABANDO
ENVIANDO EVENTO A FASTAPI
```

---

## Preparación del conjunto de datos

Los scripts deben ejecutarse desde la carpeta principal del proyecto.

### Verificar el conjunto de datos

```bash
python3 scripts/verificar_dataset.py
```

### Dividir el conjunto de datos

```bash
python3 scripts/dividir_dataset_generico.py
```

### Aumentar imágenes positivas

```bash
python3 scripts/aumentar_recortes_positivos.py
```

### Aumentar imágenes negativas

```bash
python3 scripts/aumentar_negativos.py
```

### Preparar el conjunto de datos para HOG

```bash
python3 scripts/preparar_dataset_hog.py
```

---

## Flujo de ejecución recomendado

Se recomienda utilizar dos terminales.

### Terminal 1: servidor FastAPI

```bash
cd /home/irar/Escritorio/ProyectoTaxiDetector
source .venv/bin/activate
python3 TelegramBot/api_server.py
```

### Terminal 2: aplicación C++

```bash
cd /home/irar/Escritorio/ProyectoTaxiDetector
./build/detectar_taxi_camara 0 5
```

Para utilizar la cámara USB:

```bash
./build/detectar_taxi_camara 2 5
```

---

## Solución de problemas

### La cámara no se abre

Comprobar las cámaras disponibles:

```bash
v4l2-ctl --list-devices
```

Probar diferentes índices:

```bash
./build/detectar_taxi_camara 0 5
./build/detectar_taxi_camara 1 5
./build/detectar_taxi_camara 2 5
```

### El modelo no se encuentra

Verificar que exista el archivo:

```bash
ls models/hog_svm_taxi.yml
```

La aplicación debe ejecutarse desde la carpeta principal del proyecto para que las rutas relativas funcionen correctamente.

### FastAPI no responde

Comprobar que el servidor se encuentre activo:

```bash
curl http://127.0.0.1:8000/health
```

### No se envían mensajes a Telegram

Comprobar:

- El token del bot.
- El identificador del chat.
- La conexión a Internet.
- La configuración del archivo `.env.local`.
- Los permisos del bot para enviar mensajes.

### Se generan detecciones falsas

Aumentar ligeramente el score mínimo del detector.

Ejemplo:

```bash
./build/detectar_taxi_imagen_v2 ruta_imagen.jpg 1.0
```

### El taxi no es detectado

Reducir ligeramente el score mínimo y comprobar que:

- El taxi sea visible.
- La iluminación sea adecuada.
- El vehículo no aparezca demasiado pequeño.
- La imagen no esté borrosa.
- La cámara tenga una resolución adecuada.

---

## Seguridad

No se deben publicar los siguientes datos:

- Token del bot de Telegram.
- Chat ID privado.
- Variables de entorno.
- Credenciales.
- Datos personales.
- Evidencias con información sensible.

Se recomienda crear un archivo `.gitignore` con el siguiente contenido:

```gitignore
.venv/
build/
.env
.env.local
__pycache__/
*.pyc
captures/
videos/
TelegramBot/input/
TelegramBot/output/
```

Si los modelos superan el límite de tamaño permitido por GitHub, también se pueden excluir:

```gitignore
models/*.yml
TelegramBot/*.pt
```

---

## Limitaciones

- El detector HOG + SVM puede verse afectado por las condiciones de iluminación.
- El rendimiento disminuye cuando el taxi aparece muy pequeño o parcialmente oculto.
- El score mínimo debe ajustarse según la cámara y el entorno.
- El procesamiento mediante YOLO puede consumir una cantidad considerable de memoria RAM.
- El tiempo de procesamiento depende del hardware utilizado.
- Los índices de las cámaras pueden cambiar entre diferentes computadores.
- Las métricas obtenidas con imágenes individuales pueden variar durante la detección en video real.

---

## Mejoras futuras

- Incrementar la diversidad del conjunto de datos.
- Incorporar imágenes de distintas ciudades y condiciones climáticas.
- Optimizar el detector para aumentar los FPS.
- Ejecutar YOLO mediante GPU.
- Incorporar seguimiento de objetos entre fotogramas.
- Mejorar la eliminación de detecciones duplicadas.
- Almacenar los eventos en una base de datos.
- Desarrollar una interfaz web para consultar las evidencias.
- Detectar buses, camionetas y otros vehículos ecuatorianos.
- Implementar compatibilidad completa con Windows.

---

## Conclusión

El presente Proyecto Integrador – Parte II permitió desarrollar e implementar un sistema funcional para la detección de taxis amarillos en tiempo real mediante OpenCV C++, utilizando el descriptor Histogram of Oriented Gradients y el clasificador Support Vector Machine. La construcción de un conjunto de datos balanceado y la aplicación de técnicas de Data Augmentation contribuyeron al entrenamiento de un modelo con un alto nivel de desempeño, alcanzando una exactitud aproximada del 99,21 %. Asimismo, la integración con FastAPI, YOLO y Telegram permitió automatizar el proceso de detección, generación de evidencias, procesamiento y envío de notificaciones en tiempo real.

---

## Referencias

Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. En *2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05)* (Vol. 1, pp. 886–893). IEEE. https://doi.org/10.1109/CVPR.2005.177

Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297. https://doi.org/10.1007/BF00994018

Bradski, G. (2000). The OpenCV library. *Dr. Dobb's Journal of Software Tools*.

---

## Licencia

Este proyecto fue desarrollado con fines académicos como parte del Proyecto Integrador – Parte II de la Carrera de Computación de la Universidad Politécnica Salesiana.
