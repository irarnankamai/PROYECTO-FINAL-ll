Detector de Taxis Amarillos con HOG + SVM y YOLO

Sistema de visión por computador desarrollado para detectar taxis amarillos del entorno ecuatoriano en tiempo real.

La aplicación principal utiliza OpenCV C++, el descriptor Histogram of Oriented Gradients (HOG) y un clasificador Support Vector Machine (SVM) para realizar la detección inicial. Cuando se identifica un taxi amarillo, el sistema genera una imagen JPEG y un video MP4 de cinco segundos directamente en memoria RAM.

La evidencia se envía mediante libcurl y una solicitud HTTP multipart/form-data a una API desarrollada con FastAPI. Posteriormente, la imagen y el video son procesados mediante YOLO11n-seg y enviados automáticamente al usuario mediante un bot de Telegram.

La arquitectura actual evita crear archivos temporales de imagen y video en el disco durante el flujo de detección y envío.

Autores

Noé Leandro Ayavaca Guarango

Irar Cristian Nankamai Tsuink

Tutor: Ing. Vladimir Robles BykbaevInstitución: Universidad Politécnica SalesianaSede: CuencaCarrera: ComputaciónAño: 2026

Objetivo general

Desarrollar un sistema de visión por computador capaz de detectar taxis amarillos en tiempo real mediante OpenCV C++, utilizando el descriptor HOG y el clasificador SVM, con generación automática de evidencia en memoria RAM, procesamiento mediante FastAPI y YOLO, y envío de notificaciones a través de un bot de Telegram.

Funcionalidades principales

Captura de video en tiempo real.

Detección de taxis amarillos mediante HOG + SVM.

Visualización del bounding box.

Visualización del score de detección.

Generación automática de imágenes JPEG en memoria RAM.

Grabación automática de videos MP4 de cinco segundos en memoria RAM.

Codificación de video mediante FFmpeg.

Control de eventos para evitar envíos repetidos.

Envío HTTP multipart/form-data mediante libcurl.

Recepción de evidencias mediante FastAPI.

Procesamiento de imágenes y videos con YOLO11n-seg.

Envío de alertas y evidencias mediante Telegram.

Visualización de FPS, memoria RAM y estado del sistema.

Compatibilidad con cámara integrada y cámara USB.

Eliminación del uso de archivos temporales durante el flujo de evidencia.

Tecnologías utilizadas

Aplicación principal

C++20.

OpenCV.

CMake.

HOG.

SVM.

libcurl.

FFmpeg:

libavcodec.

libavformat.

libavutil.

libswscale.

Servidor y procesamiento

Python 3.

FastAPI.

Uvicorn.

Ultralytics YOLO.

OpenCV Python.

Telegram Bot API.

HTTPX.

Python Multipart.

Preparación del conjunto de datos

Python.

Albumentations.

NumPy.

OpenCV.

Arquitectura del sistema

Cámara
  │
  ▼
OpenCV C++
  │
  ▼
Detector HOG + SVM
  │
  ├── Imagen JPEG en RAM
  └── Video MP4 de 5 segundos en RAM
  │
  ▼
libcurl
HTTP multipart/form-data
  │
  ▼
FastAPI
  │
  ▼
YOLO11n-seg
  │
  ▼
Telegram Bot

El funcionamiento general es el siguiente:

La aplicación captura video en tiempo real mediante OpenCV C++.

Los fotogramas son analizados mediante el detector HOG + SVM.

Si no se detecta un taxi, el sistema continúa monitoreando.

Si se detecta un taxi amarillo:

se dibuja el bounding box;

se muestra el score;

se codifica una imagen JPEG en memoria RAM;

se graba un video MP4 de cinco segundos en memoria RAM.

C++ envía la imagen, el video y los metadatos a FastAPI mediante libcurl.

FastAPI recibe los datos sin depender de archivos temporales.

YOLO11n-seg procesa la imagen y el video.

El bot de Telegram envía la alerta y las evidencias.

El sistema se reactiva y continúa con el monitoreo.

Estructura del proyecto

ProyectoTaxiDetector/
├── build/
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
│   ├── capturador.hpp
│   ├── cliente_api.hpp
│   ├── detector_hog.hpp
│   └── grabador_video.hpp
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
│   ├── capturador.cpp
│   ├── cliente_api.cpp
│   ├── detectar_taxi_camara.cpp
│   ├── detectar_taxi_imagen.cpp
│   ├── detector_hog.cpp
│   ├── entrenar_hog_svm.cpp
│   ├── etiquetar_taxis.cpp
│   └── grabador_video.cpp
├── TelegramBot/
│   ├── api_server.py
│   ├── bot.py
│   ├── config.py
│   ├── video_handler.py
│   ├── video_processor.py
│   ├── yolo_processor.py
│   ├── yolo11n-seg.pt
│   └── requirements.txt
├── .env.local
├── activar_proyecto.sh
├── CMakeLists.txt
└── README.md

La imagen y el video generados durante una detección se mantienen en memoria RAM y no necesitan carpetas captures/ ni videos/.

Conjunto de datos

Para entrenar el detector se construyó un conjunto de datos balanceado compuesto por:

4.000 imágenes positivas.

4.000 imágenes negativas.

8.000 imágenes en total.

Las imágenes positivas contienen taxis amarillos, mientras que las imágenes negativas contienen otros vehículos, objetos y escenarios sin el vehículo objetivo.

Imágenes positivas originales

Inicialmente se recolectaron 777 imágenes positivas válidas.

Conjunto

Cantidad

Entrenamiento

543

Validación

117

Prueba

117

Total original

777

Las técnicas de Data Augmentation se aplicaron sobre el conjunto de entrenamiento hasta completar un total de 4.000 imágenes positivas.

Imágenes negativas originales

Se utilizaron inicialmente 900 imágenes negativas.

Conjunto

Cantidad

Entrenamiento

630

Validación

135

Prueba

135

Total original

900

Las técnicas de Data Augmentation se aplicaron sobre el conjunto de entrenamiento hasta completar un total de 4.000 imágenes negativas.

Configuración del descriptor HOG

Parámetro

Valor

Tamaño de ventana

128 × 64

Tamaño de bloque

16 × 16

Desplazamiento del bloque

8 × 8

Tamaño de celda

8 × 8

Número de bins

9

Longitud del descriptor

3780

Entrenamiento del modelo

El modelo HOG + SVM fue entrenado utilizando:

3.766 imágenes positivas de entrenamiento.

3.730 imágenes negativas de entrenamiento.

7.496 imágenes de entrenamiento en total.

El archivo generado durante el entrenamiento se encuentra en:

models/hog_svm_taxi.yml

Resultados del modelo HOG + SVM

Métricas obtenidas

Métrica

Resultado

Exactitud (accuracy)

99,21 %

Precisión

99,15 %

Sensibilidad (recall)

99,15 %

Especificidad

99,26 %

F1-score

99,15 %

Matriz de confusión

Resultado

Cantidad

Verdaderos positivos

116

Verdaderos negativos

134

Falsos positivos

1

Falsos negativos

1

Requisitos del sistema

Sistema operativo

El proyecto fue desarrollado y probado principalmente en:

Kubuntu/Linux.

Visual Studio Code.

Dependencias de C++

Compilador compatible con C++20.

OpenCV.

CMake 3.16 o superior.

pkg-config.

libcurl.

libavcodec.

libavformat.

libavutil.

libswscale.

Dependencias de Python

Python 3.

FastAPI.

Uvicorn.

Ultralytics.

OpenCV.

NumPy.

Requests.

HTTPX.

Python Multipart.

Albumentations.

Instalación en Kubuntu o Ubuntu

1. Instalar las dependencias del sistema

sudo apt update

sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libcurl4-openssl-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    python3 \
    python3-pip \
    python3-venv \
    v4l-utils

2. Ingresar en la carpeta del proyecto

cd ~/Escritorio/ProyectoTaxiDetector

3. Crear el entorno virtual

python3 -m venv .venv

4. Activar el entorno virtual

source .venv/bin/activate

5. Actualizar pip

python3 -m pip install --upgrade pip

6. Instalar las dependencias de Python

pip install -r TelegramBot/requirements.txt

Verificación de FFmpeg

Antes de compilar, se puede comprobar que pkg-config encuentre las bibliotecas:

pkg-config --modversion libavcodec
pkg-config --modversion libavformat
pkg-config --modversion libavutil
pkg-config --modversion libswscale

Para revisar las opciones de enlazado:

pkg-config --libs \
    libavcodec \
    libavformat \
    libavutil \
    libswscale

La salida debe incluir bibliotecas similares a:

-lavcodec -lavformat -lavutil -lswscale

Configuración del bot de Telegram

En la carpeta principal del proyecto debe existir un archivo llamado:

.env.local

Ejemplo:

TELEGRAM_BOT_TOKEN=COLOCAR_TOKEN_DEL_BOT
TELEGRAM_CHAT_ID=COLOCAR_CHAT_ID

No se debe publicar el token real del bot ni el identificador privado del chat.

Compilación del proyecto C++

Desde la carpeta principal del proyecto:

cmake -S . -B build

Luego compilar:

cmake --build build -j$(nproc)

Para realizar una reconstrucción limpia:

rm -rf build
cmake -S . -B build
cmake --build build -j$(nproc)

Los ejecutables se generan en:

build/

Ejecutables principales:

build/etiquetar_taxis
build/entrenar_hog_svm
build/detectar_taxi_imagen
build/detectar_taxi_camara

Ejecución del servidor FastAPI

Primero se debe activar el entorno virtual:

source .venv/bin/activate

Luego ejecutar:

python3 TelegramBot/api_server.py

La dirección local del servidor es:

http://127.0.0.1:8000

Para comprobar que la API está activa:

curl http://127.0.0.1:8000/health

Detección en una imagen

Sintaxis:

./build/detectar_taxi_imagen <ruta_imagen> [score_minimo]

Ejemplo:

./build/detectar_taxi_imagen \
    dataset/positivos/test/taxi_amarillo_test_0001.jpg \
    0.8

El parámetro score_minimo es opcional.

Un valor bajo aumenta la sensibilidad, pero puede generar más falsos positivos. Un valor alto reduce los falsos positivos, pero también puede impedir la detección de algunos taxis.

Detección mediante cámara

Sintaxis:

./build/detectar_taxi_camara <indice_camara> <intervalo_frames>

Ejemplo con cámara integrada:

./build/detectar_taxi_camara 0 5

Ejemplo con cámara USB:

./build/detectar_taxi_camara 2 5

Ejemplo con una detección cada diez fotogramas:

./build/detectar_taxi_camara 0 10

Parámetros

indice_camara: índice asignado a la cámara por el sistema operativo.

intervalo_frames: cantidad de fotogramas entre cada ejecución del detector.

Identificar cámaras disponibles en Linux

Para listar las cámaras disponibles:

v4l2-ctl --list-devices

También se puede ejecutar:

ls /dev/video*

Durante las pruebas se utilizaron:

Cámara integrada: /dev/video0.

Cámara USB: /dev/video2.

Los índices pueden cambiar según el computador y el orden en que se conecten las cámaras.

Controles de la aplicación

Tecla

Acción

Q

Cerrar la aplicación

ESC

Cerrar la aplicación

Evidencia generada

Cuando se detecta un taxi amarillo, el sistema:

codifica una imagen JPEG mediante cv::imencode;

mantiene los bytes de la imagen en un std::vector<unsigned char>;

codifica un video MP4 de cinco segundos mediante FFmpeg;

mantiene los bytes del video en un std::vector<unsigned char>;

envía ambos contenidos a FastAPI mediante libcurl;

libera la memoria después de completar el envío.

No se utilizan cv::imwrite, cv::VideoWriter, curl_mime_filedata ni archivos temporales para transferir la evidencia.

La transferencia utiliza datos en memoria:

imagenDatos
videoDatos
nombreImagen
nombreVideo

En libcurl se utiliza:

curl_mime_data(...)

Información enviada a FastAPI

El evento enviado desde C++ puede incluir:

Imagen JPEG.

Video MP4.

Nombre de la imagen.

Nombre del video.

Vehículo objetivo.

Identificador o índice de la cámara.

Fecha y hora.

Confianza obtenida por el detector C++.

Metadatos adicionales del evento.

Los nombres exactos de los campos deben coincidir entre cliente_api.cpp y el endpoint de FastAPI.

Información enviada a Telegram

El bot puede enviar:

Alerta de detección.

Identificador del evento.

Vehículo objetivo.

Cámara utilizada.

Fecha y hora.

Imagen recibida.

Imagen procesada mediante YOLO.

Video recibido.

Video procesado mediante YOLO.

Cantidad de objetos detectados.

Confianza promedio.

Confianza máxima.

FPS de procesamiento.

Consumo de memoria RAM.

Clases detectadas.

La imagen y el video se transfieren desde memoria RAM durante el flujo principal.

Panel en tiempo real

Durante la ejecución, la aplicación muestra:

FPS de la interfaz.

Cantidad de taxis detectados.

Consumo de memoria RAM.

Tiempo de detección.

FPS equivalente del detector.

Estado actual del sistema.

Cantidad de eventos.

Cantidad de capturas.

Cantidad de videos.

Cantidad de eventos enviados.

Estado de conexión con FastAPI.

Latencia de la API.

Los estados principales son:

ESPERANDO TAXI
EVENTO ACTIVO
GRABANDO
ENVIANDO EVENTO A FASTAPI

Durante la grabación también se muestra un indicador visual.

Control de eventos

El sistema evita enviar continuamente el mismo taxi.

El flujo general es:

se confirma la detección;

se inicia un evento;

se genera la imagen;

se graba el video;

se envía la evidencia;

el sistema espera varios resultados sin detección;

se reactiva para permitir un nuevo evento.

Este mecanismo reduce capturas y mensajes repetidos en Telegram.

Preparación del conjunto de datos

Los scripts deben ejecutarse desde la carpeta principal del proyecto.

Verificar el conjunto de datos

python3 scripts/verificar_dataset.py

Dividir el conjunto de datos

python3 scripts/dividir_dataset_generico.py

Aumentar imágenes positivas

python3 scripts/aumentar_recortes_positivos.py

Aumentar imágenes negativas

python3 scripts/aumentar_negativos.py

Preparar el conjunto de datos para HOG

python3 scripts/preparar_dataset_hog.py

Flujo de ejecución recomendado

Se recomienda utilizar dos terminales.

Terminal 1: servidor FastAPI

cd ~/Escritorio/ProyectoTaxiDetector
source .venv/bin/activate
python3 TelegramBot/api_server.py

Comprobar el servidor:

curl http://127.0.0.1:8000/health

Terminal 2: aplicación C++

cd ~/Escritorio/ProyectoTaxiDetector
./build/detectar_taxi_camara 0 5

Para utilizar una cámara USB:

./build/detectar_taxi_camara 2 5

Prueba integral del sistema

Para validar la integración completa:

iniciar FastAPI;

comprobar el endpoint /health;

ejecutar el detector C++;

mostrar un taxi amarillo ante la cámara;

esperar la confirmación de detección;

esperar los cinco segundos de grabación;

comprobar que el video MP4 se finalice en memoria;

comprobar que C++ envíe el evento a FastAPI;

verificar que FastAPI responda con estado HTTP exitoso;

confirmar la recepción de la alerta, la imagen y el video en Telegram.

Mensajes esperados en C++:

CAPTURA GENERADA EN MEMORIA RAM
GRABACION DE VIDEO INICIADA EN RAM
VIDEO MP4 FINALIZADO EN MEMORIA RAM
ENVIANDO EVENTO A FASTAPI
ULTIMO ENVIO CORRECTO

Solución de problemas

La cámara no se abre

Comprobar las cámaras disponibles:

v4l2-ctl --list-devices

Probar diferentes índices:

./build/detectar_taxi_camara 0 5
./build/detectar_taxi_camara 1 5
./build/detectar_taxi_camara 2 5

El modelo no se encuentra

Verificar que exista:

ls models/hog_svm_taxi.yml

La aplicación debe ejecutarse desde la carpeta principal del proyecto para que las rutas relativas funcionen correctamente.

FastAPI no responde

Comprobar que el servidor esté activo:

curl http://127.0.0.1:8000/health

También se debe verificar que la dirección configurada en C++ sea:

http://127.0.0.1:8000

Error DSO missing from command line

Este error indica que FFmpeg no está enlazado correctamente.

El CMakeLists.txt debe incluir:

find_package(PkgConfig REQUIRED)

pkg_check_modules(
    FFMPEG
    REQUIRED
    IMPORTED_TARGET
    libavcodec
    libavformat
    libavutil
    libswscale
)

Y el ejecutable de cámara debe enlazarse con:

target_link_libraries(
    detectar_taxi_camara
    PRIVATE
    ${OpenCV_LIBS}
    CURL::libcurl
    PkgConfig::FFMPEG
)

Después se debe regenerar la compilación:

rm -rf build
cmake -S . -B build
cmake --build build -j$(nproc)

Error en avio_alloc_context

En algunas versiones de FFmpeg, el callback de escritura debe recibir un puntero sin const:

static int escribirEnMemoria(
    void* opaco,
    std::uint8_t* datos,
    int tamano
);

No se envían mensajes a Telegram

Comprobar:

el token del bot;

el identificador del chat;

la conexión a Internet;

la configuración del archivo .env.local;

los permisos del bot;

la respuesta de FastAPI;

que los nombres multipart coincidan entre C++ y Python.

Se generan detecciones falsas

Aumentar ligeramente el score mínimo:

./build/detectar_taxi_imagen ruta_imagen.jpg 1.0

El taxi no es detectado

Reducir ligeramente el score mínimo y comprobar que:

el taxi sea visible;

la iluminación sea adecuada;

el vehículo no aparezca demasiado pequeño;

la imagen no esté borrosa;

la cámara tenga una resolución adecuada.

El video llega vacío

Comprobar que:

la grabación haya finalizado antes de extraer el vector;

grabacionCompletada() devuelva true;

extraerVideoEnMemoria() se llame después de finalizar;

videoDatos no esté vacío antes del envío;

FastAPI lea completamente el archivo recibido.

Seguridad

No se deben publicar:

token del bot de Telegram;

Chat ID privado;

variables de entorno;

credenciales;

datos personales;

evidencias con información sensible.

Se recomienda utilizar este .gitignore:

.venv/
build/
.env
.env.local
__pycache__/
*.pyc
*.pyo
*.log
TelegramBot/input/
TelegramBot/output/

Si los modelos superan el límite permitido por GitHub:

models/*.yml
TelegramBot/*.pt

No es necesario excluir carpetas captures/ o videos/ si ya no forman parte de la arquitectura.

Limitaciones

El detector HOG + SVM puede verse afectado por las condiciones de iluminación.

El rendimiento disminuye cuando el taxi aparece muy pequeño o parcialmente oculto.

El score mínimo debe ajustarse según la cámara y el entorno.

El procesamiento mediante YOLO puede consumir una cantidad considerable de memoria RAM.

El video en memoria aumenta temporalmente el consumo de RAM durante cada evento.

El tiempo de procesamiento depende del hardware utilizado.

Los índices de las cámaras pueden cambiar entre computadores.

Las métricas obtenidas con imágenes individuales pueden variar durante la detección en video real.

La aplicación depende de que FastAPI y la conexión a Internet estén disponibles para completar el envío a Telegram.

Mejoras futuras

Incrementar la diversidad del conjunto de datos.

Incorporar imágenes de distintas ciudades y condiciones climáticas.

Optimizar el detector para aumentar los FPS.

Ejecutar YOLO mediante GPU.

Incorporar seguimiento de objetos entre fotogramas.

Mejorar la eliminación de detecciones duplicadas.

Implementar una cola de reintentos para envíos fallidos.

Almacenar metadatos de eventos en una base de datos.

Desarrollar una interfaz web para consultar estadísticas.

Detectar buses, camionetas y otros vehículos ecuatorianos.

Implementar compatibilidad completa con Windows.

Agregar pruebas automatizadas para C++, FastAPI y Telegram.

Conclusión

El Proyecto Integrador – Parte II permitió desarrollar e implementar un sistema funcional para la detección de taxis amarillos en tiempo real mediante OpenCV C++, utilizando el descriptor Histogram of Oriented Gradients y el clasificador Support Vector Machine. La construcción de un conjunto de datos balanceado y la aplicación de técnicas de Data Augmentation contribuyeron al entrenamiento de un modelo con un alto nivel de desempeño, alcanzando una exactitud aproximada del 99,21 %.

La integración entre la aplicación C++, FastAPI, YOLO11n-seg y Telegram permitió automatizar el proceso de detección, generación de evidencias, procesamiento y envío de notificaciones. Además, la arquitectura fue optimizada para codificar y transferir la imagen y el video directamente desde memoria RAM, eliminando archivos temporales y reduciendo operaciones de entrada y salida sobre el sistema de archivos. Esta solución mejora la eficiencia, la organización y la robustez general del sistema.

Referencias

Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. En 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05) (Vol. 1, pp. 886–893). IEEE. https://doi.org/10.1109/CVPR.2005.177

Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273–297. https://doi.org/10.1007/BF00994018

Bradski, G. (2000). The OpenCV library. Dr. Dobb's Journal of Software Tools.

Licencia

Este proyecto fue desarrollado con fines académicos como parte del Proyecto Integrador – Parte II de la Carrera de Computación de la Universidad Politécnica Salesiana.