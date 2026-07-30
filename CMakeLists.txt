cmake_minimum_required(VERSION 3.16)

project(
    TaxiDetector
    VERSION 1.0
    DESCRIPTION "Detector de taxis amarillos con OpenCV, HOG, SVM, FFmpeg y API"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)
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

function(configurar_advertencias objetivo)
    if(MSVC)
        target_compile_options(
            ${objetivo}
            PRIVATE
            /W4
        )
    else()
        target_compile_options(
            ${objetivo}
            PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()
endfunction()

# ============================================================
# HERRAMIENTAS PARA CONSTRUIR EL DATASET Y ENTRENAR EL MODELO
# ============================================================

add_executable(
    etiquetar_taxis
    src/etiquetar_taxis.cpp
)

target_include_directories(
    etiquetar_taxis
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    etiquetar_taxis
    PRIVATE
    ${OpenCV_LIBS}
)

configurar_advertencias(etiquetar_taxis)

add_executable(
    entrenar_hog_svm
    src/entrenar_hog_svm.cpp
)

target_include_directories(
    entrenar_hog_svm
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    entrenar_hog_svm
    PRIVATE
    ${OpenCV_LIBS}
)

configurar_advertencias(entrenar_hog_svm)

# ============================================================
# DETECTOR SOBRE UNA IMAGEN
# ============================================================

add_executable(
    detectar_taxi_imagen
    src/detectar_taxi_imagen.cpp
)

target_include_directories(
    detectar_taxi_imagen
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    detectar_taxi_imagen
    PRIVATE
    ${OpenCV_LIBS}
)

configurar_advertencias(detectar_taxi_imagen)

# ============================================================
# APLICACIÓN PRINCIPAL EN TIEMPO REAL
# ============================================================

add_executable(
    detectar_taxi_camara
    src/detectar_taxi_camara.cpp
    src/detector_hog.cpp
    src/capturador.cpp
    src/grabador_video.cpp
    src/cliente_api.cpp
)

target_include_directories(
    detectar_taxi_camara
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    detectar_taxi_camara
    PRIVATE
    ${OpenCV_LIBS}
    CURL::libcurl
    PkgConfig::FFMPEG
)

configurar_advertencias(detectar_taxi_camara)
