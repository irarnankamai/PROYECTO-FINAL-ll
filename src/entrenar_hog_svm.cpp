#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

constexpr int ANCHO_HOG = 128;
constexpr int ALTO_HOG = 64;
constexpr int SEMILLA = 42;

struct Muestra
{
    std::vector<float> descriptor;
    int etiqueta = 0;
    std::string ruta;
};

struct Metricas
{
    int verdaderosPositivos = 0;
    int verdaderosNegativos = 0;
    int falsosPositivos = 0;
    int falsosNegativos = 0;

    double accuracy = 0.0;
    double precision = 0.0;
    double sensibilidad = 0.0;
    double especificidad = 0.0;
    double f1 = 0.0;
    double tiempoPromedioMs = 0.0;
};

bool esImagenValida(const fs::path& ruta)
{
    std::string extension = ruta.extension().string();

    std::transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](unsigned char c)
        {
            return static_cast<char>(std::tolower(c));
        }
    );

    return extension == ".jpg"
        || extension == ".jpeg"
        || extension == ".png"
        || extension == ".webp";
}

std::vector<fs::path> obtenerImagenes(const fs::path& carpeta)
{
    if (!fs::exists(carpeta) || !fs::is_directory(carpeta))
    {
        throw std::runtime_error(
            "No existe la carpeta: " + carpeta.string()
        );
    }

    std::vector<fs::path> imagenes;

    for (const auto& entrada : fs::directory_iterator(carpeta))
    {
        if (entrada.is_regular_file() && esImagenValida(entrada.path()))
        {
            imagenes.push_back(entrada.path());
        }
    }

    std::sort(imagenes.begin(), imagenes.end());
    return imagenes;
}

cv::HOGDescriptor crearHog()
{
    return cv::HOGDescriptor(
        cv::Size(ANCHO_HOG, ALTO_HOG),
        cv::Size(16, 16),
        cv::Size(8, 8),
        cv::Size(8, 8),
        9
    );
}

std::vector<float> extraerDescriptor(
    const cv::Mat& imagenOriginal,
    cv::HOGDescriptor& hog
)
{
    if (imagenOriginal.empty())
    {
        throw std::runtime_error("La imagen está vacía.");
    }

    cv::Mat redimensionada;

    cv::resize(
        imagenOriginal,
        redimensionada,
        cv::Size(ANCHO_HOG, ALTO_HOG),
        0.0,
        0.0,
        cv::INTER_AREA
    );

    cv::Mat gris;

    if (redimensionada.channels() == 1)
    {
        gris = redimensionada;
    }
    else if (redimensionada.channels() == 3)
    {
        cv::cvtColor(redimensionada, gris, cv::COLOR_BGR2GRAY);
    }
    else if (redimensionada.channels() == 4)
    {
        cv::cvtColor(redimensionada, gris, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        throw std::runtime_error("Número de canales no compatible.");
    }

    std::vector<float> descriptor;

    hog.compute(
        gris,
        descriptor,
        cv::Size(8, 8),
        cv::Size(0, 0)
    );

    if (descriptor.empty())
    {
        throw std::runtime_error("No se pudo calcular el descriptor HOG.");
    }

    return descriptor;
}

void cargarCarpeta(
    const fs::path& carpeta,
    int etiqueta,
    cv::HOGDescriptor& hog,
    std::vector<Muestra>& muestras
)
{
    const auto imagenes = obtenerImagenes(carpeta);

    std::cout
        << "Cargando "
        << carpeta
        << ": "
        << imagenes.size()
        << " imágenes\n";

    int procesadas = 0;
    int omitidas = 0;

    for (const auto& ruta : imagenes)
    {
        try
        {
            const cv::Mat imagen =
                cv::imread(ruta.string(), cv::IMREAD_UNCHANGED);

            if (imagen.empty())
            {
                ++omitidas;
                std::cerr << "No se pudo leer: " << ruta << '\n';
                continue;
            }

            muestras.push_back({
                extraerDescriptor(imagen, hog),
                etiqueta,
                ruta.string()
            });

            ++procesadas;

            if (procesadas % 500 == 0)
            {
                std::cout
                    << "  Procesadas: "
                    << procesadas
                    << "/"
                    << imagenes.size()
                    << '\n';
            }
        }
        catch (const std::exception& error)
        {
            ++omitidas;
            std::cerr
                << "Advertencia en "
                << ruta
                << ": "
                << error.what()
                << '\n';
        }
    }

    std::cout
        << "  Cargadas: "
        << procesadas
        << " | Omitidas: "
        << omitidas
        << '\n';
}

cv::Mat crearMatrizCaracteristicas(
    const std::vector<Muestra>& muestras
)
{
    if (muestras.empty())
    {
        throw std::runtime_error("No hay muestras.");
    }

    const int filas = static_cast<int>(muestras.size());
    const int columnas =
        static_cast<int>(muestras.front().descriptor.size());

    cv::Mat datos(filas, columnas, CV_32F);

    for (int fila = 0; fila < filas; ++fila)
    {
        if (
            static_cast<int>(muestras[fila].descriptor.size())
            != columnas
        )
        {
            throw std::runtime_error(
                "Los descriptores HOG tienen dimensiones diferentes."
            );
        }

        std::copy(
            muestras[fila].descriptor.begin(),
            muestras[fila].descriptor.end(),
            datos.ptr<float>(fila)
        );
    }

    return datos;
}

cv::Mat crearVectorEtiquetas(
    const std::vector<Muestra>& muestras
)
{
    cv::Mat etiquetas(
        static_cast<int>(muestras.size()),
        1,
        CV_32S
    );

    for (int i = 0; i < etiquetas.rows; ++i)
    {
        etiquetas.at<int>(i, 0) = muestras[i].etiqueta;
    }

    return etiquetas;
}

std::string escaparCsv(const std::string& texto)
{
    if (
        texto.find(',') == std::string::npos
        && texto.find('"') == std::string::npos
    )
    {
        return texto;
    }

    std::string resultado = "\"";

    for (const char c : texto)
    {
        resultado += (c == '"') ? "\"\"" : std::string(1, c);
    }

    resultado += '"';
    return resultado;
}

Metricas evaluarModelo(
    const cv::Ptr<cv::ml::SVM>& svm,
    const std::vector<Muestra>& muestras,
    const fs::path& archivoPredicciones
)
{
    if (muestras.empty())
    {
        throw std::runtime_error("No hay muestras para evaluar.");
    }

    fs::create_directories(archivoPredicciones.parent_path());

    std::ofstream csv(archivoPredicciones);

    if (!csv.is_open())
    {
        throw std::runtime_error(
            "No se pudo abrir: " + archivoPredicciones.string()
        );
    }

    csv << "ruta,etiqueta_real,etiqueta_predicha,correcta\n";

    Metricas metricas;
    double tiempoTotalMs = 0.0;

    for (const auto& muestra : muestras)
    {
        cv::Mat fila(
            1,
            static_cast<int>(muestra.descriptor.size()),
            CV_32F
        );

        std::copy(
            muestra.descriptor.begin(),
            muestra.descriptor.end(),
            fila.ptr<float>(0)
        );

        const auto inicio = std::chrono::steady_clock::now();
        const int predicha =
            static_cast<int>(std::lround(svm->predict(fila)));
        const auto fin = std::chrono::steady_clock::now();

        tiempoTotalMs +=
            std::chrono::duration<double, std::milli>(
                fin - inicio
            ).count();

        const int real = muestra.etiqueta;

        if (real == 1 && predicha == 1)
        {
            ++metricas.verdaderosPositivos;
        }
        else if (real == -1 && predicha == -1)
        {
            ++metricas.verdaderosNegativos;
        }
        else if (real == -1 && predicha == 1)
        {
            ++metricas.falsosPositivos;
        }
        else if (real == 1 && predicha == -1)
        {
            ++metricas.falsosNegativos;
        }

        csv
            << escaparCsv(muestra.ruta) << ','
            << real << ','
            << predicha << ','
            << (real == predicha ? 1 : 0)
            << '\n';
    }

    const double total = static_cast<double>(muestras.size());

    metricas.accuracy =
        static_cast<double>(
            metricas.verdaderosPositivos
            + metricas.verdaderosNegativos
        ) / total;

    const double dPrecision =
        metricas.verdaderosPositivos
        + metricas.falsosPositivos;

    metricas.precision =
        dPrecision > 0.0
        ? metricas.verdaderosPositivos / dPrecision
        : 0.0;

    const double dSensibilidad =
        metricas.verdaderosPositivos
        + metricas.falsosNegativos;

    metricas.sensibilidad =
        dSensibilidad > 0.0
        ? metricas.verdaderosPositivos / dSensibilidad
        : 0.0;

    const double dEspecificidad =
        metricas.verdaderosNegativos
        + metricas.falsosPositivos;

    metricas.especificidad =
        dEspecificidad > 0.0
        ? metricas.verdaderosNegativos / dEspecificidad
        : 0.0;

    const double suma =
        metricas.precision + metricas.sensibilidad;

    metricas.f1 =
        suma > 0.0
        ? 2.0 * metricas.precision * metricas.sensibilidad / suma
        : 0.0;

    metricas.tiempoPromedioMs =
        tiempoTotalMs / total;

    return metricas;
}

void imprimirMetricas(
    const std::string& nombre,
    const Metricas& m
)
{
    std::cout
        << "\n=============================================\n"
        << "RESULTADOS: " << nombre
        << "\n=============================================\n"
        << std::fixed
        << std::setprecision(4)
        << "Accuracy:      " << m.accuracy
        << " (" << m.accuracy * 100.0 << "%)\n"
        << "Precisión:     " << m.precision
        << " (" << m.precision * 100.0 << "%)\n"
        << "Sensibilidad:  " << m.sensibilidad
        << " (" << m.sensibilidad * 100.0 << "%)\n"
        << "Especificidad: " << m.especificidad
        << " (" << m.especificidad * 100.0 << "%)\n"
        << "F1-score:      " << m.f1
        << " (" << m.f1 * 100.0 << "%)\n"
        << "Tiempo promedio: "
        << m.tiempoPromedioMs
        << " ms\n\n"
        << "Matriz de confusión:\n"
        << "                     Predicho +  Predicho -\n"
        << "Real positivo       "
        << std::setw(10) << m.verdaderosPositivos
        << std::setw(12) << m.falsosNegativos
        << '\n'
        << "Real negativo       "
        << std::setw(10) << m.falsosPositivos
        << std::setw(12) << m.verdaderosNegativos
        << '\n';
}

void guardarMetricas(
    const fs::path& archivo,
    const Metricas& validacion,
    const Metricas& prueba,
    double tiempoEntrenamiento,
    int cantidadTrain,
    int longitudDescriptor
)
{
    fs::create_directories(archivo.parent_path());

    std::ofstream csv(archivo);

    if (!csv.is_open())
    {
        throw std::runtime_error(
            "No se pudo guardar: " + archivo.string()
        );
    }

    csv
        << "conjunto,accuracy,precision,sensibilidad,"
        << "especificidad,f1,vp,vn,fp,fn,"
        << "tiempo_prediccion_ms,"
        << "tiempo_entrenamiento_segundos,"
        << "muestras_train,longitud_descriptor\n"
        << std::fixed
        << std::setprecision(8);

    const auto fila =
        [&](const std::string& nombre, const Metricas& m)
    {
        csv
            << nombre << ','
            << m.accuracy << ','
            << m.precision << ','
            << m.sensibilidad << ','
            << m.especificidad << ','
            << m.f1 << ','
            << m.verdaderosPositivos << ','
            << m.verdaderosNegativos << ','
            << m.falsosPositivos << ','
            << m.falsosNegativos << ','
            << m.tiempoPromedioMs << ','
            << tiempoEntrenamiento << ','
            << cantidadTrain << ','
            << longitudDescriptor
            << '\n';
    };

    fila("validation", validacion);
    fila("test", prueba);
}

std::string porcentaje(double valor)
{
    std::ostringstream salida;
    salida
        << std::fixed
        << std::setprecision(2)
        << valor * 100.0
        << "%";
    return salida.str();
}

void textoCentrado(
    cv::Mat& imagen,
    const std::string& texto,
    const cv::Rect& area,
    double escala,
    int grosor
)
{
    int base = 0;

    const cv::Size tamano =
        cv::getTextSize(
            texto,
            cv::FONT_HERSHEY_SIMPLEX,
            escala,
            grosor,
            &base
        );

    cv::putText(
        imagen,
        texto,
        cv::Point(
            area.x + (area.width - tamano.width) / 2,
            area.y + (area.height + tamano.height) / 2
        ),
        cv::FONT_HERSHEY_SIMPLEX,
        escala,
        cv::Scalar(35, 35, 35),
        grosor,
        cv::LINE_AA
    );
}

void guardarMatrizConfusion(
    const Metricas& m,
    const std::string& nombre,
    const fs::path& archivo
)
{
    cv::Mat imagen(
        700,
        1000,
        CV_8UC3,
        cv::Scalar(255, 255, 255)
    );

    cv::putText(
        imagen,
        "Matriz de confusion - " + nombre,
        cv::Point(210, 65),
        cv::FONT_HERSHEY_SIMPLEX,
        1.05,
        cv::Scalar(35, 35, 35),
        2,
        cv::LINE_AA
    );

    constexpr int x0 = 330;
    constexpr int y0 = 190;
    constexpr int ancho = 250;
    constexpr int alto = 170;

    textoCentrado(
        imagen,
        "Taxi (+)",
        cv::Rect(x0, 120, ancho, 60),
        0.72,
        2
    );

    textoCentrado(
        imagen,
        "No taxi (-)",
        cv::Rect(x0 + ancho, 120, ancho, 60),
        0.72,
        2
    );

    textoCentrado(
        imagen,
        "Taxi (+)",
        cv::Rect(45, y0, 250, alto),
        0.72,
        2
    );

    textoCentrado(
        imagen,
        "No taxi (-)",
        cv::Rect(45, y0 + alto, 250, alto),
        0.72,
        2
    );

    const auto celda =
        [&](int columna,
            int fila,
            int valor,
            const std::string& etiqueta,
            const cv::Scalar& fondo)
    {
        const cv::Rect area(
            x0 + columna * ancho,
            y0 + fila * alto,
            ancho,
            alto
        );

        cv::rectangle(imagen, area, fondo, cv::FILLED);
        cv::rectangle(
            imagen,
            area,
            cv::Scalar(50, 50, 50),
            2
        );

        textoCentrado(
            imagen,
            etiqueta + " = " + std::to_string(valor),
            area,
            0.95,
            3
        );
    };

    celda(
        0, 0,
        m.verdaderosPositivos,
        "VP",
        cv::Scalar(205, 240, 205)
    );

    celda(
        1, 0,
        m.falsosNegativos,
        "FN",
        cv::Scalar(215, 225, 255)
    );

    celda(
        0, 1,
        m.falsosPositivos,
        "FP",
        cv::Scalar(215, 225, 255)
    );

    celda(
        1, 1,
        m.verdaderosNegativos,
        "VN",
        cv::Scalar(205, 240, 205)
    );

    cv::putText(
        imagen,
        "Accuracy: " + porcentaje(m.accuracy),
        cv::Point(330, 585),
        cv::FONT_HERSHEY_SIMPLEX,
        0.75,
        cv::Scalar(35, 35, 35),
        2,
        cv::LINE_AA
    );

    cv::putText(
        imagen,
        "F1-score: " + porcentaje(m.f1),
        cv::Point(625, 585),
        cv::FONT_HERSHEY_SIMPLEX,
        0.75,
        cv::Scalar(35, 35, 35),
        2,
        cv::LINE_AA
    );

    fs::create_directories(archivo.parent_path());

    if (!cv::imwrite(archivo.string(), imagen))
    {
        throw std::runtime_error(
            "No se pudo guardar: " + archivo.string()
        );
    }
}

void guardarGraficaMetricas(
    const Metricas& m,
    const std::string& nombre,
    const fs::path& archivo
)
{
    cv::Mat imagen(
        760,
        1100,
        CV_8UC3,
        cv::Scalar(255, 255, 255)
    );

    constexpr int izquierda = 125;
    constexpr int arriba = 110;
    constexpr int anchoArea = 910;
    constexpr int altoArea = 525;

    cv::putText(
        imagen,
        "Metricas de evaluacion - " + nombre,
        cv::Point(245, 62),
        cv::FONT_HERSHEY_SIMPLEX,
        1.05,
        cv::Scalar(35, 35, 35),
        2,
        cv::LINE_AA
    );

    for (int p = 0; p <= 100; p += 10)
    {
        const int y =
            arriba + altoArea
            - static_cast<int>(altoArea * p / 100.0);

        cv::line(
            imagen,
            cv::Point(izquierda, y),
            cv::Point(izquierda + anchoArea, y),
            cv::Scalar(225, 225, 225),
            1
        );

        cv::putText(
            imagen,
            std::to_string(p) + "%",
            cv::Point(55, y + 7),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(35, 35, 35),
            1,
            cv::LINE_AA
        );
    }

    cv::line(
        imagen,
        cv::Point(izquierda, arriba),
        cv::Point(izquierda, arriba + altoArea),
        cv::Scalar(45, 45, 45),
        2
    );

    cv::line(
        imagen,
        cv::Point(izquierda, arriba + altoArea),
        cv::Point(izquierda + anchoArea, arriba + altoArea),
        cv::Scalar(45, 45, 45),
        2
    );

    const std::vector<std::string> nombres = {
        "Precision",
        "Sensibilidad",
        "Especificidad"
    };

    const std::vector<double> valores = {
        m.precision,
        m.sensibilidad,
        m.especificidad
    };

    const std::vector<cv::Scalar> colores = {
        cv::Scalar(220, 160, 80),
        cv::Scalar(110, 190, 110),
        cv::Scalar(110, 140, 220)
    };

    constexpr int espacio = anchoArea / 3;
    constexpr int anchoBarra = 165;

    for (int i = 0; i < 3; ++i)
    {
        const double valor = std::clamp(valores[i], 0.0, 1.0);
        const int altura =
            std::max(1, static_cast<int>(valor * altoArea));

        const int centroX =
            izquierda + espacio * i + espacio / 2;

        const cv::Rect barra(
            centroX - anchoBarra / 2,
            arriba + altoArea - altura,
            anchoBarra,
            altura
        );

        cv::rectangle(imagen, barra, colores[i], cv::FILLED);
        cv::rectangle(
            imagen,
            barra,
            cv::Scalar(45, 45, 45),
            2
        );

        textoCentrado(
            imagen,
            porcentaje(valor),
            cv::Rect(
                centroX - 110,
                std::max(arriba, barra.y - 55),
                220,
                45
            ),
            0.68,
            2
        );

        textoCentrado(
            imagen,
            nombres[i],
            cv::Rect(
                centroX - 130,
                arriba + altoArea + 20,
                260,
                60
            ),
            0.60,
            2
        );
    }

    fs::create_directories(archivo.parent_path());

    if (!cv::imwrite(archivo.string(), imagen))
    {
        throw std::runtime_error(
            "No se pudo guardar: " + archivo.string()
        );
    }
}

void guardarResumen(
    const fs::path& archivo,
    const Metricas& validacion,
    const Metricas& prueba,
    double tiempoEntrenamiento,
    int muestrasTrain,
    int longitudDescriptor
)
{
    fs::create_directories(archivo.parent_path());

    std::ofstream salida(archivo);

    if (!salida.is_open())
    {
        throw std::runtime_error(
            "No se pudo guardar: " + archivo.string()
        );
    }

    salida
        << std::fixed
        << std::setprecision(4)
        << "MODELO HOG + SVM PARA TAXIS AMARILLOS\n"
        << "=====================================\n\n"
        << "Ventana HOG: " << ANCHO_HOG << " x " << ALTO_HOG << '\n'
        << "Longitud descriptor: " << longitudDescriptor << '\n'
        << "Muestras train: " << muestrasTrain << '\n'
        << "Tiempo entrenamiento: "
        << tiempoEntrenamiento
        << " segundos\n"
        << "SVM: C_SVC lineal\n"
        << "C: 0.01\n"
        << "Semilla: " << SEMILLA
        << "\n\n";

    const auto conjunto =
        [&](const std::string& nombre, const Metricas& m)
    {
        salida
            << nombre << '\n'
            << "Accuracy: " << m.accuracy * 100.0 << " %\n"
            << "Precisión: " << m.precision * 100.0 << " %\n"
            << "Sensibilidad: "
            << m.sensibilidad * 100.0
            << " %\n"
            << "Especificidad: "
            << m.especificidad * 100.0
            << " %\n"
            << "F1-score: " << m.f1 * 100.0 << " %\n"
            << "VP: " << m.verdaderosPositivos << '\n'
            << "VN: " << m.verdaderosNegativos << '\n'
            << "FP: " << m.falsosPositivos << '\n'
            << "FN: " << m.falsosNegativos << "\n\n";
    };

    conjunto("VALIDATION", validacion);
    conjunto("TEST", prueba);
}

int main()
{
    try
    {
        const fs::path raiz = fs::current_path();

        const fs::path trainPositivos =
            raiz / "dataset/hog/train/positivos";

        const fs::path trainNegativos =
            raiz / "dataset/hog/train/negativos";

        const fs::path validationPositivos =
            raiz / "dataset/hog/validation/positivos";

        const fs::path validationNegativos =
            raiz / "dataset/hog/validation/negativos";

        const fs::path testPositivos =
            raiz / "dataset/hog/test/positivos";

        const fs::path testNegativos =
            raiz / "dataset/hog/test/negativos";

        const fs::path modeloSalida =
            raiz / "models/hog_svm_taxi.yml";

        const fs::path resultados =
            raiz / "resultados";

        const fs::path graficas =
            resultados / "graficas";

        cv::HOGDescriptor hog = crearHog();

        std::cout
            << "=============================================\n"
            << "ENTRENAMIENTO HOG + SVM\n"
            << "=============================================\n"
            << "Proyecto: " << raiz << '\n'
            << "Ventana HOG: "
            << ANCHO_HOG << "x" << ALTO_HOG << '\n'
            << "Longitud descriptor: "
            << hog.getDescriptorSize()
            << "\n\n";

        std::vector<Muestra> train;
        std::vector<Muestra> validation;
        std::vector<Muestra> test;

        cargarCarpeta(trainPositivos, 1, hog, train);
        cargarCarpeta(trainNegativos, -1, hog, train);
        cargarCarpeta(validationPositivos, 1, hog, validation);
        cargarCarpeta(validationNegativos, -1, hog, validation);
        cargarCarpeta(testPositivos, 1, hog, test);
        cargarCarpeta(testNegativos, -1, hog, test);

        if (train.empty() || validation.empty() || test.empty())
        {
            throw std::runtime_error(
                "Uno o más conjuntos están vacíos."
            );
        }

        std::mt19937 generador(SEMILLA);
        std::shuffle(train.begin(), train.end(), generador);

        std::cout
            << "\nTrain: " << train.size()
            << "\nValidation: " << validation.size()
            << "\nTest: " << test.size()
            << "\n\nCreando matriz HOG...\n";

        const cv::Mat datosTrain =
            crearMatrizCaracteristicas(train);

        const cv::Mat etiquetasTrain =
            crearVectorEtiquetas(train);

        std::cout
            << "Matriz: "
            << datosTrain.rows
            << " x "
            << datosTrain.cols
            << '\n';

        cv::Ptr<cv::ml::SVM> svm =
            cv::ml::SVM::create();

        svm->setType(cv::ml::SVM::C_SVC);
        svm->setKernel(cv::ml::SVM::LINEAR);
        svm->setC(0.01);

        svm->setTermCriteria(
            cv::TermCriteria(
                cv::TermCriteria::MAX_ITER
                    | cv::TermCriteria::EPS,
                10000,
                1e-6
            )
        );

        std::cout << "\nEntrenando SVM lineal...\n";

        const auto inicio = std::chrono::steady_clock::now();

        const bool entrenado =
            svm->train(
                datosTrain,
                cv::ml::ROW_SAMPLE,
                etiquetasTrain
            );

        const auto fin = std::chrono::steady_clock::now();

        if (!entrenado || !svm->isTrained())
        {
            throw std::runtime_error(
                "El SVM no pudo entrenarse."
            );
        }

        const double tiempoEntrenamiento =
            std::chrono::duration<double>(fin - inicio).count();

        fs::create_directories(modeloSalida.parent_path());
        svm->save(modeloSalida.string());

        if (
            !fs::exists(modeloSalida)
            || fs::file_size(modeloSalida) == 0
        )
        {
            throw std::runtime_error(
                "El modelo no se guardó correctamente."
            );
        }

        std::cout
            << "Entrenamiento completado en "
            << tiempoEntrenamiento
            << " segundos.\n"
            << "Modelo guardado en:\n"
            << modeloSalida
            << '\n';

        const Metricas metricasValidation =
            evaluarModelo(
                svm,
                validation,
                resultados / "predicciones_validation.csv"
            );

        const Metricas metricasTest =
            evaluarModelo(
                svm,
                test,
                resultados / "predicciones_test.csv"
            );

        imprimirMetricas("VALIDATION", metricasValidation);
        imprimirMetricas("TEST", metricasTest);

        guardarMetricas(
            resultados / "metricas_hog_svm.csv",
            metricasValidation,
            metricasTest,
            tiempoEntrenamiento,
            static_cast<int>(train.size()),
            static_cast<int>(hog.getDescriptorSize())
        );

        guardarMatrizConfusion(
            metricasValidation,
            "VALIDATION",
            graficas / "matriz_confusion_validation.png"
        );

        guardarMatrizConfusion(
            metricasTest,
            "TEST",
            graficas / "matriz_confusion_test.png"
        );

        guardarGraficaMetricas(
            metricasValidation,
            "VALIDATION",
            graficas / "metricas_validation.png"
        );

        guardarGraficaMetricas(
            metricasTest,
            "TEST",
            graficas / "metricas_test.png"
        );

        guardarResumen(
            resultados / "resumen_modelo_hog_svm.txt",
            metricasValidation,
            metricasTest,
            tiempoEntrenamiento,
            static_cast<int>(train.size()),
            static_cast<int>(hog.getDescriptorSize())
        );

        std::cout
            << "\n=============================================\n"
            << "PROCESO FINALIZADO CORRECTAMENTE\n"
            << "=============================================\n"
            << "Modelo: " << modeloSalida << '\n'
            << "Métricas: "
            << resultados / "metricas_hog_svm.csv"
            << '\n'
            << "Resumen: "
            << resultados / "resumen_modelo_hog_svm.txt"
            << '\n'
            << "Gráficas: " << graficas
            << "\n=============================================\n";

        return 0;
    }
    catch (const cv::Exception& error)
    {
        std::cerr
            << "\nERROR DE OPENCV: "
            << error.what()
            << '\n';

        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "\nERROR: "
            << error.what()
            << '\n';

        return 1;
    }
}