#include <opencv2/opencv.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

constexpr int ANCHO_MAXIMO_PANTALLA = 1200;
constexpr int ALTO_MAXIMO_PANTALLA = 750;

bool esImagenValida(const fs::path& ruta)
{
    std::string extension = ruta.extension().string();

    std::transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](unsigned char caracter)
        {
            return static_cast<char>(std::tolower(caracter));
        }
    );

    return extension == ".jpg"
        || extension == ".jpeg"
        || extension == ".png"
        || extension == ".webp";
}

std::vector<fs::path> obtenerImagenes(const fs::path& carpeta)
{
    std::vector<fs::path> imagenes;

    for (const auto& entrada : fs::directory_iterator(carpeta))
    {
        if (!entrada.is_regular_file())
        {
            continue;
        }

        if (esImagenValida(entrada.path()))
        {
            imagenes.push_back(entrada.path());
        }
    }

    std::sort(imagenes.begin(), imagenes.end());

    return imagenes;
}

std::set<std::string> cargarImagenesProcesadas(
    const fs::path& archivoCsv
)
{
    std::set<std::string> procesadas;

    if (!fs::exists(archivoCsv))
    {
        return procesadas;
    }

    std::ifstream archivo(archivoCsv);

    if (!archivo.is_open())
    {
        return procesadas;
    }

    std::string linea;

    // Ignorar encabezado.
    std::getline(archivo, linea);

    while (std::getline(archivo, linea))
    {
        if (linea.empty())
        {
            continue;
        }

        std::stringstream flujo(linea);
        std::string nombreOriginal;

        std::getline(flujo, nombreOriginal, ',');

        if (!nombreOriginal.empty())
        {
            procesadas.insert(nombreOriginal);
        }
    }

    return procesadas;
}

cv::Mat ajustarParaPantalla(
    const cv::Mat& imagenOriginal,
    double& escala
)
{
    const double escalaAncho =
        static_cast<double>(ANCHO_MAXIMO_PANTALLA)
        / imagenOriginal.cols;

    const double escalaAlto =
        static_cast<double>(ALTO_MAXIMO_PANTALLA)
        / imagenOriginal.rows;

    escala = std::min(
        {
            1.0,
            escalaAncho,
            escalaAlto
        }
    );

    if (escala >= 1.0)
    {
        return imagenOriginal.clone();
    }

    cv::Mat imagenRedimensionada;

    cv::resize(
        imagenOriginal,
        imagenRedimensionada,
        cv::Size(),
        escala,
        escala,
        cv::INTER_AREA
    );

    return imagenRedimensionada;
}

cv::Rect convertirRoiAOriginal(
    const cv::Rect& roiPantalla,
    double escala,
    const cv::Size& tamanoOriginal
)
{
    int x = static_cast<int>(roiPantalla.x / escala);
    int y = static_cast<int>(roiPantalla.y / escala);
    int ancho = static_cast<int>(roiPantalla.width / escala);
    int alto = static_cast<int>(roiPantalla.height / escala);

    x = std::clamp(x, 0, tamanoOriginal.width - 1);
    y = std::clamp(y, 0, tamanoOriginal.height - 1);

    ancho = std::min(
        ancho,
        tamanoOriginal.width - x
    );

    alto = std::min(
        alto,
        tamanoOriginal.height - y
    );

    return cv::Rect(x, y, ancho, alto);
}

std::string generarNombreRecorte(
    const std::string& conjunto,
    int indice
)
{
    std::ostringstream nombre;

    nombre
        << "taxi_recorte_"
        << conjunto
        << "_"
        << std::setw(4)
        << std::setfill('0')
        << indice
        << ".jpg";

    return nombre.str();
}

void mostrarInstrucciones(
    cv::Mat& imagen,
    int actual,
    int total
)
{
    std::string texto1 =
        "Imagen " + std::to_string(actual)
        + "/" + std::to_string(total);

    std::string texto2 =
        "Dibuje el cuadro alrededor del taxi y presione ENTER";

    cv::rectangle(
        imagen,
        cv::Point(0, 0),
        cv::Point(imagen.cols, 70),
        cv::Scalar(0, 0, 0),
        cv::FILLED
    );

    cv::putText(
        imagen,
        texto1,
        cv::Point(15, 28),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(255, 255, 255),
        2
    );

    cv::putText(
        imagen,
        texto2,
        cv::Point(15, 57),
        cv::FONT_HERSHEY_SIMPLEX,
        0.58,
        cv::Scalar(255, 255, 255),
        1
    );
}

int main(int argc, char* argv[])
{
    if (argc != 5)
    {
        std::cerr
            << "Uso:\n"
            << argv[0]
            << " <carpeta_entrada>"
            << " <carpeta_salida>"
            << " <archivo_csv>"
            << " <conjunto>\n\n"
            << "Ejemplo:\n"
            << argv[0]
            << " dataset/positivos/train"
            << " dataset/positivos/recortes/train"
            << " resultados/anotaciones/train.csv"
            << " train\n";

        return 1;
    }

    const fs::path carpetaEntrada = argv[1];
    const fs::path carpetaSalida = argv[2];
    const fs::path archivoCsv = argv[3];
    const std::string conjunto = argv[4];

    if (!fs::exists(carpetaEntrada))
    {
        std::cerr
            << "ERROR: no existe la carpeta de entrada:\n"
            << carpetaEntrada
            << '\n';

        return 1;
    }

    fs::create_directories(carpetaSalida);

    if (archivoCsv.has_parent_path())
    {
        fs::create_directories(
            archivoCsv.parent_path()
        );
    }

    const std::vector<fs::path> imagenes =
        obtenerImagenes(carpetaEntrada);

    if (imagenes.empty())
    {
        std::cerr
            << "ERROR: no se encontraron imágenes en:\n"
            << carpetaEntrada
            << '\n';

        return 1;
    }

    const bool csvExiste = fs::exists(archivoCsv);

    std::ofstream csv(
        archivoCsv,
        std::ios::app
    );

    if (!csv.is_open())
    {
        std::cerr
            << "ERROR: no se pudo abrir el CSV:\n"
            << archivoCsv
            << '\n';

        return 1;
    }

    if (!csvExiste || fs::file_size(archivoCsv) == 0)
    {
        csv
            << "nombre_original,nombre_recorte,"
            << "conjunto,x,y,ancho,alto,"
            << "ruta_original,ruta_recorte\n";
    }

    std::set<std::string> procesadas =
        cargarImagenesProcesadas(archivoCsv);

    int indiceRecorte =
        static_cast<int>(procesadas.size()) + 1;

    int posicion = 0;

    std::cout
        << "=============================================\n"
        << "ETIQUETADOR DE TAXIS AMARILLOS\n"
        << "=============================================\n"
        << "Conjunto:             " << conjunto << '\n'
        << "Imágenes encontradas: " << imagenes.size() << '\n'
        << "Ya procesadas:        " << procesadas.size() << '\n'
        << "\nControles:\n"
        << "  ENTER o ESPACIO: aceptar selección\n"
        << "  C: cancelar selección actual\n"
        << "  ESC: no seleccionar\n"
        << "\nDespués de seleccionar:\n"
        << "  G: guardar recorte\n"
        << "  R: repetir selección\n"
        << "  N: omitir imagen\n"
        << "  Q: guardar progreso y salir\n\n";

    for (const fs::path& rutaImagen : imagenes)
    {
        ++posicion;

        const std::string nombreOriginal =
            rutaImagen.filename().string();

        if (procesadas.contains(nombreOriginal))
        {
            continue;
        }

        cv::Mat imagenOriginal =
            cv::imread(rutaImagen.string());

        if (imagenOriginal.empty())
        {
            std::cerr
                << "No se pudo leer: "
                << rutaImagen
                << '\n';

            continue;
        }

        bool imagenTerminada = false;

        while (!imagenTerminada)
        {
            double escala = 1.0;

            cv::Mat imagenPantalla =
                ajustarParaPantalla(
                    imagenOriginal,
                    escala
                );

            mostrarInstrucciones(
                imagenPantalla,
                posicion,
                static_cast<int>(imagenes.size())
            );

            const cv::Rect roiPantalla =
                cv::selectROI(
                    "Seleccionar taxi amarillo",
                    imagenPantalla,
                    false,
                    false
                );

            if (
                roiPantalla.width <= 0
                || roiPantalla.height <= 0
            )
            {
                std::cout
                    << "No se seleccionó ROI para "
                    << nombreOriginal
                    << ". Presione N para omitir, "
                    << "R para intentar otra vez o Q para salir.\n";

                const int tecla =
                    cv::waitKey(0) & 0xFF;

                if (tecla == 'q' || tecla == 'Q')
                {
                    cv::destroyAllWindows();

                    std::cout
                        << "Progreso guardado. Programa finalizado.\n";

                    return 0;
                }

                if (tecla == 'n' || tecla == 'N')
                {
                    imagenTerminada = true;
                }

                continue;
            }

            const cv::Rect roiOriginal =
                convertirRoiAOriginal(
                    roiPantalla,
                    escala,
                    imagenOriginal.size()
                );

            if (
                roiOriginal.width < 30
                || roiOriginal.height < 30
            )
            {
                std::cout
                    << "La selección es demasiado pequeña. "
                    << "Repita el cuadro.\n";

                continue;
            }

            cv::Mat recorte =
                imagenOriginal(roiOriginal).clone();

            cv::Mat vistaPrevia = recorte.clone();

            cv::putText(
                vistaPrevia,
                "G: guardar | R: repetir | N: omitir | Q: salir",
                cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                cv::Scalar(0, 0, 255),
                2
            );

            cv::imshow(
                "Vista previa del recorte",
                vistaPrevia
            );

            const int tecla =
                cv::waitKey(0) & 0xFF;

            cv::destroyWindow(
                "Vista previa del recorte"
            );

            if (tecla == 'q' || tecla == 'Q')
            {
                cv::destroyAllWindows();

                std::cout
                    << "Progreso guardado. Programa finalizado.\n";

                return 0;
            }

            if (tecla == 'r' || tecla == 'R')
            {
                continue;
            }

            if (tecla == 'n' || tecla == 'N')
            {
                imagenTerminada = true;
                continue;
            }

            if (tecla != 'g' && tecla != 'G')
            {
                std::cout
                    << "Tecla no reconocida. "
                    << "Repitiendo selección.\n";

                continue;
            }

            const std::string nombreRecorte =
                generarNombreRecorte(
                    conjunto,
                    indiceRecorte
                );

            const fs::path rutaRecorte =
                carpetaSalida / nombreRecorte;

            const std::vector<int> parametrosJpeg = {
                cv::IMWRITE_JPEG_QUALITY,
                95
            };

            const bool guardada =
                cv::imwrite(
                    rutaRecorte.string(),
                    recorte,
                    parametrosJpeg
                );

            if (!guardada)
            {
                std::cerr
                    << "ERROR: no se pudo guardar:\n"
                    << rutaRecorte
                    << '\n';

                return 1;
            }

            csv
                << nombreOriginal << ','
                << nombreRecorte << ','
                << conjunto << ','
                << roiOriginal.x << ','
                << roiOriginal.y << ','
                << roiOriginal.width << ','
                << roiOriginal.height << ','
                << rutaImagen.string() << ','
                << rutaRecorte.string()
                << '\n';

            csv.flush();

            procesadas.insert(nombreOriginal);

            std::cout
                << "Guardado: "
                << nombreRecorte
                << " | ROI: "
                << roiOriginal.x << ", "
                << roiOriginal.y << ", "
                << roiOriginal.width << ", "
                << roiOriginal.height
                << '\n';

            ++indiceRecorte;
            imagenTerminada = true;
        }
    }

    cv::destroyAllWindows();

    std::cout
        << "\n=============================================\n"
        << "ETIQUETADO FINALIZADO\n"
        << "=============================================\n"
        << "Conjunto:          " << conjunto << '\n'
        << "Recortes guardados: "
        << procesadas.size() << '\n'
        << "Archivo CSV:       " << archivoCsv << '\n';

    return 0;
}