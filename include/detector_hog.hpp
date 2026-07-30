#pragma once

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

#include <string>
#include <vector>

struct TaxiDetectado
{
    cv::Rect caja;
    float score = 0.0F;
};

class DetectorHOG
{
public:
    DetectorHOG();

    bool cargarModelo(
        const std::string& rutaModelo
    );

    [[nodiscard]]
    bool modeloCargado() const;

    [[nodiscard]]
    std::vector<TaxiDetectado> detectar(
        const cv::Mat& frame
    );

private:
    cv::Ptr<cv::ml::SVM> svm_;
    cv::HOGDescriptor hog_;
};