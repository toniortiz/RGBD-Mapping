#ifndef DETECTORADJUSTER_H
#define DETECTORADJUSTER_H

#include "Extractor.h"
#include <opencv2/features2d.hpp>
#include <string>

class DetectorAdjuster : public cv::Feature2D {
public:
    // Initial values are for SURF detector
    DetectorAdjuster(const Extractor::eAlgorithm& detector, double initialThresh = 200.f,
        double minThresh = 2, double maxThresh = 10000, double increaseFactor = 1.3,
        double decreaseFactor = 0.7);

    virtual void tooFew(int minv, int n_detected);
    virtual void tooMany(int maxv, int n_detected);
    virtual bool good() const;

    virtual cv::Ptr<DetectorAdjuster> clone() const;

    void setIncreaseFactor(double new_factor);
    void setDecreaseFactor(double new_factor);

    virtual void detect(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask = cv::noArray());

protected:
    double _thresh, _minThresh, _maxThresh;
    double _increaseFactor, _decreaseFactor;
    const Extractor::eAlgorithm _detectorAlgorithm;
};
#endif // DETECTORADJUSTER_H
