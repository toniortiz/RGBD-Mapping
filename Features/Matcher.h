#ifndef MATCHER_H
#define MATCHER_H

#include "System/common.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <pcl/correspondence.h>
#include <set>
#include <vector>

class Matcher {
public:
    SMART_POINTER_TYPEDEFS(Matcher);

    Matcher(double nnratio = 0.6);

    // Computes the distance between two descriptors
    static double descriptorDistance(const cv::Mat& a, const cv::Mat& b);

    int knnMatch(const FramePtr F1, FramePtr F2, std::vector<cv::DMatch>& vMatches);

    static void drawMatches(const FramePtr F1, const FramePtr F2, const std::vector<cv::DMatch>& m12, const int delay = 1);
    static void drawMatches(const FramePtr F1, const FramePtr F2, const pcl::CorrespondencesPtr corrs, const int delay = 1);
    static cv::Mat getImageMatches(const FramePtr F1, const FramePtr F2, const std::vector<cv::DMatch>& m12);

public:
    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;

protected:
    double _NNratio;

    cv::Ptr<cv::DescriptorMatcher> _matcher;
};

#endif // ORBMATCHER_H
