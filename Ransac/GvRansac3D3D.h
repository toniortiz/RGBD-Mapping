#ifndef GVRANSAC3D3D_H
#define GVRANSAC3D3D_H

#include "System/common.h"
#include <opencv2/features2d.hpp>
#include <vector>

// Minimize the point-to-point squared distance error using RANSAC pipeline
// Eq. (1) in Henry et.al. 2012.
class GvRansac3D3D {
public:
    SMART_POINTER_TYPEDEFS(GvRansac3D3D);

    GvRansac3D3D(FramePtr prevFrame, FramePtr curFrame, const std::vector<cv::DMatch>& matches);

    bool compute();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    std::vector<Vec3> _points1, _points2;
    std::vector<cv::DMatch> _matches;

    // Ransac parameters
    double _inlierThreshold;
    double _probability;
    int _maxIterations;
    bool _refine;

    // Result
    std::vector<cv::DMatch> _inliers;
    SE3 _T;
};

#endif // GVRANSAC3D3D_H
