#ifndef RANSAC3D3D_H
#define RANSAC3D3D_H

#include "System/common.h"
#include <opencv2/features2d.hpp>
#include <pcl/correspondence.h>

// Minimize the point-to-point squared distance error using RANSAC pipeline
// Eq. (1) in Henry et.al. 2012.
class Ransac3D3D {
public:
    Ransac3D3D(FramePtr prevFrame, FramePtr curFrame, const std::vector<cv::DMatch>& matches);

    void compute();

    void toDMatch(std::vector<cv::DMatch>& matches);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    PointCloud::Ptr _srcCloud, _tgtCloud;
    pcl::Correspondences _corrs, _inlierCorrs;
    std::vector<cv::DMatch> _matches;

    // Parameters
    double _inlierThreshold;
    unsigned int _iterations;
    bool _refine;

    // Estimation
    SE3 _T;
};

#endif // RANSAC3D3D_H
