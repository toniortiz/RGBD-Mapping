#ifndef RANSAC3D2D_H
#define RANSAC3D2D_H

#include "System/common.h"
#include <opencv2/core.hpp>

class Ransac3D2D {
public:
    enum Algorithm { SOLVEPNP_ITERATIVE = 0,
        SOLVEPNP_EPNP = 1, //!< EPnP: Efficient Perspective-n-Point Camera Pose Estimation
        SOLVEPNP_P3P = 2, //!< Complete Solution Classification for the Perspective-Three-Point Problem
        SOLVEPNP_DLS = 3, //!< A Direct Least-Squares (DLS) Method for PnP
        SOLVEPNP_UPNP = 4, //!< Exhaustive Linearization for Robust Camera Pose and Focal Length Estimation
        SOLVEPNP_AP3P = 5, //!< An Efficient Algebraic Solution to the Perspective-Three-Point Problem
        SOLVEPNP_MAX_COUNT //!< Used for count
    };

    Ransac3D2D(FramePtr prevFrame, FramePtr curFrame, const std::vector<cv::DMatch>& matches);

    bool compute(std::vector<cv::DMatch>& inliers, SE3& T);

    // Parameters
    int _iterations;
    float _reprojectionError;
    double _confidence;
    Algorithm _algorithm;

private:
    std::vector<cv::DMatch> _matches;
    std::vector<cv::Point3d> _objs;
    std::vector<cv::Point2d> _imgs;
    cv::Mat _K;
};

#endif // RANSAC3D2D_H
