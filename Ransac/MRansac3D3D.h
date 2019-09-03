#ifndef MRANSAC3D3D_H
#define MRANSAC3D3D_H

#include "System/common.h"
#include <opencv2/core.hpp>

class MRansac3D3D {
public:
    MRansac3D3D(FramePtr prevFrame, FramePtr curFrame, const std::vector<cv::DMatch>& matches);

    bool compute(std::vector<cv::DMatch>& inliers, SE3& T);

    // Parameters
    int _iterations;
    uint _minInlierTh;
    float _maxMahalanobisDistance;
    uint _sampleSize;

private:
    std::vector<cv::DMatch> sampleMatches();
    Eigen::Matrix4f computeHypothesis(const std::vector<cv::DMatch>& vMatches, bool& valid);
    double computeError(const Eigen::Matrix4f& transformation4f, std::vector<cv::DMatch>& vInlierMatches);
    double errorFunction2(const Eigen::Vector4f& x1, const Eigen::Vector4f& x2, const Eigen::Matrix4d& transformation);
    double depthCovariance(double depth);
    double depthStdDev(double depth);

    std::vector<FeaturePtr> _prevFeats, _curFeats;
    std::vector<cv::DMatch> _matches;
};

#endif // MRANSAC3D3D_H
