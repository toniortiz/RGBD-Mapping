#include "GvRansac3D3D.h"
#include "Core/Feature.h"
#include "Core/Frame.h"
#include <opengv/point_cloud/PointCloudAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>

using namespace std;

typedef opengv::point_cloud::PointCloudAdapter Adapter;
typedef opengv::sac_problems::point_cloud::PointCloudSacProblem ProblemICP;
typedef opengv::sac::Ransac<ProblemICP> RANSAC;

GvRansac3D3D::GvRansac3D3D(FramePtr prevFrame, FramePtr curFrame, const vector<cv::DMatch>& matches)
    : _inlierThreshold(0.07)
    , _probability(0.99)
    , _maxIterations(1000)
    , _refine(true)
{
    _points1.reserve(matches.size());
    _points2.reserve(matches.size());
    _matches.reserve(matches.size());

    for (const auto& m : matches) {
        const Vec3& p1 = prevFrame->_features[m.queryIdx]->_Xc;
        const Vec3& p2 = curFrame->_features[m.trainIdx]->_Xc;

        if (isnan(p1.z()) || isnan(p2.z()))
            continue;
        if (p1.z() <= 0 || p2.z() <= 0)
            continue;

        _points1.push_back(p1);
        _points2.push_back(p2);
        _matches.push_back(m);
    }
}

bool GvRansac3D3D::compute()
{
    Adapter adapter(_points2, _points1);
    shared_ptr<ProblemICP> icpproblem_ptr(new ProblemICP(adapter));

    RANSAC ransac;
    ransac.sac_model_ = icpproblem_ptr;
    ransac.probability_ = _probability;
    ransac.max_iterations_ = _maxIterations;
    ransac.threshold_ = _inlierThreshold;

    bool bOK = ransac.computeModel();

    _inliers.clear();
    _T = SE3(Mat44::Identity());

    if (bOK) {
        opengv::transformation_t T = ransac.model_coefficients_;
        _inliers.reserve(ransac.inliers_.size());

        if (_refine) {
            opengv::transformation_t nlT;
            icpproblem_ptr->optimizeModelCoefficients(ransac.inliers_, T, nlT);
            T = nlT;
        }

        _T = SE3(T.leftCols(3), T.rightCols(1));

        for (size_t i = 0; i < ransac.inliers_.size(); ++i)
            _inliers.push_back(_matches[ransac.inliers_.at(i)]);
    }

    return bOK;
}
