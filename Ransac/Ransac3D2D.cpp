#include "Ransac3D2D.h"
#include "Core/Feature.h"
#include "Core/Frame.h"
#include "Core/PinholeCamera.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;

Ransac3D2D::Ransac3D2D(FramePtr prevFrame, FramePtr curFrame, const std::vector<cv::DMatch>& matches)
    : _iterations(500)
    , _reprojectionError(2.0f)
    , _confidence(0.85)
    , _algorithm(SOLVEPNP_ITERATIVE)
{
    _objs.reserve(matches.size());
    _imgs.reserve(matches.size());
    _matches.reserve(matches.size());

    for (const auto& m : matches) {
        const Vec3& obj = prevFrame->_features[m.queryIdx]->_Xc;
        const Vec2& img = curFrame->_features[m.trainIdx]->_uXi;

        if (obj.isZero())
            continue;

        _matches.push_back(m);
        _objs.push_back(cv::Point3d(obj.x(), obj.y(), obj.z()));
        _imgs.push_back(cv::Point2d(img.x(), img.y()));
    }

    _K = prevFrame->_camera->_cvK.clone();
}

bool Ransac3D2D::compute(vector<cv::DMatch>& inliers, SE3& T)
{
    if (_objs.size() < 5)
        return false;

    cv::Mat rvec(3, 1, CV_64F), tvec(3, 1, CV_64F), inls;
    bool b = false;

    try {
        b = cv::solvePnPRansac(_objs, _imgs, _K, cv::Mat(), rvec, tvec, false,
            _iterations, _reprojectionError, _confidence, inls, _algorithm);

        cv::Mat R(3, 3, CV_64F);
        cv::Rodrigues(rvec, R);

        Mat33 rot;
        Vec3 tras;
        cv::cv2eigen(R, rot);
        cv::cv2eigen(tvec, tras);
        SE3 cT(rot, tras);

        if (b && cT.matrix().norm() < 1000.0) {
            inliers.reserve(_matches.size());
            for (int i = 0; i < inls.rows; ++i) {
                int n = inls.at<int>(i);
                inliers.push_back(_matches[n]);
            }

            T = cT;
        } else {
            b = false;
            inliers.clear();
            WARNING_STREAM("Ransac3D2D::compute -> Fail");
        }
    } catch (cv::Exception e) {
        ERROR_STREAM("Ransac3D2D::compute -> "s + e.what());
        b = false;
        inliers.clear();
    }

    return b;
}
