#include "Tracking.h"
#include "BA/BA.h"
#include "Core/Frame.h"
#include "Core/Map.h"
#include "Core/PinholeCamera.h"
#include "Features/Extractor.h"
#include "Features/Matcher.h"
#include "ICP/Gicp.h"

using namespace std;

Tracking::Tracking(MapPtr map, ExtractorPtr extractor, CameraPtr camera)
    : _map(map)
    , _extractor(extractor)
    , _camera(camera)
    , _state(FIRST_FRAME)
    , _prevT(Mat44::Identity())
    , _gamma(10)
    , _phi(300)
{
}

SE3 Tracking::track(const cv::Mat& color, const cv::Mat& depth, const double timestamp)
{
    _curFrame.reset(new Frame(color, depth, timestamp, _extractor, _camera));

    switch (_state) {
    case FIRST_FRAME:
        firstFrame();
        break;

    case OK:
        twoStageICP();

        if (needNewKeyFrame())
            createNewKeyFrame();
        break;

    case LOST:
        // not implemented yet
        break;
    }

    _prevFrame = _curFrame;
    _prevT = _curT;
    return _curFrame->getPose();
}

cv::Mat Tracking::getImageMatches()
{
    unique_lock<mutex> lock(_mutexImg);
    return _imgMatches.clone();
}

void Tracking::firstFrame()
{
    _curFrame->setPose(SE3(Mat44::Identity()));
    _map->addKeyFrame(_curFrame);
    _prevKF = _curFrame;
    _state = OK;
}

void Tracking::twoStageICP()
{
    vector<cv::DMatch> matches;
    matches.reserve(_curFrame->_N);
    Matcher matcher(0.9);
    matcher.knnMatch(_prevFrame, _curFrame, matches);

    vector<cv::DMatch> inliers;
    SE3 T;
    BA::twoFrameBA(_prevFrame, _curFrame, matches, inliers, T);

    if (inliers.size() < _gamma) {
        T = _prevT;
        inliers.clear();
    }

    if (inliers.size() >= _phi) {
        _curT = T;
        _curFrame->setPose(T * _prevFrame->getPose());
    } else {
        _prevFrame->createPointCloud(8);
        _curFrame->createPointCloud(8);
        Gicp icp(_prevFrame, _curFrame, T);
        icp.icp.setMaximumIterations(7);
        icp.icp.setMaxCorrespondenceDistance(0.03);
        icp.compute();

        T = icp._T;
        _curT = T;
        _curFrame->setPose(T * _prevFrame->getPose());
    }
    _state = OK;

    unique_lock<mutex> lock(_mutexImg);
    _imgMatches = Matcher::getImageMatches(_prevFrame, _curFrame, inliers);
}

double tnorm(const SE3& T)
{
    Vec3 t = T.translation();
    return t.norm();
}

double rnorm(const SE3& T)
{
    Mat33 R = T.rotationMatrix();
    return acos(0.5 * (R(0, 0) + R(1, 1) + R(2, 2) - 1.0));
}

bool Tracking::needNewKeyFrame()
{
    // New keyframes are added when the accumulated motion since the previous
    // keyframe exceeds either 10° in rotation or 20 cm in translation
    static const double mint = 0.20; // m
    static const double minr = 0.1745; // rad

    SE3 delta = _curFrame->getPoseInverse() * _prevKF->getPose();
    bool c1 = tnorm(delta) > mint;
    bool c2 = rnorm(delta) > minr;

    if (c1 || c2)
        return true;
    else
        return false;
}

void Tracking::createNewKeyFrame()
{
    _map->addKeyFrame(_curFrame);
    _prevKF = _curFrame;
}
