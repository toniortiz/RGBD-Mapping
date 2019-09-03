#include "Tracking.h"
#include "BA/BA.h"
#include "Core/Frame.h"
#include "Core/PinholeCamera.h"
#include "Features/Extractor.h"
#include "Features/Matcher.h"
#include "ICP/Gicp.h"

using namespace std;

Tracking::Tracking(ExtractorPtr extractor, CameraPtr camera)
    : _extractor(extractor)
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
        break;

    case LOST:
        // not implemented yet
        break;
    }

    _prevFrame = _curFrame;
    _prevT = _curT;
    return _curFrame->getPose();
}

void Tracking::firstFrame()
{
    _curFrame->setPose(SE3(Mat44::Identity()));
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

    Matcher::drawMatches(_prevFrame, _curFrame, inliers);
    _state = OK;
}
