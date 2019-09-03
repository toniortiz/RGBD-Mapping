#ifndef TRACKING_H
#define TRACKING_H

#include "System/common.h"
#include <mutex>
#include <opencv2/core.hpp>

class Tracking {
public:
    SMART_POINTER_TYPEDEFS(Tracking);

    enum eState {
        FIRST_FRAME = 0,
        OK,
        LOST
    };

    Tracking(MapPtr map, ExtractorPtr extractor, CameraPtr camera);

    SE3 track(const cv::Mat& color, const cv::Mat& depth, const double timestamp);

    cv::Mat getImageMatches();

private:
    void firstFrame();
    void twoStageICP();

    bool needNewKeyFrame();
    void createNewKeyFrame();

    MapPtr _map;
    ExtractorPtr _extractor;
    CameraPtr _camera;
    eState _state;

    SE3 _curT;
    SE3 _prevT;

    FramePtr _prevFrame, _curFrame;
    FramePtr _prevKF;

    cv::Mat _imgMatches;
    std::mutex _mutexImg;

    // Parameters
    unsigned int _gamma;
    unsigned int _phi;
};

#endif // TRACKING_H
