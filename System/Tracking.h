#ifndef TRACKING_H
#define TRACKING_H

#include "System/common.h"
#include <opencv2/core.hpp>

class Tracking {
public:
    SMART_POINTER_TYPEDEFS(Tracking);

    enum eState {
        FIRST_FRAME = 0,
        OK,
        LOST
    };

    Tracking(ExtractorPtr extractor, CameraPtr camera);

    SE3 track(const cv::Mat& color, const cv::Mat& depth, const double timestamp);

private:
    void firstFrame();
    void twoStageICP();

    ExtractorPtr _extractor;
    CameraPtr _camera;
    eState _state;

    SE3 _curT;
    SE3 _prevT;

    FramePtr _prevFrame, _curFrame;

    // Parameters
    unsigned int _gamma;
    unsigned int _phi;
};

#endif // TRACKING_H
