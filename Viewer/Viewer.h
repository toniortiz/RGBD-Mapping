#ifndef VIEWER_H
#define VIEWER_H

#include "System/common.h"
#include <mutex>
#include <thread>
#include <vector>

class Viewer {
public:
    SMART_POINTER_TYPEDEFS(Viewer);

    Viewer(TrackingPtr pTracking, MapPtr pMap, DenseMapPtr denseMap, const bool& start = true);

    void start();

    void run();

    void requestFinish();

    bool isFinished();

    void release();

    void join();

    void setMeanTime(const double& time);

private:
    bool stop();

    TrackingPtr _tracker;
    MapPtr _map;
    DenseMapPtr _denseMap;

    float _imageWidth, _imageHeight;

    float _viewpointX, _viewpointY, _viewpointZ, _viewpointF;

    bool checkFinish();
    void setFinish();
    bool _finishRequested;
    bool _finished;
    std::mutex _mutexFinish;

    double _meanTrackTime;
    std::mutex _mutexStatistics;

    std::thread _thread;
};

#endif // VIEWER_H
