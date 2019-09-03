#ifndef FRAME_H
#define FRAME_H

#include "System/common.h"
#include <mutex>
#include <opencv2/opencv.hpp>
#include <vector>
#include <pcl/registration/registration.h>

class Frame {
public:
    SMART_POINTER_TYPEDEFS(Frame);

public:
    Frame();

    Frame(const cv::Mat& imBGR, const cv::Mat& imDepth, const double& timeStamp, ExtractorPtr extractor, CameraPtr cam);

    void extract();

    void setPose(const SE3& Tcw);
    SE3 getPose();
    SE3 getPoseInverse();

    void drawMatchedPoints();

    double getDepth(const Vec2& xi);

    // Dense point cloud operations
    void createPointCloud(int res);
    void computeNormals(double radius);
    void downsample(float leaf);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    // Feature extractor
    ExtractorPtr _extractor;

    // Frame timestamp.
    double _timestamp;

    CameraPtr _camera;

    // Number of KeyPoints.
    size_t _N;

    // Features extracted
    std::vector<cv::KeyPoint> _keys;
    cv::Mat _descriptors;
    std::vector<FeaturePtr> _features;

    // Current and Next Frame id.
    static int _nextId;
    int _id;

    // Scale pyramid info.
    int _scaleLevels;
    double _scaleFactor;
    double _logScaleFactor;
    std::vector<double> _scaleFactors;
    std::vector<double> _invScaleFactors;
    std::vector<double> _levelSigma2;
    std::vector<double> _invLevelSigma2;

    // Undistorted Image Bounds (computed once).
    static double _minX;
    static double _maxX;
    static double _minY;
    static double _maxY;

    static bool _initialComputations;

    cv::Mat _colorIm;
    cv::Mat _grayIm;
    cv::Mat _depthIm;

    PointCloudColor::Ptr _pointCloud;
    PointCloudColorNormal::Ptr _pointCloudNormals;

private:
    // Camera pose.
    SE3 _Tcw;
    SE3 _Twc;
};

#endif // FRAME_H
