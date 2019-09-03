#include "Frame.h"
#include "Feature.h"
#include "Features/Extractor.h"
#include "PinholeCamera.h"
#include <opencv2/core/eigen.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <thread>

using namespace std;

int Frame::_nextId = 0;
bool Frame::_initialComputations = true;
double Frame::_minX, Frame::_minY, Frame::_maxX, Frame::_maxY;

Frame::Frame() {}

Frame::Frame(const cv::Mat& imBGR, const cv::Mat& imDepth, const double& timeStamp, ExtractorPtr extractor, CameraPtr cam)
    : _extractor(extractor)
    , _timestamp(timeStamp)
    , _camera(cam)
    , _colorIm(imBGR)
    , _pointCloud(nullptr)
    , _pointCloudNormals(nullptr)
{
    _id = _nextId++;

    cvtColor(_colorIm, _grayIm, CV_BGR2GRAY);
    imDepth.convertTo(_depthIm, CV_64F, _camera->depthFactor());

    // Scale Level Info
    _scaleLevels = _extractor->getLevels();
    _scaleFactor = _extractor->getScaleFactor();
    _logScaleFactor = log(_scaleFactor);
    _scaleFactors = _extractor->getScaleFactors();
    _invScaleFactors = _extractor->getInverseScaleFactors();
    _levelSigma2 = _extractor->getScaleSigmaSquares();
    _invLevelSigma2 = _extractor->getInverseScaleSigmaSquares();

    // Feature extraction
    extract();

    // This is done only for the first Frame (or after a change in the calibration)
    if (_initialComputations) {
        _camera->undistortBounds(_minX, _maxX, _minY, _maxY);
        _initialComputations = false;
    }
}

void Frame::extract()
{
    _extractor->detectAndCompute(_grayIm, cv::Mat(), _keys, _descriptors);

    _N = _keys.size();

    _features.resize(_N);
    for (size_t i = 0; i < _N; ++i) {
        FeaturePtr ftr(new Feature(this, Vec2(_keys[i].pt.x, _keys[i].pt.y),
            _keys[i].octave, _keys[i].angle, _descriptors.row(i), i));
        _features[i] = ftr;
    }
}

void Frame::setPose(const SE3& Tcw)
{
    _Tcw = Tcw;
    _Twc = Tcw.inverse();
}

SE3 Frame::getPose()
{
    return _Tcw;
}

SE3 Frame::getPoseInverse()
{
    return _Twc;
}

double Frame::getDepth(const Vec2& xi)
{
    return _depthIm.at<double>(xi.y(), xi.x());
}

void Frame::createPointCloud(int res)
{
    if (_pointCloud)
        return;

    _pointCloud.reset(new PointCloudColor());

    for (int m = 0; m < _depthIm.rows; m += res) {
        for (int n = 0; n < _depthIm.cols; n += res) {
            const double z = _depthIm.at<double>(m, n);

            if (z <= 0)
                continue;

            Vec3 xc = _camera->backproject(Vec2(n, m), z);
            PointCloudColor::PointType p;
            p.x = float(xc.x());
            p.y = float(xc.y());
            p.z = float(xc.z());

            p.b = _colorIm.data[m * _colorIm.step + n * _colorIm.channels() + 0]; // blue
            p.g = _colorIm.data[m * _colorIm.step + n * _colorIm.channels() + 1]; // green
            p.r = _colorIm.data[m * _colorIm.step + n * _colorIm.channels() + 2]; // red

            _pointCloud->push_back(p);
        }
    }
}

void Frame::computeNormals(double radius)
{
    if (!_pointCloud)
        return;
    if (_pointCloudNormals)
        return;

    _pointCloudNormals.reset(new PointCloudColorNormal());

    pcl::NormalEstimation<PointCloudColor::PointType, PointCloudColorNormal::PointType> ne;
    ne.setInputCloud(_pointCloud);

    pcl::search::KdTree<PointCloudColor::PointType>::Ptr tree(new pcl::search::KdTree<PointCloudColor::PointType>());
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(radius);
    ne.compute(*_pointCloudNormals);
}

void Frame::downsample(float leaf)
{
    if (!_pointCloud)
        return;

    pcl::VoxelGrid<PointCloudColor::PointType> voxel;
    voxel.setLeafSize(leaf, leaf, leaf);

    voxel.setInputCloud(_pointCloud);
    voxel.filter(*_pointCloud);
}

void Frame::drawMatchedPoints()
{
    if (_keys.empty())
        return;

    cv::Mat out;
    cv::cvtColor(_grayIm, out, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < _N; i++) {
        FeaturePtr ftr = _features[i];
        if (ftr->isValid()) {
            const cv::KeyPoint& kp = _keys[i];
            cv::circle(out, kp.pt, 4 * (kp.octave + 1), cv::Scalar(0, 255, 0), 1);
        }
    }

    cv::imshow("Features", out);
    cv::waitKey(1);
}
