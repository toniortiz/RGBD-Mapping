#ifndef COMMON_H
#define COMMON_H

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <memory>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sophus/se3.hpp>

#define SMART_POINTER_TYPEDEFS(T)         \
    typedef std::unique_ptr<T> UniquePtr; \
    typedef std::shared_ptr<T> Ptr;       \
    typedef std::shared_ptr<const T> ConstPtr

#define INFO_STREAM(x) std::cout << "[INFO] " << x << std::endl;
#define WARNING_STREAM(x) std::cout << "\033[33m[WARN] " << x << "\033[0m" << std::endl;
#define ERROR_STREAM(x) std::cout << "\033[31m[ERROR] " << x << "\033[0m" << std::endl;

typedef Eigen::Vector3d Vec3;
typedef Eigen::Vector2d Vec2;
typedef Eigen::Vector4d Vec4;
typedef Eigen::Matrix3d Mat33;
typedef Eigen::Matrix4d Mat44;
typedef Eigen::VectorXd VecX;
typedef Eigen::Matrix<double, 6, 6> Mat66;
typedef Eigen::Matrix<double, 3, 4> Mat34;
typedef Eigen::Matrix2d Mat22;
typedef Eigen::Quaterniond Quat;
typedef Eigen::Isometry3d Isom3;
typedef Sophus::SE3d SE3;

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Vec3)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Vec2)

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudColor;
typedef pcl::PointCloud<pcl::PointXYZRGBNormal> PointCloudColorNormal;

class Frame;
typedef std::shared_ptr<Frame> FramePtr;

class PinholeCamera;
typedef std::shared_ptr<PinholeCamera> CameraPtr;

class Extractor;
typedef std::shared_ptr<Extractor> ExtractorPtr;

class Feature;
typedef std::shared_ptr<Feature> FeaturePtr;

/*
 * [1] Henry et.al., RGB-D Mapping: Using Kinect-Style Depth Cameras for Dense 3D Modeling of Indoor Environments, 2012.
*/

#endif // COMMON_H
