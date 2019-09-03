#ifndef GICP_H
#define GICP_H

#include "System/common.h"
#include "pcl/registration/gicp.h"

class Gicp {
public:
    Gicp(FramePtr prevFrame, FramePtr curFrame, const SE3& T);

    bool compute();

    PointCloudColor::Ptr _srcCloud, _dstCloud;
    SE3 _T;
    pcl::GeneralizedIterativeClosestPoint<PointCloudColor::PointType, PointCloudColor::PointType> icp;
};

#endif // GICP_H
