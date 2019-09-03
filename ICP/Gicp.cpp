#include "Gicp.h"
#include "Core/Frame.h"
#include <iostream>

using namespace std;

Gicp::Gicp(FramePtr prevFrame, FramePtr curFrame, const SE3& T)
{
    _srcCloud = prevFrame->_pointCloud;
    _dstCloud = curFrame->_pointCloud;
    _T = T;

    icp.setMaximumIterations(10);
    icp.setEuclideanFitnessEpsilon(1);
    icp.setMaxCorrespondenceDistance(0.07);
    icp.setTransformationEpsilon(1e-9);
}

bool Gicp::compute()
{
    icp.setInputSource(_srcCloud);
    icp.setInputTarget(_dstCloud);

    PointCloudColor aligned;
    icp.align(aligned, _T.matrix().cast<float>());

    _T = SE3(icp.getFinalTransformation().cast<double>());

    return icp.hasConverged();
}
