#include "Ransac3D3D.h"
#include "Core/Feature.h"
#include "Core/Frame.h"
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

using namespace std;

Ransac3D3D::Ransac3D3D(FramePtr prevFrame, FramePtr curFrame, const vector<cv::DMatch>& matches)
    : _inlierThreshold(0.06)
    , _iterations(500)
    , _refine(true)
{
    _srcCloud.reset(new PointCloud);
    _tgtCloud.reset(new PointCloud);
    _srcCloud->reserve(matches.size());
    _tgtCloud->reserve(matches.size());
    _corrs.reserve(matches.size());
    _matches.reserve(matches.size());

    size_t idx = 0;
    for (const auto& m : matches) {
        const Vec3& src = prevFrame->_features[m.queryIdx]->_Xc;
        const Vec3& tgt = curFrame->_features[m.trainIdx]->_Xc;

        if (isnan(src.z()) || isnan(tgt.z()))
            continue;
        if (src.z() <= 0 || tgt.z() <= 0)
            continue;

        _srcCloud->push_back(PointCloud::PointType(src.x(), src.y(), src.z()));
        _tgtCloud->push_back(PointCloud::PointType(tgt.x(), tgt.y(), tgt.z()));

        _corrs.emplace_back(idx, idx, m.distance);
        _matches.push_back(m);
        idx++;
    }

    _inlierCorrs.reserve(_corrs.size());
}

void Ransac3D3D::compute()
{
    pcl::registration::CorrespondenceRejectorSampleConsensus<PointCloud::PointType> sac;
    sac.setInlierThreshold(_inlierThreshold);
    sac.setMaximumIterations(_iterations);
    sac.setRefineModel(_refine);

    sac.setInputSource(_srcCloud);
    sac.setInputTarget(_tgtCloud);
    sac.getRemainingCorrespondences(_corrs, _inlierCorrs);

    _T = SE3(sac.getBestTransformation().cast<double>());
}

void Ransac3D3D::toDMatch(vector<cv::DMatch>& matches)
{
    matches.resize(_inlierCorrs.size());
    for (size_t i = 0; i < _inlierCorrs.size(); ++i) {
        int idx = _inlierCorrs[i].index_query;
        matches[i] = _matches[idx];
    }
}
