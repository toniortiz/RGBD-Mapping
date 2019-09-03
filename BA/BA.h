#ifndef BA_H
#define BA_H

#include "System/common.h"
#include <opencv2/features2d.hpp>

class BA {
public:
    static void twoFrameBA(FramePtr prevFrame, FramePtr curFrame, const std::vector<cv::DMatch>& matches, std::vector<cv::DMatch>& inlierMatches, SE3& T);
};

#endif // BA_H
