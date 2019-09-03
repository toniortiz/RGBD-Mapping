#include "Matcher.h"
#include "Core/Feature.h"
#include "Core/Frame.h"
#include "Core/PinholeCamera.h"
#include "Extractor.h"
#include <limits.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <stdint-gcc.h>

using namespace std;

const int Matcher::TH_HIGH = 100;
const int Matcher::TH_LOW = 50;
const int Matcher::HISTO_LENGTH = 30;

Matcher::Matcher(double nnratio)
    : _NNratio(nnratio)
{
    _matcher = cv::BFMatcher::create(Extractor::_norm);
}

int Matcher::knnMatch(const FramePtr F1, FramePtr F2, vector<cv::DMatch>& vMatches)
{
    vector<vector<cv::DMatch>> matchesKnn;
    set<int> trainIdxs;

    _matcher->knnMatch(F1->_descriptors, F2->_descriptors, matchesKnn, 2);

    for (size_t i = 0; i < matchesKnn.size(); i++) {
        cv::DMatch& m1 = matchesKnn[i][0];
        cv::DMatch& m2 = matchesKnn[i][1];

        if (m1.distance < _NNratio * m2.distance) {
            if (trainIdxs.count(m1.trainIdx) > 0)
                continue;

            //            size_t i1 = static_cast<size_t>(m1.queryIdx);
            //            size_t i2 = static_cast<size_t>(m1.trainIdx);

            //            if (F1->_features[i1]->_Xw.isZero())
            //                continue;
            //            if (!F2->_features[i2]->isValid())
            //                continue;

            //            F2->_features[i2]->_Xw = F1->_features[i1]->_Xw;
            trainIdxs.insert(m1.trainIdx);
            vMatches.push_back(m1);
        }
    }

    return int(vMatches.size());
}

double Matcher::descriptorDistance(const cv::Mat& a, const cv::Mat& b)
{
    return cv::norm(a, b, Extractor::_norm);
}

void Matcher::drawMatches(const FramePtr F1, const FramePtr F2, const vector<cv::DMatch>& m12, const int delay)
{
    cv::Mat out;

    cv::drawMatches(F1->_colorIm, F1->_keys, F2->_colorIm, F2->_keys, m12, out, cv::Scalar::all(-1),
        cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("Matches", out);
    cv::waitKey(delay);
}

void Matcher::drawMatches(const FramePtr F1, const FramePtr F2, const pcl::CorrespondencesPtr corrs, const int delay)
{
    CameraPtr cam = F1->_camera;

    vector<cv::KeyPoint> keys1, keys2;
    keys1.reserve(corrs->size());
    keys2.reserve(corrs->size());
    vector<cv::DMatch> matches;
    matches.reserve(corrs->size());
    cout << corrs->size() << endl;

    int idx = 0;
    for (size_t i = 0; i < corrs->size(); ++i) {
        int im = corrs->at(i).index_match; //F2 tgt
        int iq = corrs->at(i).index_query; // F1 src

        auto p1c = F1->_pointCloud->points[iq];
        Vec2 p1i = cam->project(Vec3(p1c.x, p1c.y, p1c.z));

        auto p2c = F2->_pointCloud->points[im];
        Vec2 p2i = cam->project(Vec3(p2c.x, p2c.y, p2c.z));

        if (p1i.x() < F1->_minX || p1i.x() > F1->_maxX || p1i.y() < F1->_minY || p1i.y() > F1->_maxY)
            continue;
        if (p2i.x() < F2->_minX || p2i.x() > F2->_maxX || p2i.y() < F2->_minY || p2i.y() > F2->_maxY)
            continue;

        cv::KeyPoint kp1;
        kp1.pt = cv::Point2f(p1i.x(), p1i.y());
        keys1.push_back(kp1);

        cv::KeyPoint kp2;
        kp2.pt = cv::Point2f(p2i.x(), p2i.y());
        keys2.push_back(kp2);

        cv::DMatch m(idx, idx, corrs->at(i).distance);
        matches.push_back(m);
        idx++;
    }

    cv::Mat out;

    cv::drawMatches(F1->_colorIm, keys1, F2->_colorIm, keys2, matches, out, cv::Scalar::all(-1),
        cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("Matches", out);
    cv::waitKey(delay);
}

cv::Mat Matcher::getImageMatches(const FramePtr F1, const FramePtr F2, const std::vector<cv::DMatch>& m12)
{
    cv::Mat out;
    cv::drawMatches(F1->_colorIm, F1->_keys, F2->_colorIm, F2->_keys, m12, out, cv::Scalar::all(-1),
        cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return out;
}
