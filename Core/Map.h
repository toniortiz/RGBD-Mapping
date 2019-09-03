#ifndef MAP_H
#define MAP_H

#include "System/common.h"
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

class Map {
public:
    SMART_POINTER_TYPEDEFS(Map);

    Map();

    void addKeyFrame(FramePtr pKF);
    void eraseKeyFrame(FramePtr pKF);
    void eraseKeyFrame(const int& id);

    void informNewBigChange();
    int getLastBigChangeIdx();

    std::vector<FramePtr> getAllKeyFrames();

    size_t KeyFramesInMap();

    int getMaxKFid();

    void clear();

    std::mutex _mutexMapUpdate;

protected:
    std::unordered_map<int, FramePtr> _keyframes;

    int _maxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int _bigChangeIdx;

    std::mutex _mutexMap;
};

#endif // MAP_H
