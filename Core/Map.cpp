#include "Map.h"
#include "Frame.h"
#include <mutex>

using namespace std;

Map::Map()
    : _maxKFid(0)
    , _bigChangeIdx(0)
{
}

void Map::addKeyFrame(FramePtr pKF)
{
    unique_lock<mutex> lock(_mutexMap);
    _keyframes.insert({ pKF->_id, pKF });
    if (pKF->_id > _maxKFid)
        _maxKFid = pKF->_id;
}

void Map::eraseKeyFrame(FramePtr pKF)
{
    unique_lock<mutex> lock(_mutexMap);
    _keyframes.erase(pKF->_id);
}

void Map::eraseKeyFrame(const int& id)
{
    unique_lock<mutex> lock(_mutexMap);
    _keyframes.erase(id);
}

void Map::informNewBigChange()
{
    unique_lock<mutex> lock(_mutexMap);
    _bigChangeIdx++;
}

int Map::getLastBigChangeIdx()
{
    unique_lock<mutex> lock(_mutexMap);
    return _bigChangeIdx;
}

vector<FramePtr> Map::getAllKeyFrames()
{
    unique_lock<mutex> lock(_mutexMap);
    vector<FramePtr> kfs;
    kfs.reserve(_keyframes.size());
    for (auto& [id, pKF] : _keyframes)
        kfs.push_back(pKF);
    return kfs;
}

size_t Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(_mutexMap);
    return _keyframes.size();
}

int Map::getMaxKFid()
{
    unique_lock<mutex> lock(_mutexMap);
    return _maxKFid;
}

void Map::clear()
{
    _keyframes.clear();
    _maxKFid = 0;
}
