#include "Viewer.h"
#include "Core/Map.h"
#include "System/Tracking.h"
#include "Viewer/DenseMapDrawer.h"
#include <mutex>
#include <opencv2/core.hpp>
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;

Viewer::Viewer(TrackingPtr pTracking, MapPtr pMap, DenseMapPtr denseMap, const bool& start)
    : _tracker(pTracking)
    , _map(pMap)
    , _denseMap(denseMap)
    , _finishRequested(false)
    , _finished(true)
    , _meanTrackTime(0)
{
    _imageWidth = 640;
    _imageHeight = 480;

    _viewpointX = 0;
    _viewpointY = -0.7f;
    _viewpointZ = -1.8f;
    _viewpointF = 500;

    if (start)
        _thread = thread(&Viewer::run, this);
}

void Viewer::start()
{
    if (!_thread.joinable())
        _thread = thread(&Viewer::run, this);
}

void Viewer::run()
{
    _finished = false;

    pangolin::CreateWindowAndBind("Map Viewer", 1024, 768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuShowKeyFrames("menu.KeyFrames", true, true);
    pangolin::Var<bool> menuShowGraph("menu.Graph", true, true);
    pangolin::Var<int> menuNodes("menu.Nodes:", 0);
    pangolin::Var<double> menuTime("menu.Track time:", 0);

    pangolin::Var<bool> menuShowDenseMap("menu.Dense Map", true, true);
    pangolin::Var<bool> menuSaveOctoMap("menu.Save Octomap", false, false);
    pangolin::Var<int> menuSize("menu.Size", 0);
    pangolin::Var<double> menuMemory("menu.Memory (Mb)", 0);
    pangolin::Var<bool> menuOptimizeMap("menu.Optimize Map", false, false);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, _viewpointF, _viewpointF, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(_viewpointX, _viewpointY, _viewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::View& d_features = pangolin::Display("imgFeatures")
                                     .SetAspect(/*1024.0*/ 2048.0 / 768.0);
    pangolin::GlTexture texFeatures(_imageWidth * 2, _imageHeight, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

    pangolin::CreateDisplay()
        .SetBounds(0.0, 0.3f, pangolin::Attach::Pix(175), 1.0)
        .SetLayout(pangolin::LayoutEqual)
        .AddDisplay(d_features);

    while (1) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        if (menuShowDenseMap)
            _denseMap->render();
        if (menuSaveOctoMap) {
            _denseMap->save("./OctoMap.ot");
            menuSaveOctoMap.Reset();
        }
        if (menuOptimizeMap) {
            _map->informNewBigChange();
            menuOptimizeMap.Reset();
        }
        menuSize = _denseMap->size();
        menuMemory = double(_denseMap->memory()) / 1000000.0; // Mb

        menuNodes = _map->KeyFramesInMap();

        {
            unique_lock<mutex> lock(_mutexStatistics);
            menuTime = _meanTrackTime;
        }

        cv::Mat feats = _tracker->getImageMatches();
        if (!feats.empty()) {
            texFeatures.Upload(feats.data, GL_BGR, GL_UNSIGNED_BYTE);
            d_features.Activate();
            glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
            texFeatures.RenderToViewportFlipY();
        }

        pangolin::FinishFrame();

        if (checkFinish())
            break;
    }

    setFinish();
}

void Viewer::requestFinish()
{
    unique_lock<mutex> lock(_mutexFinish);
    _finishRequested = true;
}

bool Viewer::checkFinish()
{
    unique_lock<mutex> lock(_mutexFinish);
    return _finishRequested;
}

void Viewer::setFinish()
{
    unique_lock<mutex> lock(_mutexFinish);
    _finished = true;
}

bool Viewer::isFinished()
{
    unique_lock<mutex> lock(_mutexFinish);
    return _finished;
}

void Viewer::join()
{
    if (_thread.joinable()) {
        _thread.join();
        cout << "Viewer thread JOINED" << endl;
    }
}

void Viewer::setMeanTime(const double& time)
{
    unique_lock<mutex> lock(_mutexStatistics);
    _meanTrackTime = time;
}
