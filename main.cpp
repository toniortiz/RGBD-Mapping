#include "Core/Map.h"
#include "Features/Extractor.h"
#include "IO/Dataset.h"
#include "IO/DatasetTUM.h"
#include "System/Tracking.h"
#include "Viewer/DenseMapDrawer.h"
#include "Viewer/Viewer.h"
#include <iomanip>
#include <iostream>
#include <pangolin/pangolin.h>

using namespace std;

int main()
{
    Dataset::Ptr ds(new DatasetTUM());
    ds->open("/home/antonio/Documents/M.C.C/Tesis/Dataset/TUM/rgbd_dataset_freiburg1_room/");
    ds->print(cout);

    Extractor::Ptr extractor(new Extractor(Extractor::SVO, Extractor::BRIEF, Extractor::NORMAL));
    Map::Ptr map(new Map());
    DenseMapDrawer::Ptr denseMap(new DenseMapDrawer(map));
    Tracking::Ptr tracker(new Tracking(map, extractor, ds->camera()));
    Viewer::Ptr viewer(new Viewer(tracker, map, denseMap));

    ofstream f("CameraTrajectory.txt");
    f << fixed;

    cv::TickMeter tm;
    for (size_t i = 0; i < ds->size(); ++i) {
        auto [imgs, ts] = ds->getData(i);

        tm.start();
        SE3 Tcw = tracker->track(imgs.first, imgs.second, ts);
        tm.stop();

        viewer->setMeanTime(tm.getTimeSec() / tm.getCounter());

        SE3 Twc = Tcw.inverse();
        Quat q = Twc.unit_quaternion();
        Vec3 t = Twc.translation();
        f << setprecision(6) << ts << setprecision(7) << " " << t.x() << " " << t.y() << " " << t.z()
          << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }

    f.close();

    cout << "Press enter to close" << endl;
    getchar();

    denseMap->requestFinish();
    while (!denseMap->isFinished())
        usleep(5000);
    denseMap->join();

    viewer->requestFinish();
    while (!viewer->isFinished())
        usleep(5000);
    pangolin::BindToContext("Map Viewer");
    viewer->join();

    return 0;
}
