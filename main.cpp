#include "Features/Extractor.h"
#include "IO/Dataset.h"
#include "IO/DatasetTUM.h"
#include "System/Tracking.h"
#include <iomanip>
#include <iostream>

using namespace std;

int main()
{
    Dataset::Ptr ds(new DatasetTUM());
    ds->open("/home/antonio/Documents/M.C.C/Tesis/Dataset/TUM/rgbd_dataset_freiburg1_room/");
    ds->print(cout);

    Extractor::Ptr extractor(new Extractor(Extractor::SVO, Extractor::BRIEF, Extractor::NORMAL));

    Tracking tracker(extractor, ds->camera());

    ofstream f("CameraTrajectory.txt");
    f << fixed;

    for (size_t i = 0; i < ds->size(); ++i) {
        auto [imgs, ts] = ds->getData(i);
        SE3 Tcw = tracker.track(imgs.first, imgs.second, ts);

        SE3 Twc = Tcw.inverse();
        Quat q = Twc.unit_quaternion();
        Vec3 t = Twc.translation();
        f << setprecision(6) << ts << setprecision(7) << " " << t.x() << " " << t.y() << " " << t.z()
          << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }

    f.close();

    return 0;
}
