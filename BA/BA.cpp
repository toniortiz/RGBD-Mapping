#include "BA.h"
#include "Core/Feature.h"
#include "Core/Frame.h"
#include "Core/PinholeCamera.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/structure_only/structure_only_solver.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;

void BA::twoFrameBA(FramePtr prevFrame, FramePtr curFrame, const vector<cv::DMatch>& matches, vector<cv::DMatch>& inlierMatches, SE3& T)
{
    static const double deltaMono = sqrt(5.991);
    static const double deltaStereo = sqrt(7.815);
    double fx = prevFrame->_camera->fx();
    double fy = prevFrame->_camera->fy();
    double cx = prevFrame->_camera->cx();
    double cy = prevFrame->_camera->cy();
    double bf = prevFrame->_camera->baseLineFx();

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    //    g2o::CameraParameters* cam_params = new g2o::CameraParameters(prevFrame->_camera->fx(),
    //        Vec2(prevFrame->_camera->cx(), prevFrame->_camera->cy()), prevFrame->_camera->baseLine());
    //    cam_params->setId(0);
    //    optimizer.addParameter(cam_params);

    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(g2o::SE3Quat(Mat33::Identity(), Vec3::Zero()));
    vSE3->setFixed(false);
    vSE3->setId(0);
    optimizer.addVertex(vSE3);

    vector<bool> inliers(matches.size(), true);
    int good = 0;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> edgesMono;
    edgesMono.reserve(matches.size());

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> edgesStereo;
    edgesStereo.reserve(matches.size());

    for (size_t i = 0; i < matches.size(); i++) {
        const Vec3& prevXc = prevFrame->_features[matches[i].queryIdx]->_Xc;
        if (prevXc.isZero()) {
            inliers[i] = false;
            continue;
        }

        good++;
        FeaturePtr curFtr = curFrame->_features[matches[i].trainIdx];

        if (curFtr->_right <= 0) {
            g2o::EdgeSE3ProjectXYZOnlyPose* edge = new g2o::EdgeSE3ProjectXYZOnlyPose();

            edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            edge->setMeasurement(curFtr->_uXi);
            double invSigma2 = curFrame->_invLevelSigma2[curFtr->_level];
            edge->setInformation(Mat22::Identity() * invSigma2);

            edge->fx = fx;
            edge->fy = fy;
            edge->cx = cx;
            edge->cy = cy;
            edge->Xw = prevXc;

            g2o::RobustKernelHuber* kernel = new g2o::RobustKernelHuber();
            kernel->setDelta(deltaMono);
            edge->setRobustKernel(kernel);

            optimizer.addEdge(edge);
            edge->setId(i);
            edgesMono.push_back(edge);
        } else {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* edge = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

            edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            Vec3 meas(curFtr->_uXi.x(), curFtr->_uXi.y(), curFtr->_right);
            edge->setMeasurement(meas);
            double invSigma2 = curFrame->_invLevelSigma2[curFtr->_level];
            edge->setInformation(Mat33::Identity() * invSigma2);

            edge->fx = fx;
            edge->fy = fy;
            edge->cx = cx;
            edge->cy = cy;
            edge->bf = bf;
            edge->Xw = prevXc;

            g2o::RobustKernelHuber* kernel = new g2o::RobustKernelHuber();
            kernel->setDelta(deltaStereo);
            edge->setRobustKernel(kernel);

            optimizer.addEdge(edge);
            edge->setId(i);
            edgesStereo.push_back(edge);
        }
    }

    for (size_t it = 0; it < 4; ++it) {
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);

        for (g2o::EdgeSE3ProjectXYZOnlyPose* e : edgesMono) {
            if (!inliers[e->id()])
                e->computeError();

            if (e->chi2() > 5.991) {
                inliers[e->id()] = false;
                e->setLevel(1);
                good--;
            } else {
                inliers[e->id()] = true;
                e->setLevel(0);
            }

            if (it == 2)
                e->setRobustKernel(nullptr);
        }

        for (g2o::EdgeStereoSE3ProjectXYZOnlyPose* e : edgesStereo) {
            if (!inliers[e->id()])
                e->computeError();

            if (e->chi2() > 7.815) {
                inliers[e->id()] = false;
                e->setLevel(1);
                good--;
            } else {
                inliers[e->id()] = true;
                e->setLevel(0);
            }

            if (it == 2)
                e->setRobustKernel(nullptr);
        }

        if (good < 5)
            break;
    }

    for (size_t i = 0; i < inliers.size(); ++i) {
        if (inliers[i])
            inlierMatches.push_back(matches[i]);
    }

    g2o::VertexSE3Expmap* recov = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat q = recov->estimate();

    T = SE3(q.rotation(), q.translation());
}
