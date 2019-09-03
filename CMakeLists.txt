cmake_minimum_required(VERSION 2.8)
project(RGBD-ICP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall   -O3 -march=native -pthread")

#------------------------------------------
#                    PCL
#------------------------------------------
find_package(PCL REQUIRED COMPONENTS common registration segmentation)
add_definitions(${PCL_DEFINITIONS})
include_directories(${PCL_INCLUDE_DIRS})
link_libraries(${PCL_LIBRARY_DIRS})

#------------------------------------------
#                    Sophus
#------------------------------------------
find_package(Sophus)
include_directories(${Sophus_INCLUDE_DIRS})

#------------------------------------------
#                    OpenCV
#------------------------------------------
find_package( OpenCV REQUIRED )
include_directories(${OPENCV_INCLUDE_DIRS})

#------------------------------------------
#                    Eigen
#------------------------------------------
find_package(Eigen3 3.0 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

#------------------------------------------
#                    fast
#------------------------------------------
FIND_PACKAGE(fast REQUIRED)
include_directories(${fast_INCLUDE_DIRS})

#------------------------------------------
#                    opengv
#------------------------------------------
find_package(opengv REQUIRED)

#------------------------------------------
#                    G2O
#------------------------------------------
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package( G2O REQUIRED )
include_directories(${G2O_INCLUDE_DIR})
find_package( Cholmod )
include_directories(${CHOLMOD_INCLUDE_DIR})
find_package(CSparse)
include_directories(${CSPARSE_INCLUDE_DIR})
list(APPEND G2O_LIBS
  g2o_cli g2o_ext_freeglut_minimal g2o_simulator
  g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d
  g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only
  g2o_types_sba g2o_types_slam3d g2o_csparse_extension
  g2o_opengl_helper g2o_solver_dense g2o_stuff
  g2o_types_sclam2d g2o_parser g2o_solver_pcg
  g2o_types_data g2o_types_sim3
)

#------------------------------------------
#                    Octomap
#------------------------------------------
find_package(octomap)
include_directories(${OCTOMAP_INCLUDE_DIRS})

#------------------------------------------
#                    Pangolin
#------------------------------------------
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})


#---------------- Project -----------------
include_directories(${PROJECT_SOURCE_DIR})

add_library(${PROJECT_NAME} SHARED
    Core/PinholeCamera.cpp
    Core/Feature.cpp
    Core/Frame.cpp
    Core/Map.cpp

    Ransac/Ransac3D3D.cpp
    Ransac/GvRansac3D3D.cpp
    Ransac/Ransac3D2D.cpp
    Ransac/MRansac3D3D.cpp

    BA/BA.cpp

    ICP/Gicp.cpp

    Features/ORBextractor.cpp
    Features/Matcher.cpp
    Features/DetectorAdjuster.cpp
    Features/Extractor.cpp
    Features/StatefulFeatureDetector.cpp
    Features/SVOextractor.cpp
    Features/VideoDynamicAdaptedFeatureDetector.cpp
    Features/VideoGridAdaptedFeatureDetector.cpp

    System/Tracking.cpp

    IO/Dataset.cpp
    IO/DatasetCORBS.cpp
    IO/DatasetICL.cpp
    IO/DatasetTUM.cpp
    IO/DatasetMicrosoft.cpp

    Utils/Random.cpp

    Viewer/Viewer.cpp
    Viewer/DenseMapDrawer.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${PCL_LIBRARIES}
    ${Sophus_LIBRARIES}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${fast_LIBRARIES}
    opengv
    ${G2O_LIBS}
    ${CHOLMOD_LIBRARIES}
    ${CSPARSE_LIBRARY}
    ${OCTOMAP_LIBRARIES}
    ${Pangolin_LIBRARIES}
)

add_executable(main main.cpp)
target_link_libraries(main ${PROJECT_NAME})
