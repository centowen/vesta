CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
# set (CASACORE_ROOT_DIR "/home/lindroos/sw/")
SET(CMAKE_CXX_FLAGS "-DCASACORE_VERSION_2")
set (CASACORE_ROOT_DIR "/astro/casacore")
PROJECT(vesta)

FIND_PACKAGE(Glog REQUIRED)
INCLUDE_DIRECTORIES(${GLOG_INCLUDE_DIRS} PATHS "/home/lindroos/sw/")

FIND_PACKAGE(Ceres REQUIRED PATHS "/home/lindroos/sw/")
INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})


FIND_PACKAGE(CasaCore REQUIRED)
INCLUDE_DIRECTORIES(${CASACORE_INCLUDE_DIRS})

FIND_PACKAGE(CUDA)

SET(STACKER_LIBRARIES)
ADD_DEFINITIONS(-DCASACORE_VERSION_2)
list(APPEND STACKER_LIBRARIES /home/lindroos/prog/stacker/stacker_clib/libstacker.so)
INCLUDE_DIRECTORIES(/home/lindroos/prog/stacker/stacker_clib)

option(ENABLE_CUDA "Enable the cuda build" ON)

if(ENABLE_CUDA)
	CUDA_ADD_EXECUTABLE(vesta vesta_clib/main.cpp vesta_clib/problem_setup.cpp vesta_clib/CommonCuda.cu vesta_clib/GaussianCostFunctionCircular.cpp vesta_clib/GaussianCostFunctionCircularAndPointSource.cpp vesta_clib/GaussianCostFunction.cpp vesta_clib/DiskAndDeltaCost.cpp vesta_clib/DiskCost.cpp vesta_clib/PointSourceCostFunction.cpp)
	TARGET_LINK_LIBRARIES(vesta ${CASACORE_LIBRARIES})
	TARGET_LINK_LIBRARIES(vesta ${CERES_LIBRARIES})
	TARGET_LINK_LIBRARIES(vesta ${STACKER_LIBRARIES})
	ADD_DEFINITIONS(-DENABLE_CUDA)
else()
	ADD_EXECUTABLE(vesta vesta_clib/main.cpp vesta_clib/problem_setup.cpp vesta_clib/GaussianCostFunctionCircular.cpp vesta_clib/GaussianCostFunctionCircularAndPointSource.cpp vesta_clib/GaussianCostFunction.cpp vesta_clib/DiskAndDeltaCost.cpp vesta_clib/DiskCost.cpp vesta_clib/PointSourceCostFunction.cpp)
	ADD_LIBRARY(chi2_scan SHARED vesta_clib/chi2.cpp vesta_clib/GaussianCostFunctionCircular.cpp vesta_clib/GaussianCostFunctionCircularAndPointSource.cpp vesta_clib/GaussianCostFunction.cpp vesta_clib/DiskAndDeltaCost.cpp vesta_clib/DiskCost.cpp vesta_clib/PointSourceCostFunction.cpp)

	TARGET_LINK_LIBRARIES(vesta ${CASACORE_LIBRARIES})
	TARGET_LINK_LIBRARIES(vesta ${CERES_LIBRARIES})
	TARGET_LINK_LIBRARIES(vesta ${STACKER_LIBRARIES})

	TARGET_LINK_LIBRARIES(chi2_scan ${CASACORE_LIBRARIES})
	TARGET_LINK_LIBRARIES(chi2_scan ${CERES_LIBRARIES})
	TARGET_LINK_LIBRARIES(chi2_scan ${STACKER_LIBRARIES})
endif()

