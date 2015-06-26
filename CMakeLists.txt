CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
# set (CASACORE_ROOT_DIR "/home/lindroos/sw/")
set (CASACORE_ROOT_DIR "/astro/casacore")
PROJECT(vesta)
ADD_EXECUTABLE(vesta vesta_clib/main.cpp vesta_clib/problem_setup.cpp vesta_clib/GaussianCostFunctionCircular.cpp vesta_clib/GaussianCostFunctionCircularAndPointSource.cpp vesta_clib/GaussianCostFunction.cpp vesta_clib/DiskAndDeltaCost.cpp vesta_clib/DiskCost.cpp vesta_clib/PointSourceCostFunction.cpp)

FIND_PACKAGE(Ceres REQUIRED PATHS "/home/lindroos/sw/")
INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(vesta ${CERES_LIBRARIES})


FIND_PACKAGE(CasaCore REQUIRED)
INCLUDE_DIRECTORIES(${CASACORE_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(vesta ${CASACORE_LIBRARIES})

SET(STACKER_LIBRARIES)
list(APPEND STACKER_LIBRARIES /home/lindroos/scripts/stacker/stacker_clib/libstacker.so)
INCLUDE_DIRECTORIES(/home/lindroos/scripts/stacker/stacker_clib)
TARGET_LINK_LIBRARIES(vesta ${STACKER_LIBRARIES})
#TARGET_LINK_LIBRAR
