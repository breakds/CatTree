# created from template OpenCV included c++ project
#   by BreakDS 
#   Mar 23 2012 @ Univeristy of Wisconsin-Madison
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# Ask CMake to search for current c/c++ compiler
set(CMAKE_CXX_COMPILER_INIT "c++")


# Project Description Starts Here
PROJECT(CatTree)


# System & Project Information
MESSAGE(STATUS "Project Name:           ${PROJECT_NAME}")
MESSAGE(STATUS "Project Source Dir:     ${PROJECT_SOURCE_DIR}")
MESSAGE(STATUS "Kernel:                 ${CMAKE_SYSTEM}")
EXECUTE_PROCESS(COMMAND hostname OUTPUT_VARIABLE HOSTNAME)
MESSAGE(STATUS "Hostname:               ${HOSTNAME}")
MESSAGE(STATUS "Build Type:             ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "GCC:                    ${CMAKE_CXX_COMPILER}")

# option (BUILD_UNITTEST "build executables in purpose of unittest." ON)


# ================================================================================
# Include Directory Structure for breakds@emperor
if ("${HOSTNAME}" MATCHES "emperor.cs.wisc.edu")
  include_directories(
    "/scratch/usr/local/include"
    "~/pf/projects"
    ".")
  # OpenCV Library
  add_library( OPENCV SHARED IMPORTED)
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/scratch/usr/local/lib/libopencv_core.so" )
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/scratch/usr/local/lib/libopencv_highgui.so" )
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/scratch/usr/local/lib/libopencv_features2d.so" )
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/scratch/usr/local/lib/libopencv_objdetect.so" )
  set(OPENCV "OPENCV")

  # CUDA SDK 5.0
  set(CUDA_TOOLKIT_ROOT_DIR "/s/cuda-5.0.35/") # Force it to search for CUDA 4.2.9
  find_package("CUDA" 5.0 REQUIRED)
  set(CUDA_NVCC_FLAGS "-arch=sm_20")
endif ("${HOSTNAME}" MATCHES "emperor.cs.wisc.edu")


# ================================================================================
# Include Directory Structure for breakds@beggar
if ("${HOSTNAME}" MATCHES "beggar.cs.wisc.edu")
  include_directories(
    "/scratch/usr/local/include"
    "~/pf/projects"
    "/scratch/opt/include"
    ".")
  # OpenCV Library
  add_library( OPENCV SHARED IMPORTED)
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/scratch/opt/lib/libopencv_core.so" )
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/scratch/opt/lib/libopencv_highgui.so" )
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/scratch/opt/lib/libopencv_features2d.so" )
  set_property( TARGET OPENCV PROPERTY
    IMPORTED_LOCATION "/scratch/opt/lib/libopencv_objdetect.so" )
  set(OPENCV "OPENCV")

  # CUDA SDK 5.0
  set(CUDA_TOOLKIT_ROOT_DIR "/s/cuda-5.0.35/") # Force it to search for CUDA 4.2.9
  find_package("CUDA" 5.0 REQUIRED)
  set(CUDA_NVCC_FLAGS "-arch=sm_30")
endif ("${HOSTNAME}" MATCHES "beggar.cs.wisc.edu")




# ================================================================================
# Include Directory Structure for break@zhang-submit.chtc.wisc.edu
if ("${HOSTNAME}" MATCHES "zhang-submit.chtc.wisc.edu")
  include_directories(
    "~/pf/projects"
    ".")
  # OpenCV Library
  set( CMAKE_EXE_LINKER_FLAGS "-lopencv_core -lopencv_imgproc -lopencv_highgui" )
  set(OPENCV "")
  
  # CUDA SDK 5.0
  find_package("CUDA" 5.0 REQUIRED)
  set(CUDA_NVCC_FLAGS "-arch=sm_30 --compiler-bindir /usr/bin")
endif ("${HOSTNAME}" MATCHES "zhang-submit.chtc.wisc.edu")

# ================================================================================
# Include Directory Structure for break@zhang-submit.chtc.wisc.edu
if ("${HOSTNAME}" MATCHES "zhang-c005.chtc.wisc.edu")
  include_directories(
    "~/pf/projects"
    ".")
  # OpenCV Library
  set( CMAKE_EXE_LINKER_FLAGS "-lopencv_core -lopencv_imgproc -lopencv_highgui" )
  set(OPENCV "")
  
endif ("${HOSTNAME}" MATCHES "zhang-c005.chtc.wisc.edu")



# ================================================================================
# For breakds@AzraelWaker
if ("${HOSTNAME}" MATCHES "AzraelWaker")
  include_directories(
    "~/pf/projects"
    ".")
  
  # Linker Flags
  set( CMAKE_EXE_LINKER_FLAGS "-lopencv_core -lopencv_imgproc -lopencv_highgui" )
  set(OPENCV "")

  # CUDA SDK 5.0
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda/")
  find_package("CUDA" 5.0 REQUIRED)
  set(CUDA_NVCC_FLAGS "-arch=sm_30")
endif ("${HOSTNAME}" MATCHES "AzraelWaker")

# For breakds@AzraelSaber
if ("${HOSTNAME}" MATCHES "AzraelSaber")
  include_directories(
    "~/pf/projects"
    ".")
  
  # Linker Flags
  set( CMAKE_EXE_LINKER_FLAGS "-lopencv_core -lopencv_imgproc -lopencv_highgui" )
  set(OPENCV "")

endif ("${HOSTNAME}" MATCHES "AzraelSaber")


if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)



# Whether build as shared libraries by default
# SET(BUILD_SHARED_LIBS ON)



# ================================================================================
# Compile CUDA related libraries
MESSAGE(STATUS "CUDA VERISON: ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" )
MESSAGE(STATUS "NVCC VERISON: ${CUDA_NVCC_EXECUTABLE}" )




# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
# cuda_add_library(cuda_opt SHARED opt/BP_CUDA.cu)




# ================================================================================
# COMPILATION_FLAGS
# Note -Wno-non-virtual-dtor is for a defect in OpenCV
set(CMAKE_CXX_FLAGS "${OPENCV_FLAGS} -Wall -Wextra -Wno-non-virtual-dtor -std=c++0x")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fopenmp -D_GLIBCXX_DEBUG")
set(CMAKE_CXX_FLAGS_GPROF "-O1 -pg")


# caltech101 experiment
# ADD_EXECUTABLE(setup experiment/setup.cpp)
# ADD_EXECUTABLE(plant experiment/plant.cpp)
# ADD_EXECUTABLE(clustring experiment/psuedoKmean.cpp)
# ADD_EXECUTABLE(prim experiment/prim.cpp)


# ADD_EXECUTABLE(poweriter experiment/poweriter_v2.cpp)
# ADD_EXECUTABLE(exp experiment/exp.cpp)
# ADD_EXECUTABLE(climbing experiment/climbing.cpp)


# TARGET_LINK_LIBRARIES(camvid ${OPENCV})


# TARGET_LINK_LIBRARIES(exp "-static")
# ADD_EXECUTABLE(spiral experiment/spiral.cpp)

# ADD_EXECUTABLE(vis experiment/visualize.cpp)

# experiment on Camvid
# ADD_EXECUTABLE(plant experiment/CamvidPlant.cpp)
# TARGET_LINK_LIBRARIES(plant ${OPENCV})

# ADD_EXECUTABLE(reconstruct experiment/CamvidReconstruction.cpp)
# TARGET_LINK_LIBRARIES(reconstruct ${OPENCV})

ADD_EXECUTABLE(beta experiment/betasys.cpp)
TARGET_LINK_LIBRARIES(beta ${OPENCV})

ADD_EXECUTABLE(plant experiment/betaplant.cpp)
TARGET_LINK_LIBRARIES(plant ${OPENCV})

ADD_EXECUTABLE(multiplant experiment/multiplant.cpp)
TARGET_LINK_LIBRARIES(multiplant ${OPENCV})

# ADD_EXECUTABLE(gen experiment/FeatureGen.cpp)
# TARGET_LINK_LIBRARIES(gen ${OPENCV})










