##
# CMakeList file for TDDFT
##

CMAKE_MINIMUM_REQUIRED(VERSION 3.17)
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)


# Set the name of the project and the target
SET(TARGET_PROJ "invDFT")
SET(TARGET_EXE "invDFT_exe")
SET(TARGET_LIB "invDFT_lib")
SET(TARGET_VERSION "1.0")

# Add -O2 flag when compiled in Release mode
IF("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    add_compile_options("-O2")
ENDIF()

#Assign the project name
PROJECT(${TARGET_PROJ} LANGUAGES CXX)

#GPU enable
IF (WITH_GPU)
    IF ("${GPU_LANG}" STREQUAL "cuda")
        enable_language(CUDA)
    ELSEIF ("${GPU_LANG}" STREQUAL "hip")
        enable_language(HIP)
    ELSE ()
        MESSAGE(FATAL_ERROR "\n"
                "*** Could not find valid GPU_LANG=cuda/hip ***\n\n"
        )
    ENDIF()
ENDIF()



# Define the executable
ADD_EXECUTABLE(${TARGET_EXE} ./src/main.cc)

# Add CMake modules
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)


SET(ENABLE_MPI TRUE)
IF(ENABLE_MPI)
    FIND_PACKAGE(MPI REQUIRED)
    INCLUDE_DIRECTORIES(${MPI_CXX_INCLUDE_PATH})
    #TARGET_LINK_LIBRARIES(MPI_LIBRARIES PUBLIC ${MPI_CXX_LIBRARIES})
    SET(MPI_LIBRARIES ${MPI_CXX_LIBRARIES})
ENDIF()




# Declare all source files the target consists of:
SET(TARGET_SRC ./src/gaussian/gaussianFunctionContainer.cpp
	       ./src/gaussian/gaussianFunctionManager.cpp
	       ./src/slater/SlaterFunctionManager.cpp
	       ./src/BFGSInverseDFTSolver.cpp
	       ./src/InverseDFTSolverFunction.cpp
	       ./src/InverseDFTEngine.cpp
	       ./src/MultiVectorAdjointLinearSolverProblem.cpp
	       ./src/inverseDFTParameters.cpp
	       ./src/TriangulationManagerVxc.cpp
)


#Declare all source files for GPU
IF ("${GPU_LANG}" STREQUAL "cuda")

    SET(DEVICE_SRC
            ./src/MultiVectorAdjointLinearSolverProblem.cpp
	    )

ELSEIF ("${GPU_LANG}" STREQUAL "hip")

    SET(DEVICE_SRC
            ./src/MultiVectorAdjointLinearSolverProblem.cpp)

ENDIF()

#Add GPU language specific properties
IF (WITH_GPU)
    IF ("${GPU_LANG}" STREQUAL "cuda")
        set_source_files_properties(${DEVICE_SRC} PROPERTIES LANGUAGE CUDA)
    ELSEIF ("${GPU_LANG}" STREQUAL "hip")
        set_source_files_properties(${DEVICE_SRC} PROPERTIES LANGUAGE HIP)
        IF ("${GPU_VENDOR}" STREQUAL "amd")
            ADD_DEFINITIONS(-D__HIP_PLATFORM_AMD__)
        ELSEIF ("${GPU_VENDOR}" STREQUAL "nvidia")
            ADD_DEFINITIONS(-D__HIP_PLATFORM_NVIDIA__)
        ENDIF()
    ENDIF()
ENDIF()
IF (WITH_GPU)
    LIST(APPEND TARGET_SRC ${DEVICE_SRC})
ENDIF()





# Add the library
ADD_LIBRARY(${TARGET_LIB} SHARED ${TARGET_SRC})


#-----------------------------Link MPI--------------------
TARGET_LINK_LIBRARIES(${TARGET_LIB} PUBLIC ${MPI_LIBRARIES})


#-----------------------------Link DFTFE--------------------
FIND_LIBRARY(DFTFE_LIBRARY
	NAMES dftfe dftfeReal
        HINTS ${DFTFE_INSTALL_PATH}/build/release/real/
        HINTS ${DFTFE_INSTALL_PATH}/
	NO_DEFAULT_PATH
)

IF(DFTFE_LIBRARY STREQUAL "DFTFE_LIBRARY-NOTFOUND")
    MESSAGE(FATAL_ERROR "-- DFTFE was not found in ${DFTFE_INSTALL_PATH}")
ENDIF()


TARGET_INCLUDE_DIRECTORIES(${TARGET_LIB} PUBLIC "${DFTFE_INCLUDE_PATH}")
TARGET_LINK_LIBRARIES(${TARGET_LIB} PUBLIC ${DFTFE_LIBRARY})

###########################################################################
    ADD_DEFINITIONS(-DUSE_REAL)
    MESSAGE(STATUS "Using real")
###########################################################################

#
#Set GPU compilation flag & link libraries
#
IF (WITH_GPU)
    ADD_DEFINITIONS(-DDFTFE_WITH_DEVICE)

    IF ("${GPU_VENDOR}" STREQUAL "nvidia")
        ADD_DEFINITIONS(-DDFTFE_WITH_DEVICE_NVIDIA)
    ELSEIF ("${GPU_VENDOR}" STREQUAL "amd")
        ADD_DEFINITIONS(-DDFTFE_WITH_DEVICE_AMD)
    ENDIF()


    IF ("${GPU_LANG}" STREQUAL "cuda")
        MESSAGE(STATUS "DFTFE with GPU LANG CUDA")
        ADD_DEFINITIONS(-DDFTFE_WITH_DEVICE_LANG_CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_include_directories(${TARGET_LIB} PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(${TARGET_LIB} PUBLIC CUDA::cudart)
        target_link_libraries(${TARGET_LIB} PUBLIC CUDA::cublas)
    ELSEIF ("${GPU_LANG}" STREQUAL "hip")
        MESSAGE(STATUS "DFTFE with GPU LANG HIP")
        ADD_DEFINITIONS(-DDFTFE_WITH_DEVICE_LANG_HIP)
        find_package(hipblas)
        target_link_libraries(${TARGET_LIB} PUBLIC roc::hipblas)
    ENDIF()
    IF (WITH_GPU_AWARE_MPI)
        ADD_DEFINITIONS(-DDFTFE_WITH_DEVICE_AWARE_MPI)
        MESSAGE(STATUS "DFTFE with GPU Aware MPI")
    ENDIF()
ENDIF()


############################################################################
#
#Dependencies
#

FIND_PACKAGE(deal.II 9.5.1 REQUIRED
        HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)
IF(NOT ${deal.II_FOUND})
    MESSAGE(FATAL_ERROR "\n"
            "*** Could not locate deal.II. ***\n\n"
            "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
            "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

#
# Are all deal.II dependencies fullfilled?
# DEAL_II_WITH_64BIT_INDICES is a requirement because of
# the choice of the parallel storage data-structure for
# wavefunctions in DFT-FE. The maximum global index of this
# parallel storage data-structure is equal to
# total number of dofs times total number of wavefunctions,
# which can cross the 32 bit integer limits even for a moderate
# problem size.
IF(NOT DEAL_II_WITH_LAPACK OR
        NOT DEAL_II_WITH_P4EST OR
        NOT DEAL_II_WITH_64BIT_INDICES OR
        NOT DEAL_II_WITH_MPI)
    MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_LAPACK = ON
    DEAL_II_WITH_P4EST = ON
    DEAL_II_WITH_64BIT_INDICES = ON
    DEAL_II_WITH_MPI = ON
which is required."
    )
ENDIF()


#DEAL_II_INITIALIZE_CACHED_VARIABLES()


TARGET_INCLUDE_DIRECTORIES(${TARGET_LIB} PUBLIC ${DEAL_II_INCLUDE_RELDIR})
IF("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    TARGET_LINK_LIBRARIES (${TARGET_LIB} PUBLIC dealii::dealii_release)
ELSE()
   TARGET_LINK_LIBRARIES (${TARGET_LIB} PUBLIC dealii::dealii_debug)
ENDIF()

#TARGET_INCLUDE_DIRECTORIES(${TARGET_LIB} PUBLIC ${DEAL_II_INCLUDE_DIRS})
#IF("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
#    TARGET_LINK_LIBRARIES (${TARGET_LIB} PUBLIC ${DEAL_II_LIBRARIES_RELEASE})
#ELSE()
#    TARGET_LINK_LIBRARIES (${TARGET_LIB} PUBLIC ${DEAL_II_LIBRARIES_DEBUG})
#ENDIF()


#TARGET_INCLUDE_DIRECTORIES(${TARGETLIB} PUBLIC ${DEAL_II_INCLUDE_DIRS})
#IF("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
#    TARGET_LINK_LIBRARIES (${TARGETLIB} PUBLIC deal.II::deal_II)
#ELSE()
#    TARGET_LINK_LIBRARIES (${TARGETLIB} PUBLIC deal.II::deal_II.g)
#ENDIF()

IF (WITH_CUSTOMIZED_DEALII)
    ADD_DEFINITIONS(-DDFTFE_WITH_CUSTOMIZED_DEALII)
    MESSAGE(STATUS "Using customized dealii")
ENDIF()


#
#Set use petsc and slepc flag
#
IF (DEAL_II_WITH_PETSC AND DEAL_II_WITH_SLEPC)
    ADD_DEFINITIONS(-DUSE_PETSC)
    MESSAGE(STATUS "Using petsc and slepc")
ENDIF()

#
# ELPA. Pass path to -DCMAKE_PREFIX_PATH
#
MESSAGE(STATUS "ELPA Dir = ${CMAKE_PREFIX_PATH}")
find_package(ELPA 2024 REQUIRED)
MESSAGE(STATUS "Will Link With ELPA")
target_link_libraries(${TARGET_LIB} PUBLIC ELPA::ELPA)
#
# Alglib. Set with -DALGLIB_DIR=/path/to/prefix
#
IF (DEFINED ALGLIB_DIR)
    FIND_LIBRARY(ALGLIB_LIBRARY
            NAMES Alglib alglib
            HINTS ${ALGLIB_DIR}/lib ${ALGLIB_DIR}
            NO_DEFAULT_PATH
    )
    IF(ALGLIB_LIBRARY STREQUAL "ALGLIB_LIBRARY-NOTFOUND")
        MESSAGE(FATAL_ERROR "-- Alglib was not found in ${ALGLIB_DIR}")
    ENDIF()
    MESSAGE("-- Use Alglib from ${ALGLIB_DIR}")
    TARGET_LINK_LIBRARIES(${TARGET_LIB} PUBLIC "${ALGLIB_LIBRARY}"
    )
    TARGET_INCLUDE_DIRECTORIES (${TARGET_LIB} PUBLIC "${ALGLIB_DIR}")
    TARGET_INCLUDE_DIRECTORIES (${TARGET_LIB} PUBLIC "${ALGLIB_DIR}/include")
ELSE()
    MESSAGE(FATAL_ERROR "-- Provide path to AlgLIB: -DALGLIB_DIR=/path/to/prefix")
ENDIF (DEFINED ALGLIB_DIR)



#
# LIBXC. Set with -DLIBXC_DIR=/path/to/prefix
#
IF (DEFINED LIBXC_DIR)
    FIND_LIBRARY(LIBXC_LIBRARY
            NAMES xc
            HINTS ${LIBXC_DIR}/lib
            NO_DEFAULT_PATH
    )
    IF(LIBXC_LIBRARY STREQUAL "LIBXC_LIBRARY-NOTFOUND")
        MESSAGE(FATAL_ERROR "-- LibXC was not found in ${LIBXC_DIR}")
    ENDIF()
    MESSAGE("-- Use LibXC from ${LIBXC_DIR}")
    TARGET_LINK_LIBRARIES(${TARGET_LIB}
            PUBLIC "${LIBXC_LIBRARY}"
    )
    TARGET_INCLUDE_DIRECTORIES (${TARGET_LIB} PUBLIC "${LIBXC_DIR}/include")
ELSE()
    MESSAGE(FATAL_ERROR "-- Provide path to prefix of LibXC: -DLIBXC_DIR=/path/to/prefix")
ENDIF (DEFINED LIBXC_DIR)

###########################################################################
#GPU
#
# NCCL. Pass path to -DCMAKE_PREFIX_PATH
#
IF (WITH_GPU AND "${GPU_LANG}" STREQUAL "cuda")
    IF (WITH_DCCL)
        find_package(NCCL REQUIRED)
        IF (NCCL_FOUND)
            MESSAGE(STATUS "Will Link With NCCL")
            ADD_DEFINITIONS(-DDFTFE_WITH_CUDA_NCCL)
            find_package(MPI REQUIRED)
            target_link_libraries(${TARGET_LIB} PUBLIC NCCL MPI::MPI_CXX)
        ENDIF()
    ENDIF()
ENDIF()

#
# RCCL. Pass path to -DCMAKE_PREFIX_PATH
#
IF (WITH_GPU AND "${GPU_LANG}" STREQUAL "hip")
    IF (WITH_DCCL)
        find_package(RCCL REQUIRED)
        IF (RCCL_FOUND)
            MESSAGE(STATUS "Will Link With RCCL")
            ADD_DEFINITIONS(-DDFTFE_WITH_HIP_RCCL)
            find_package(MPI REQUIRED)
            target_link_libraries(${TARGET_LIB} PUBLIC RCCL MPI::MPI_CXX)
        ENDIF()
    ENDIF()
ENDIF()
###########################################################################


TARGET_INCLUDE_DIRECTORIES(${TARGET_LIB} PUBLIC ${CMAKE_SOURCE_DIR}/include)
TARGET_LINK_LIBRARIES(${TARGET_EXE} PUBLIC ${TARGET_LIB})

# TESTING
IF (WITH_TESTING)
    ENABLE_TESTING()
    INCLUDE(CTest)
    ADD_SUBDIRECTORY(./tests)
ENDIF()
