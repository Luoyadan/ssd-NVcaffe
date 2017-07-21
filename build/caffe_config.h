/* Sources directory */
#define SOURCE_FOLDER "/opt/caffe"

/* Binaries directory */
#define BINARY_FOLDER "/opt/caffe/build"

/* NVIDIA Cuda */
#define HAVE_CUDA

/* NVIDIA cuDNN */
#define HAVE_CUDNN
#define USE_CUDNN
 
/* NVIDIA cuDNN */
/* #undef CPU_ONLY */
if(NCCL_FOUND)
/* #undef USE_NCCL */
endif()


/* Test device */
#define CUDA_TEST_DEVICE -1

/* Temporary (TODO: remove) */
#if 1
  #define CMAKE_SOURCE_DIR SOURCE_FOLDER "/src/"
  #define EXAMPLES_SOURCE_DIR BINARY_FOLDER "/examples/"
  #define CMAKE_EXT ".gen.cmake"
#else
  #define CMAKE_SOURCE_DIR "src/"
  #define EXAMPLES_SOURCE_DIR "examples/"
  #define CMAKE_EXT ""
#endif

/* Matlab */
/* #undef HAVE_MATLAB */

/* IO libraries */
#define USE_OPENCV
#define USE_LEVELDB
#define USE_LMDB
/* #undef ALLOW_LMDB_NOLOCK */
