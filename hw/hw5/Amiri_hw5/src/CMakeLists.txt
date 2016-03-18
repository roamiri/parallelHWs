set(PROJECTNAME stencil)
project(${PROJECTNAME})
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-Wall")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall")

# Pass options to NVCC
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O2
    -arch=sm_30
    )

set(srcs
	stencil.cu
)

set(hdrs
	commons.h
	deviceKernels.h
	timer.h
	utils.h
)
cuda_add_executable(${PROJECTNAME} ${srcs} ${hdrs})

install(TARGETS ${PROJECTNAME} RUNTIME DESTINATION bin)