/*
 * deviceKernels.h
 *
 *  Created on: Mar 13, 2016
 *      Author: roohi
 */

#ifndef DEVICEKERNELS_H_
#define DEVICEKERNELS_H_

#include "commons.h"

void stencil_cpu(int* input, int size_in, int* output, int size_out, int radius)
{
	int index;
	for(index=radius;index<size_in;index++)
	{
		int result = 0;
		for (int offset = -radius ; offset <= radius ; offset++)
		{
	        result += input[index + offset];
		}
		if(index-radius<size_out)
			output[index-radius] = result;
	}
}

__global__ void stencil_1d_sh(int *in, int *out)
{
    // __shared__ keyword to declare variables in shared block memory
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    int gindex = threadIdx.x + (blockIdx.x * blockDim.x) + RADIUS;
    int lindex = threadIdx.x + RADIUS;

    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS)
    {
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }

    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
        result += temp[lindex + offset];

    // Store the result
    out[gindex-RADIUS] = result;
}

__global__ void stencil_1d_gl(int* in, int* out)
{
	int result;
	if(threadIdx.x>=RADIUS)
	{
		result = 0;
		for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
		{
	        result += in[threadIdx.x + offset];
		}
		out[threadIdx.x-RADIUS] = result;
	}
}

void runKernel(void (*kernel)(int*,int*), int* device_in, int* device_out, int* host_out,
		int num_elements,int blk_size,char* check, char* buffer_time )
{
	//for the GPU
	cudaEvent_t startGpu, stopGpu;
	cudaEventCreate(&startGpu);
	cudaEventCreate(&stopGpu);

	cudaEventRecord(startGpu);
	kernel<<< (num_elements + blk_size - 1)/blk_size, blk_size >>> (device_in, device_out);
	cudaEventRecord(stopGpu);

	// Check errors from launching the kernel
	cudaCheck(cudaPeekAtLastError());

	cudaCheck( cudaMemcpy( host_out, device_out, num_elements * sizeof(int), cudaMemcpyDeviceToHost) );

	cudaEventSynchronize(stopGpu);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, startGpu, stopGpu);

	// check GPU output
	checkProcess(host_out, num_elements, check);
	sprintf(buffer_time, "%.6f", milliseconds);
}


#endif /* DEVICEKERNELS_H_ */
