/*
 * stencil.cu
 *
 *  Created on: Mar 13, 2016
 *      Author: Roohollah Amiri
 *      to compile: mkdir build;cd build;cmake ../;make
 *		to execute: ./cpuVsgpu
 */

#include <stdio.h>
#include "timer.h"
#include "utils.h"
#include "commons.h"
#include "deviceKernels.h"

int main()
{
  int *h_in, *h_out;//[NUM_ELEMENTS + 2 * RADIUS], h_out[NUM_ELEMENTS];
  int *d_in, *d_out;

  int NUM_ELEMENTS[5] = {100,10000,100000,1000000,10000000};
  int length = sizeof(NUM_ELEMENTS)/sizeof(int);
  int N;

  printf("PART A:\n");
  /*Output Table*/
  char bufferCpu [50];char bufferGpu_gl[50]; char bufferGpu_sh[50];char bufferCheck[50];
//  sprintf(bufferCpu, "CPU(%sS)", MICRO);sprintf(bufferGpu, "GPU(%sS)", MICRO);sprintf(bufferCheck, "check");
  sprintf(bufferCpu,"CPU(mS)");sprintf(bufferGpu_gl, "GPU(global)(mS)");
  sprintf(bufferGpu_sh, "GPU(shared)(mS)");sprintf(bufferCheck,"check");
  printElement("N", 2*nameWidth);printElement(separator, nameWidth);
  printElement(bufferCpu, 3*nameWidth);printElement(bufferCheck, 2*nameWidth);
  printElement(bufferGpu_gl, 3*nameWidth);printElement(bufferCheck, 2*nameWidth);
  printElement(bufferGpu_sh, 3*nameWidth);printElement(bufferCheck, 3*nameWidth,1);

  // Running for part A
  for(int j=0;j<length;j++)
  {
	  N = NUM_ELEMENTS[j];
	  size_t size = (N + 2*RADIUS)*sizeof(int);
	  h_in  = (int *)malloc(size);
	  h_out = (int *)malloc(N *sizeof(int));

	  // Initialize host data
	  initArray(h_in,N + 2*RADIUS,1);
	  initArray(h_out,N,0);

	  //for the CPU
	  double startCpu, stopCpu;

	  /*CPU*/
	  GET_TIME(startCpu);
	  stencil_cpu(h_in, N + 2*RADIUS, h_out, N, RADIUS);
	  GET_TIME(stopCpu);
	  char bufferTimeCpu[50];
	  sprintf(bufferTimeCpu, "%.6f", 1e3*(stopCpu-startCpu));

	  // check GPU output
	  char buffercheckCpu[50];
	  checkProcess(h_out, N, buffercheckCpu);

		// Allocate space on the device
		cudaCheck( cudaMalloc( &d_in, size) );
		cudaCheck( cudaMalloc( &d_out, N*sizeof(int)) );

		// Copy input data to device
		cudaCheck( cudaMemcpy( d_in, h_in, size, cudaMemcpyHostToDevice) );

		// Run global memory kernel
		char buffercheckGpu_gl[50];char bufferTimeGpu_gl[50];
		runKernel(stencil_1d_gl,d_in,d_out,h_out,N,BLOCK_SIZE,buffercheckGpu_gl,bufferTimeGpu_gl);

		initArray(h_out,N,0);

		// Run shared memory kernel
		char buffercheckGpu_sh[50];char bufferTimeGpu_sh[50];
		runKernel(stencil_1d_sh,d_in,d_out,h_out,N,BLOCK_SIZE,buffercheckGpu_sh,bufferTimeGpu_sh);

		/*print results in output table*/
		printElement(N, 2*nameWidth);printElement(separator, nameWidth);
		printElement(bufferTimeCpu, 2*nameWidth);printElement(separator, nameWidth);
		printElement(buffercheckCpu, 2*nameWidth);
		printElement(bufferTimeGpu_gl, 3*nameWidth);printElement(buffercheckGpu_gl, 2*nameWidth);
		printElement(bufferTimeGpu_sh, 3*nameWidth);printElement(buffercheckGpu_sh, 2*nameWidth,1);

		// Free out memory
		cudaFree(d_in);
		cudaFree(d_out);
		free(h_in);
		free(h_out);
  }

  N = 1000000;
  printf("PART B: Num_Elements=%d\n",N);
  // Running for part B

  printElement("#threads/block", 2*nameWidth);printElement(separator, 4);
  printElement(bufferGpu_gl, 3*nameWidth);printElement(bufferGpu_sh, 3*nameWidth,1);

  int block_size[5]={16,64,256,512,1024};
  length = sizeof(block_size)/sizeof(int);
  for(int j=0;j<length;j++)
    {
  	  int blk_size = block_size[j];
  	  size_t size = (N + 2*RADIUS)*sizeof(int);
  	  h_in  = (int *)malloc(size);
  	  h_out = (int *)malloc(N *sizeof(int));

  	  // Initialize host data
  	  initArray(h_in,N + 2*RADIUS,1);
  	  initArray(h_out,N,0);

  	  // check GPU output
  	  char buffercheckCpu[50];
  	  checkProcess(h_out, N, buffercheckCpu);

  		// Allocate space on the device
  		cudaCheck( cudaMalloc( &d_in, size) );
  		cudaCheck( cudaMalloc( &d_out, N*sizeof(int)) );

  		// Copy input data to device
  		cudaCheck( cudaMemcpy( d_in, h_in, size, cudaMemcpyHostToDevice) );

  		// Run global memory kernel
  		char buffercheckGpu_gl[50];char bufferTimeGpu_gl[50];
  		runKernel(stencil_1d_gl,d_in,d_out,h_out,N,blk_size,buffercheckGpu_gl,bufferTimeGpu_gl);

  		initArray(h_out,N,0);

  		// Run shared memory kernel
  		char buffercheckGpu_sh[50];char bufferTimeGpu_sh[50];
  		runKernel(stencil_1d_sh,d_in,d_out,h_out,N,blk_size,buffercheckGpu_sh,bufferTimeGpu_sh);

  		/*print results in output table*/
  		printElement(blk_size, 2*nameWidth);printElement(separator, nameWidth);
  		printElement(bufferTimeGpu_gl, 3*nameWidth);
  		printElement(bufferTimeGpu_sh, 3*nameWidth,1);

  		// Free out memory
  		cudaFree(d_in);
  		cudaFree(d_out);
  		free(h_in);
  		free(h_out);
    }
  return 0;
}


