/*
Program: cpuVsgpu
 
Author: Roohollah Amiri

to compile: mkdir build;cd build;cmake ../;make
to execute: ./cpuVsgpu 

*/

#include<iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "timer.h"

#define NY 32

#define MICRO "\u03bc"

__global__ void myKernel ( int *a, int *b, int *c , int NX ) 
{
	//int tid = blockIdx.x;

	int tid = threadIdx.x;

	if (tid < NX)
		a[tid] = b[tid] + c[tid];
}


void addVectors (int *a, int *b, int *c, int NX) 
{
	for (int i=0; i<NX; i++) 
	{
		a[i]=b[i]+c[i];
	} 
}

void random_ints(int* a, int N, int max)
{
	srand(time(NULL));
	for(int i=0;i<N;i++)
	{
		a[i] = rand() % max + 0;
	}
}

const char separator    = ' ';
const int nameWidth     = 6;
/**
 * Function for printing output as a table. If newline=1, there will be new line after the printed text.
 */
template<typename T> void printElement(T t, const int& width, int newline = 0)
{
	if(!newline)
		std::cout << std::left << std::setw(width) << std::setfill(separator) << t;
	else
		std::cout << std::left << std::setw(width) << std::setfill(separator) << t << std::endl;
}

  int main (void) 
  {
     //let's see how many CUDA capable GPUs we have

     int gpuCount;

     cudaGetDeviceCount( &gpuCount);

     printf(" Number of GPUs = %d\n", gpuCount);

     int myDevice = 0;

     cudaSetDevice( myDevice );
 
	int NumElements[5] = {100, 10000, 1000000, 10000000, 100000000};
	int NX;
	char bufferCpu [50];char bufferGpu [50];
	sprintf(bufferCpu, "CPU(%sS)", MICRO);sprintf(bufferGpu, "GPU(%sS)", MICRO);
	printElement("N", 2*nameWidth);printElement(bufferCpu, 3*nameWidth);printElement(bufferGpu, 2*nameWidth,1);
	
	for(int i=0;i<5;i++)
	{
		NX = NumElements[i];
		
		//let's use the device to do some calculations     

		int a[NX], b[NX], c[NX];  //create arrays on the host
		int *d_a, *d_b, *d_c;     //create pointers for the device

		cudaMalloc( (void**) &d_a, NX*sizeof(int) ); //Be careful not to dereference this pointer, attach d_ to varibles
		cudaMalloc( (void**) &d_b, NX*sizeof(int) ); 
		cudaMalloc( (void**) &d_c, NX*sizeof(int) ); 


		// Let's fill the arrays with some numbers

		for (int i=0; i<NX; i++) {a[i] = 0;}
		random_ints(b,NX,20);random_ints(c,NX,20);
		
		
	
		// Let's create the infrastructure to time the host & device operations 

		double start, finish; //for the CPU
	
		cudaEvent_t timeStart, timeStop; //WARNING!!! use events only to time the device
		cudaEventCreate(&timeStart);
		cudaEventCreate(&timeStop);
		float elapsedTime; // make sure it is of type float, precision is milliseconds (ms) !!!

		GET_TIME(start);

		// Let's do the following operation on the arrays on the host: a = b + c 
		addVectors (a, b, c, NX);

		GET_TIME(finish);

// 		printf("elapsed wall time (host) = %.6f %sS\n", 1e6*(finish-start), MICRO);
		char bufferTimeCpu[50];
		sprintf(bufferTimeCpu, "%.6f", 1e6*(finish-start));

		// Let's print the results on the screen

// 		printf("b, c, a=b+c\n");

	//     for (int i=0; i<5; i++) {
	//         printf("%d %2d %3d\n", b[i], c[i], a[i]);
	//     }
			
		cudaMemcpy( d_b, b, NX*sizeof(int), cudaMemcpyHostToDevice );  //memcpy(dest,src,...
		cudaMemcpy( d_c, c, NX*sizeof(int), cudaMemcpyHostToDevice );  //memcpy(dest,src,...

		cudaEventRecord(timeStart, 0); //don't worry for the 2nd argument zero, it is about cuda streams

		dim3 threadsPerBlock(16,16); //Best practice of having 256 threads per block
		dim3 numBlocks(NX/threadsPerBlock.x,NY/threadsPerBlock.y);
	
		myKernel<<<numBlocks,threadsPerBlock>>> (d_a, d_b, d_c, NX);  // Be careful with the syntax one less "<" you are in trouble! 
		cudaEventRecord(timeStop, 0);
		cudaEventSynchronize(timeStop);

		//WARNING!!! do not simply print (timeStop-timeStart)!!

		cudaEventElapsedTime(&elapsedTime, timeStart, timeStop);
	
// 		printf("elapsed wall time (device) = %3.1f %sS\n", 1e3*elapsedTime, MICRO);
		char bufferTimeGpu[50];
		sprintf(bufferTimeGpu, "%.6f", 1e3*elapsedTime);

		
		printElement(NX, 2*nameWidth);printElement(bufferTimeCpu, 2*nameWidth); printElement(separator, nameWidth);printElement(bufferTimeGpu, 2*nameWidth,1);
		
		cudaEventDestroy(timeStart); 
		cudaEventDestroy(timeStop);

		cudaMemcpy( a, d_a, NX*sizeof(int), cudaMemcpyDeviceToHost );

	//     for (int i=0; i<NX; i++) {
	//         printf("%3d\n", a[i]);
	//     }

		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
	}

     return 0; 
  }
