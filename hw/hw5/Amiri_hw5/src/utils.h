/*
 * utils.h
 *
 *  Created on: Mar 13, 2016
 *      Author: roohi
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
#include <iomanip>
#include "commons.h"

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



// CUDA API error checking macro
static void handleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess)
    {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
        exit( EXIT_FAILURE );
    }
}
#define cudaCheck( err ) (handleError( err, __FILE__, __LINE__ ))



void initArray(int* arr, int size, int value)
{
	int i;
	for( i = 0; i < size; ++i )
		arr[i] = value; // With a value of 1 and RADIUS of 3, all output values should be 7
}

void checkProcess(int* arr, int size, char* buffer)
{
	// Verify every out value is 7
	int i;
	for( i = 0; i < size; ++i )
		if (arr[i] != 7)
		{
//			printf("error at item=%d with value=%d\n",i,arr[i]);
			break;
		}

	if (i == size)
		sprintf(buffer,"SUCCESS");
	else
		sprintf(buffer,"FAIL");
}


#endif /* UTILS_H_ */
