/* File:     utility.h
 *
 * Purpose:  Useful Macros and Functions that is used in project
 * 
 * -1:       Define a macro that returns the number of seconds that 
 *           have elapsed since some point in the past.  The timer
 *           should return times with microsecond accuracy.
 * Note:     The argument passed to the GET_TIME macro should be
 *           a double, *not* a pointer to a double.
 * Example:  
 *    #include "utility.h"
 *    . . .
 *    double start, finish, elapsed;
 *    . . .
 *    GET_TIME(start);
 *    . . .
 *    Code to be timed
 *    . . .
 *    GET_TIME(finish);
 *    elapsed = finish - start;
 *    printf("The code to be timed took %e seconds\n", elapsed);
 *    
 * -2:       Define PRECISION Macro to make single or double precision computing
 *           For double precision change the macro PRECISION to 1       
 *
 *
 */
#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <sys/time.h>
#include "limits.h"

/* The argument now should be a double (not a pointer to a double) */
#define GET_TIME(now) { \
   struct timeval t; \
   gettimeofday(&t, NULL); \
   now = t.tv_sec + t.tv_usec/1000000.0; \
}

#define PRECISION 0

#if PRECISION 
    #define REAL double
    #define INTEGER unsigned long int //unsigned int
    #define MACHINE_EPSILON DBL_EPSILON
    #define MAX_INT ULONG_MAX //UINT_MAX 
#else
    #define REAL float
    #define INTEGER unsigned int
    #define MACHINE_EPSILON FLT_EPSILON
    #define MAX_INT UINT_MAX
#endif

#endif
