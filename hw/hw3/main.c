/* 
 * Calculating the exponential function [exp(x)] using infinite series. 
 *
 * to compile: run compile script by : sh compile.sh
 * to execute: Go to build Folder and run : ./expofx  ${input}
 *
 * 
 *
 * 
 * Author: Roohollah Amiri
 * Date: 02/15/2016
*/


#include <stdio.h>
#include <string.h>
#include <float.h>
#include "utility.h"
#include "power.h"
#include "pefactorial.h"

int quotient(integer dividend, integer divisor);
int oneTermOfTaylor(int x, int p);
float oneTermOfTaylor_small_int(int x, int p);
int factorial_small_integer(int n);

/*for testing factorial fucntion*/

// int main(int argc, char* argv[]) {
//     int n = atoi(argv[1]);
//     double start, finish, elapsed;
//     GET_TIME(start);
//     puts(integer_to_string(factorial(n)));
//     GET_TIME(finish);
//     elapsed = finish - start;
//     printf("The code to be timed took %e seconds\n", elapsed);
//     return 0;
// }


int main(int argc, char* argv[])
{
	int power = atoi(argv[1]);
	printf("e to the power of %d\n", power);
	
	int num_terms = atoi(argv[2]);
	printf("Number of Terms =%d\n", num_terms);
	
	
	int ex = 0;
	REAL ex1 =0.0;
	
	if(num_terms <= 20)
	{
		for(int i=0;i<num_terms;i++)
		{
			ex1 = ex1 + oneTermOfTaylor_small_int(power, i);
		}
		#ifdef PRECISION
		printf("The result = %e\n", ex1);
		#else
		printf("The result = %f\n", ex1);
		#endif
	}
	else
	{
		for(int i=0;i<num_terms;i++)
		{
			ex = ex + oneTermOfTaylor(power, i);
		}
		printf("The result is = %d\n", ex);
	}
	


	return 0;
}

int quotient(integer dividend, integer divisor)
{
// 	integer mod;
	integer dum;
// 	mod = create_integer(divisor.num_components + 1);
// 	mod_integer(dividend, divisor, mod);
	int quef = -1;
	dum = create_integer(dividend.num_components + divisor.num_components+ 1);
	copy_integer(divisor, dum);
	int getmeOut = -1;
	while(getmeOut<0)
	{
		quef++;
         multiply_small_integer(divisor, quef, dum);
		 if(compare_integers(dividend,dum)<0)
		 {
			 getmeOut = 1;
		}
	}
	free_integer(dum);
	return quef-1;
}

/*Function to calculate x^p/p! */
int oneTermOfTaylor(int x, int p)
{
	if(p==0)
		return 1;
	char str[15];
	sprintf(str, "%d", x);
	integer xx = string_to_integer(str);
	integer a  = power_big(xx, p);
	integer b = factorial(p);
	int res = quotient(a,b);
	free_integer(a);
	free_integer(b);
	free_integer(xx);
	printf("term = %d\n", p);
	return res;
}

REAL oneTermOfTaylor_small_int(int x, int p)
{
	int a  = power(x, p);
	int b = factorial_small_integer(p);
	REAL res = (REAL)a/(REAL)b;
	return res;
}


int factorial_small_integer(int n)
{
	if(n<0)
		return -1;
	
	if(n > 1)
		return n*factorial_small_integer(n-1);
	else
		return 1;
}

// int main(int argc, char* argv[]) 
// {
// 	integer a1, a2, result;
// 	int quot;
// 	a1.num_components = a2.num_components = 0; /* quelch compiler warnings */
// 	if (argc > 2) a1 = string_to_integer(argv[2]);
// 	if (argc > 3) a2 = string_to_integer(argv[3]);
// 	if (strcmp(argv[1], "add") == 0) {
// 		result = create_integer(a1.num_components + a2.num_components + 1);
// 		add_integer(a1, a2, result);
// 	} else if (strcmp(argv[1], "subtract") == 0) {
// 		result = create_integer(a1.num_components + a2.num_components + 1);
// 		subtract_integer(a1, a2, result);
// 	} else if (strcmp(argv[1], "multiply") == 0) {
// 		result = create_integer(a1.num_components + a2.num_components + 1);
// 		multiply_integer(a1, a2, result);
// 	} else if (strcmp(argv[1], "mod") == 0) {
// 		result = create_integer(a2.num_components + 1);
// 		mod_integer(a1, a2, result);
// 		quot = quotient(a1,a2);
// 	} else if (strcmp(argv[1], "factorial") == 0) {
// 		int i, n = atoi(argv[2]);
// 		result = create_integer(n*n + 1);
// 		set_zero_integer(result);
// 		result.c[0] = 1;
// 		for(i=2; i<=n; i++) {
// 			multiply_small_integer(result, i, result);
// 		}
// 	}
// 	puts(integer_to_string(result));
// 	printf("solution = %d\n", quot);
// 	return 0;
// }