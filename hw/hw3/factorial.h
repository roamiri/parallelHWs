/* The authors of this work have released all rights to it and placed it
in the public domain under the Creative Commons CC0 1.0 waiver
(http://creativecommons.org/publicdomain/zero/1.0/).

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Retrieved from: http://en.literateprograms.org/Factorials_with_prime_factorization_(C)?oldid=6227
*/

#include <math.h>  /* for log() */
#include <stdlib.h>
#include <stdio.h>
#include "integer.h"
#include "timer.h"

/* Return the power of the prime number p in the
   factorization of n! */
int multiplicity(int n, int p) {
    int q = n, m = 0;
    if (p > n) return 0;
    if (p > n/2) return 1;
    while (q >= p) {
        q /= p;
        m += q;
    }
    return m;
}

unsigned char* prime_table(int n) {
    int i, j;
    unsigned char* sieve = (unsigned char*)calloc(n+1, sizeof(unsigned char));
    sieve[0] = sieve[1] = 1;
    for (i=2; i*i <= n; i++)
        if (sieve[i] == 0)
            for (j=i*i; j <= n; j+=i)
                sieve[j] = 1;
    return sieve;
}
void exponent(unsigned int base, unsigned int power, integer result) {
    int bit;
    integer temp = create_integer(result.num_components*2);
    set_zero_integer(result);
    result.c[0] = 1;

    bit=sizeof(power)*CHAR_BIT - 1;
    while ((power & (1 << bit)) == 0) bit--;
    for( ; bit>=0; bit--) {
        multiply_integer(result, result, temp);
        copy_integer(temp, result);
        if ((power & (1 << bit)) != 0) {
            multiply_small_integer(result, base, result);
        }
    }

    free_integer(temp);
}
integer factorial(int n) {
    unsigned char* primes = prime_table(n);
    int p;
    integer result =
      create_integer((int)ceil((n*log((double)n)/log(2) + 1)/COMPONENT_BITS));
    integer p_raised = create_integer(result.num_components);
    integer temp     = create_integer(result.num_components*2);
    set_zero_integer(result);
    result.c[0] = 1;

    for(p = 2; p <= n; p++) {
        if (primes[p] == 1) continue;
        exponent(p, multiplicity(n,p), p_raised);
        multiply_integer(result, p_raised, temp);
        copy_integer(temp, result);
    }

    free_integer(temp);
    free_integer(p_raised);
    free(primes);
    return result;
}


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
