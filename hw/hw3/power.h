/* Extended version of power function that can work
 *  for float x and negative y*/

#include <stdio.h>
#include "utility.h"
#include "integer.h"


REAL power(REAL x, int y)
{
    REAL temp;
    if( y == 0)
       return 1;
    temp = power(x, y/2);       
    if (y%2 == 0)
        return temp*temp;
    else
    {
        if(y > 0)
            return x*temp*temp;
        else
            return (temp*temp)/x;
    }
}  


integer power_big(integer x, int y)
{
    integer temp;
    if( y == 0)
	{
		integer res  = create_integer(x.num_components);
		set_zero_integer(res);
		res.c[0]=1;
		return res;
	}
    temp = power_big(x, y/2);       
    if (y%2 == 0)
	{
		integer res  = create_integer(temp.num_components*2);
		multiply_integer(temp,temp,res);
		return res;
	}
    else
    {
        if(y > 0)
		{
			integer res  = create_integer(temp.num_components*2);
			integer res1  = create_integer(temp.num_components*2);
			multiply_integer(temp,temp,res);
			multiply_integer(x,res,res1);
            return res1;
		}
//         else
// 		{
//             return divide_small_integer(shift_left_one_integer(temp),x,);
// 		}
    }
}  

integer powerXY(integer x, int y)
{
	if( y == 0)
	{
		set_zero_integer(x);
		x.c[0]=1;
		return x;
	}
	
	if (y%2 == 0)
	{
		integer res  = create_integer(x.num_components*2);
		for(int i=0;i<y/2;i++)
		{
			multiply_integer(x,x,res);
		}
		return res;
	}
	else
	{
		
	}
}





