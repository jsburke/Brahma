#include "cpu_simple.h"


const data_t GRAV_CONST = 6.674e-11;

data_t body_body_partial_force(int focus, int other, p_octant oct)  // not completely right
{

}

void   body_body_accum_accel(int focus, int other, p_octant oct)
{
	data_t force = body_body_force(focus, other, oct);
}