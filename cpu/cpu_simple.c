#include "cpu_simple.h"
#include <math.h>

const data_t GRAV_CONST = 6.674e-11;

data_t body_body_distance(int focus, int other, p_octant oct)
{
	// first get the orthogonal diffs
	int x_diff = (oct->pos_x[focus]) - (oct->pos_x[other]);
	int y_diff = (oct->pos_y[focus]) - (oct->pos_y[other]);
	int z_diff = (oct->pos_z[focus]) - (oct->pos_z[other]);

	// square and squareroot
	return sqrt(x_diff*x_diff + y_diff*y_diff + z_diff*z_diff);
}

data_t body_body_force(int focus, int other, p_octant oct)  // not completely right
{
	data_t distance = body_body_distance(focus, other, oct);
	data_t dist_sq	= distance * distance;

	//  G * (m1 * m2) / r^2

	return GRAV_CONST * (((oct->mass[focus]) * (oct->mass[other])) / dist_sq);
}

void   body_body_accum_accel(int focus, int other, p_octant oct)
{
	data_t force = body_body_force(focus, other, oct);
}