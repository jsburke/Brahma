#include "cpu_simple.h"
#include <math.h>

data_t body_body_distance(int focus, int other, p_octant oct)
{
	// first get the orthogonal diffs
	int x_diff = (oct->pos_x[focus]) - (oct->pos_x[other]);
	int y_diff = (oct->pos_y[focus]) - (oct->pos_y[other]);
	int z_diff = (oct->pos_z[focus]) - (oct->pos_z[other]);

	// square and squareroot
	return sqrt(x_diff*x_diff + y_diff*y_diff + z_diff*z_diff);
}