#include "cpu_simple.h"

void   body_body_accum_accel(int focus, int other, p_octant oct)  //  always calc as force on other from focus
{
	//  get the distance vectors, scalar distance
	data_t r_x, r_y, r_z, r;

	r_x = -((oct->pos_x[focus]) - (oct->pos_x[other]));  // keep sign because we want directionality
	r_y = -((oct->pos_y[focus]) - (oct->pos_y[other]));  // will be trampled by squaring in macro
	r_z = -((oct->pos_z[focus]) - (oct->pos_z[other]));  // for distance anyhow

	r 	= BODY_BODY_DISTANCE(r_x, r_y, r_z);

	//  Force calculations

	data_t F_x, F_y, F_z, F_part;

	F_part = FORCE_PARTIAL(oct->mass[focus], oct->mass[other], r);

	F_x = F_part * r_x;
	F_y = F_part * r_y;
	F_z = F_part * r_z;
}