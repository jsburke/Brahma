#include "cpu_simple.h"

//  Check asm after compile for below, want to try to keep everything in registers
void   body_body_accum_accel(int focus, int other, p_octant oct)  //  always calc as force on other from focus
{
	//  get the distance vectors, scalar distance
	data_t r_x, r_y, r_z, r;

	r_x = -((oct->pos_x[focus]) - (oct->pos_x[other]));  // keep sign because we want directionality
	r_y = -((oct->pos_y[focus]) - (oct->pos_y[other]));  // will be trampled by squaring in macro
	r_z = -((oct->pos_z[focus]) - (oct->pos_z[other]));  // for distance anyhow

	r 	= DISTANCE(r_x, r_y, r_z);

	//  Force calculations

	data_t F_x, F_y, F_z, F_part;

	F_part = FORCE_PARTIAL(oct->mass[focus], oct->mass[other], r);

	F_x = F_part * r_x;
	F_y = F_part * r_y;
	F_z = F_part * r_z;

	//  Acceleration accumulations

	data_t mass = oct->mass[focus];

	oct->acc_x[focus] += F_x / mass;
	oct->acc_y[focus] += F_y / mass;
	oct->acc_z[focus] += F_z / mass;

	//  for other body, remember equal and opposite, Newton No. 3

	data_t mass = oct->mass[other];

	oct->acc_x[other] += -F_x / mass;
	oct->acc_y[other] += -F_y / mass;
	oct->acc_z[other] += -F_z / mass;
}

// not a fan of below because breaks DRY, try to find more elegant solution
void 	body_oct_accum_accel(p_octant local, int body, p_octant distal)
{
	//  get the distance vectors, scalar distance
	data_t r_x, r_y, r_z, r;

	r_x = -((local->pos_x[body]) - (distal->mass_center_x));
	r_y = -((local->pos_y[body]) - (distal->mass_center_y));
	r_z = -((local->pos_z[body]) - (distal->mass_center_z));

	r 	= DISTANCE(r_x, r_y, r_z);

	//  Force calculations

	data_t F_x, F_y, F_z, F_part;

	F_part = FORCE_PARTIAL(local->mass[body], distal->mass_total, r);

	F_x = F_part * r_x;
	F_y = F_part * r_y;
	F_z = F_part * r_z;

	//  Acceleration accumulations

	data_t mass = local->mass[body];

	local->acc_x[body] += F_x / mass;
	local->acc_y[body] += F_y / mass;
	local->acc_z[body] += F_z / mass;
}

void	body_pos_update(int body, p_octant oct, int time)
{}

void	body_vel_update(int body, p_octant oct, int time)
{}
