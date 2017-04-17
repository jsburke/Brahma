#include "cpu_simple.h"

//  Check asm after compile for below, want to try to keep everything in registers
void	body_body_accum_accel(int focus, int other, p_octant oct)  //  always calc as force on other from focus
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

void	body_pos_update(int body, p_octant oct, time_t timestep)
{
	oct->pos_x[body] += DISPLACE(oct->vel_x[body], oct->acc_x[body], timestep);
	oct->pos_y[body] += DISPLACE(oct->vel_y[body], oct->acc_y[body], timestep);
	oct->pos_z[body] += DISPLACE(oct->vel_z[body], oct->acc_z[body], timestep);
}

void	body_vel_update(int body, p_octant oct, time_t time)
{
	oct->vel_x[body] += oct->acc_x[body] * timestep;
	oct->vel_y[body] += oct->acc_y[body] * timestep;
	oct->vel_z[body] += oct->acc_z[body] * timestep;
}

int 	body_alloc(p_octant root, nbody *bodies[], int num_bodies)
{
	data_t upper_x, lower_x, half_x upper_y, lower_y, half_y upper_z, lower_z, half_z;
	int octant, suboctant;
	data_t body_x, body_y, body_z;
	int i;
	p_octant root_children = root->children;

	for(i = 0; i < num_bodies; i++)
	{
		octant 		= 0;
		suboctant 	= 0;

		body_x = bodies[i]->pos_x;
		body_y = bodies[i]->pos_y;
		body_z = bodies[i]->pos_z;

		// find location

		if(body_x >= 0) // even octant
		{
			octant 	+= 1;
			upper_x = MAX_POS_POSITION;
			half_x  = POS_QUARTER_MARK;
			lower_x = 0;
		}
		else
		{
			upper_x = 0;
			half_x  = NEG_QUARTER_MARK;
			lower_x = MAX_NEG_POSITION;
		}
		suboctant += (body_x >= half_x) 1 : 0;

		if(body_y >= 0)
		{
			octant 	+= 2;
			upper_y = MAX_POS_POSITION;
			half_y  = POS_QUARTER_MARK;
			lower_y = 0;
		}
		else
		{
			upper_y = 0;
			half_y  = NEG_QUARTER_MARK;
			lower_y = MAX_NEG_POSITION;
		}
		suboctant += (body_y >= half_y) 2 : 0;

		if(body_z >= 0)
		{
			octant 	+= 4;
			upper_z = MAX_POS_POSITION;
			half_z  = POS_QUARTER_MARK;
			lower_z = 0;
		}
		else
		{
			upper_z = 0;
			half_z  = NEG_QUARTER_MARK;
			lower_z = MAX_NEG_POSITION;
		}
		suboctant += (body_z >= half_z) 4 : 0;

		if(KILL == octant_add_body(root_children[octant]->children[suboctant], bodies[i]))
		{
			return KILL;
		}
	}

	return PASS;
}