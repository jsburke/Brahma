#include "cpu_n2n.h"

const data_t GRAV_CONST = 6.674e-11;

int		body_count(char* filename)
{
	int count = 0;

	while(*filename) // still characters to process
	{
		if(isdigit(*filename))
		{
			count *= 10;
			count += strtol(filename, &filename, 10);
		}
		filename++;
	}

	return count;
}

void	force_zero(data_t* x, data_t* y, data_t* z, int len)
{
	int i;

	for(i = 0; i < len; i++)
	{
		x[i] = 0;
		y[i] = 0;
		z[i] = 0;
	}
}

void	force_accum(data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int focus, int comp)
{
	//  First the distance
	data_t r_x, r_y, r_z, r;

	r_x = pos_x[focus] - pos_x[comp];
	r_y = pos_y[focus] - pos_y[comp];
	r_z = pos_z[focus] - pos_z[comp];

	r = DISTANCE(r_x, r_y, r_z);

	//  then the force for the focus

	data_t F_part;

	F_part = FORCE_PARTIAL(GRAV_CONST, mass[focus], mass[comp], r);

	fma_x[focus]  += F_part * r_x;
	fma_y[focus]  += F_part * r_y;
	fma_z[focus]  += F_part * r_z;

	// force for the comparison
	// we know this by Newton's 3rd law

	fma_x[comp]   += -fma_x[focus];
	fma_y[comp]   += -fma_y[focus];
	fma_z[comp]   += -fma_z[focus];
}

void	position_update(data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* vel_x, data_t* vel_y, data_t* vel_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int len, int time)
{
	int i;

	for(i = 0; i < len; i++)
	{
		// convert forces to acceleration, saves a multiply later
		fma_x[i] /= mass[i];
		fma_y[i] /= mass[i];
		fma_z[i] /= mass[i];

		pos_x[i] += DISPLACE(vel_x[i], fma_x[i], time);
		pos_y[i] += DISPLACE(vel_y[i], fma_y[i], time);
		pos_z[i] += DISPLACE(vel_z[i], fma_z[i], time);
	}
}

void	velocity_update(data_t* mass, data_t* vel_x, data_t* vel_y, data_t* vel_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int len, int time)
{

}
