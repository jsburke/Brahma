#include "cpu_n2n.h"

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

void	force_accum(data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* frc_x, data_t* frc_y, data_t* frc_z, int focus, int comp)
{
	//  First the distance
	data_t r_x, r_y, r_z, r;

	r_x = pos_x[focus] - pos_x[comp];
	r_y = pos_y[focus] - pos_y[comp];
	r_z = pos_z[focus] - pos_z[comp];

	r = DISTANCE(r_x, r_y, r_z);

	//  then the force for the focus

	data_t F_part;

	F_part = FORCE_PARTIAL(mass[focus], mass[comp], r);

	frc_x[focus]  += F_part * r_x;
	frc_y[focus]  += F_part * r_y;
	frc_z[focus]  += F_part * r_z;

	// force for the comparison
	// we know this by Newton's 3rd law

	frc_x[comp]   += -frc_x[focus];
	frc_y[comp]   += -frc_y[focus];
	frc_z[comp]   += -frc_z[focus];
}