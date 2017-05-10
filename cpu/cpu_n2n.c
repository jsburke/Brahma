#include "cpu_n2n.h"

const 	data_t GRAV_CONST = 6.674e-11;
#define LINE_LEN			512

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

	//printf("F_part %lf | m1 %.2lf kg | m2 %.2lf kg | r %.2lf km\n", F_part, mass[focus], mass[comp], r);

	fma_x[focus]  += F_part * r_x;
	fma_y[focus]  += F_part * r_y;
	fma_z[focus]  += F_part * r_z;

}

void	position_update(data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* vel_x, data_t* vel_y, data_t* vel_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int len, int time)
{
	//  NB, when this is invoked, fma arrays will have forces built up in force_accum()
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
	// NB, when this is invoked, fma arrays should be accelerations set in position_update()
	int i;

	for(i = 0; i < len; i++)
	{
		vel_x[i] += fma_x[i] * time;
		vel_y[i] += fma_y[i] * time;
		vel_z[i] += fma_z[i] * time;
	}
}

int 	fileread_build_arrays(char* filename, data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* vel_x, data_t* vel_y, data_t* vel_z, int len)
{
	// returns true -- false
	FILE *fp = fopen(filename, "r");

	if(fp == NULL) return 0;

	int i = 0;
	char *buf = (char*) malloc(LINE_LEN);
	int buf_len = 0;
	char* tmp;

	while((i < len) && (fgets(buf, LINE_LEN - 1, fp) != NULL))
	{
		buf_len = strlen(buf);

		if((buf_len > 0) && (buf[buf_len - 1] == '\n'))
			buf[buf_len - 1] = '\0'; 

		// extract here
		tmp 		= strtok(buf, ",");
		mass[i] 	= STR_TO_DATA_T(tmp);

		tmp 		= strtok(NULL, ",");
		pos_x[i] 	= STR_TO_DATA_T(tmp);

		tmp 		= strtok(NULL, ",");
		pos_y[i] 	= STR_TO_DATA_T(tmp);

		tmp 		= strtok(NULL, ",");
		pos_z[i] 	= STR_TO_DATA_T(tmp);

		tmp 		= strtok(NULL, ",");
		vel_x[i] 	= STR_TO_DATA_T(tmp);

		tmp 		= strtok(NULL, ",");
		vel_y[i] 	= STR_TO_DATA_T(tmp);

		tmp 		= strtok(NULL, ",");
		vel_z[i] 	= STR_TO_DATA_T(tmp);

		i++;
	}
	free(buf);
	fclose(fp);
	return 1;
}
