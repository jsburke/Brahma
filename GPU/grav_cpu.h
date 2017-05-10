#ifndef GRAV_CPU_H
#define GRAV_CPU_H

#include <math.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef double data_t;
#define DATA_T_DOUBLE 1  //  done for conditional compile, change if data_t is change

#ifdef DATA_T_FLOAT		//  conditional compiles for data_t resolutions
	#define SQRT(x) sqrtf(x)
	#define STR_TO_DATA_T(str) strtof(str, NULL)
#elif  DATA_T_DOUBLE
	#define SQRT(x) sqrt(x)
	#define STR_TO_DATA_T(str) strtod(str, NULL)
#endif

int 	body_count(char* filename);  //const char* ???

void	force_zero(data_t* x, data_t* y, data_t* z, int len);
void	force_accum(data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int focus, int comp);

void	position_update(data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* vel_x, data_t* vel_y, data_t* vel_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int len, int time);
void	velocity_update(data_t* mass, data_t* vel_x, data_t* vel_y, data_t* vel_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int len, int time);

int 	fileread_build_arrays(char* filename, data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* vel_x, data_t* vel_y, data_t* vel_z, int len);
#endif