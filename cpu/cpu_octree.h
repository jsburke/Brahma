#ifndef CPU_OCTREE_H
#define CPU_OCTREE_H

#include <math.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include "octree.h"

#ifdef DATA_T_FLOAT		//  conditional compiles for data_t resolutions
	#define SQRT(x) sqrtf(x)
	#define STR_TO_DATA_T(str) strtof(str, NULL)
#elif  DATA_T_DOUBLE
	#define SQRT(x) sqrt(x)
	#define STR_TO_DATA_T(str) strtod(str, NULL)
#endif

#ifdef THREAD_ACTIVE
	#include <omp.h>
	#define NUM_THREADS		32
#endif

#ifdef VECTOR_ACTIVE
	#include <immintrin.h>
	#include <xmmintrin.h>
	#include <smmintrin.h>
#endif

//  Functions & Macros for the math that will be needed
#define DISTANCE(r_x, r_y, r_z) SQRT((r_x * r_x) + (r_y * r_y) + (r_z * r_z))
#define FORCE_PARTIAL(g, m1, m2, r) g*((m1 * m2)/(r * r * r))
// NB : expansion below is one MUL less than direct implementation
#define DISPLACE(vel, accel, time, half_time) time * (vel + (accel * half_time))

void	time_set_up(data_t timestep);
int 	body_count(char* filename);
int 	fileread_build_tree(char* filename, octant *root, int len);

void	force_zero(octant* root);
void	force_accum(octant* root);

void	position_update(octant* root);
void	velocity_update(octant* root);



#endif