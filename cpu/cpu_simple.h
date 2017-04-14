#ifndef CPU_SIMPLE_H
#define CPU_SIMPLE_H

#include <math.h>
#include "../octree/octree.h"

//  Functions & Macros for the math that will be needed
#define BODY_BODY_DISTANCE(r_x, r_y, r_z) sqrt((r_x * r_x) + (r_y * r_y) + (r_z * r_z))
data_t 	body_body_partial_force(int focus, int other, p_octant oct);
void 	body_body_accum_accel(int focus, int other, p_octant oct);

data_t 	body_oct_distance(int body, p_octant oct);
data_t 	body_oct_force(int body, p_octant oct);
//  not doing accumulation of acceleration since we only need three divides and accums

#endif