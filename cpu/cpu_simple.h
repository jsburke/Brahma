#ifndef CPU_SIMPLE_H
#define CPU_SIMPLE_H

#include "../octree/octree.h"

//  Functions for the math that will be needed
data_t 		body_body_distance(int focus, int other, p_octant oct);
data_t 		body_body_force(int focus, int other, p_octant oct);
void 		body_body_accum_accel(int focus, int other, p_octant oct);

data_t 		body_oct_distance(int body, p_octant oct);
data_t 		body_oct_force(int body, p_octant oct);
//  not doing accumulation of acceleration since we only need three divides and accums

#endif