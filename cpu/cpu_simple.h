#ifndef CPU_SIMPLE_H
#define CPU_SIMPLE_H

#include <math.h>
#include "../octree/octree.h"

const data_t GRAV_CONST = 6.674e-11;

//  Functions & Macros for the math that will be needed
#define DISTANCE(r_x, r_y, r_z) sqrt((r_x * r_x) + (r_y * r_y) + (r_z * r_z))
#define FORCE_PARTIAL(m1, m2, r) GRAV_CONST * ((m1 * m2)/(r * r * r))

void 	body_body_accum_accel(int focus, int other, p_octant oct);
void 	body_oct_accum_accel(p_octant local, int body, p_octant distal);

void	body_pos_update(int body, p_octant oct, int time);
void	body_vel_update(int body, p_octant oct, int time);
//  not doing accumulation of acceleration since we only need three divides and accums

#endif