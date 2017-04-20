#ifndef CPU_SIMPLE_H
#define CPU_SIMPLE_H

#include <math.h>
#include "../octree/octree.h"

#typedef time_t int;

const data_t GRAV_CONST = 6.674e-11;

//  Functions & Macros for the math that will be needed
#define DISTANCE(r_x, r_y, r_z) sqrt((r_x * r_x) + (r_y * r_y) + (r_z * r_z))
#define FORCE_PARTIAL(m1, m2, r) GRAV_CONST * ((m1 * m2)/(r * r * r))

// NB : expansion below is one MUL less than direct implementation
#define DISPLACE(vel, accel, time) time * (vel + (0.5 * accel * time))

int 	body_alloc(p_octant root, nbody *bodies[], int num_bodies);

void 	body_body_accum_accel(int focus, int other, p_octant oct);
void 	body_oct_accum_accel(p_octant local, int body, p_octant distal);

void	body_pos_update(int body, p_octant oct, time_t time);
void	body_vel_update(int body, p_octant oct, time_t time);

#endif