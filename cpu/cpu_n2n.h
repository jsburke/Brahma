#ifndef CPU_N2N_H
#define CPU_N2N_H

#include <math.h>
#include <ctype.h>
#include <stdlib.h>

typedef float data_t;

//  Functions & Macros for the math that will be needed
#define DISTANCE(r_x, r_y, r_z) sqrt((r_x * r_x) + (r_y * r_y) + (r_z * r_z))
#define FORCE_PARTIAL(g, m1, m2, r) ((m1 * m2)/(r * r * r))

// NB : expansion below is one MUL less than direct implementation
#define DISPLACE(vel, accel, time) time * (vel + (0.5 * accel * time))

int 	body_count(char* filename);  //const char* ???
void	force_zero(data_t* x, data_t* y, data_t* z, int len);

#endif