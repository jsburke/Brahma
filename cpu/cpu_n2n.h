#ifndef CPU_N2N_H
#define CPU_N2N_H

#include <math.h>
#include <ctype.h>
#include <stdlib.h>

typedef float data_t;

//  Functions & Macros for the math that will be needed
#define DISTANCE(r_x, r_y, r_z) sqrt((r_x * r_x) + (r_y * r_y) + (r_z * r_z))
#define FORCE_PARTIAL(g, m1, m2, r) g*((m1 * m2)/(r * r * r))

// NB : expansion below is one MUL less than direct implementation
#define DISPLACE(vel, accel, time) time * (vel + (0.5 * accel * time))

int 	body_count(char* filename);  //const char* ???

void	force_zero(data_t* x, data_t* y, data_t* z, int len);
void	force_accum(data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* frc_x, data_t* frc_y, data_t* frc_z, int focus, int comp);

void	position_update(data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* vel_x, data_t* vel_y, data_t* vel_z, data_t* frc_x, data_t* frc_y, data_t* frc_z, int len);
void	velocity_update(data_t* mass, data_t* vel_x, data_t* vel_y, data_t* vel_z, data_t* frc_x, data_t* frc_y, data_t* frc_z, int len);

#endif