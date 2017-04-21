#ifndef OCTREE_H
#define OCTREE_H

#include <stdlib.h>

typedef double data_t;
#define DATA_T_DOUBLE		1
//#define DATA_T_FLOAT		1

#define CHILD_COUNT 		8

//levels if needed
#define ROOT 				0
#define LEVEL_1				1
#define LEVEL_2				2

//  octant struct and functions

typedef struct octant
{
	// center of mass for the octant
	data_t 	mass_center_x;
	data_t 	mass_center_y;
	data_t 	mass_center_z;
	data_t 	mass_total;

	// children
	struct octant *children[CHILD_COUNT];  //  NULL if level above planets

	// leaf arrays
	// malloc to create array
	// NULL if no leafs
	int 	leaf_count;  // index that will be written to, ie 2 means add 3rd body next or remove 2nd
	data_t 	*mass;

	data_t 	*pos_x;
	data_t 	*pos_y;
	data_t 	*pos_z;

	data_t 	*vel_x;
	data_t 	*vel_y;
	data_t 	*vel_z;

	data_t 	*acc_x;
	data_t 	*acc_y;
	data_t 	*acc_z;
} octant;

octant*		octant_new(int lvl);

void 		center_of_mass_update(octant* root);

#endif