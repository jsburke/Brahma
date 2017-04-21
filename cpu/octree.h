#ifndef OCTREE_H
#define OCTREE_H

#include <stdlib.h>
#include <stdio.h>
#include <float.h>  // for limits

typedef double data_t;
#define DATA_T_DOUBLE		1
//#define DATA_T_FLOAT		1
#define DATA_T_ERR 			-1

#define CHILD_COUNT 		8

//levels if needed
#define ROOT 				0
#define LEVEL_1				1
#define LEVEL_2				2

#define AREA_CAPACITY		10000

//  below sets the range that he objects can exist in
//  restricted based on what data_t 
#ifdef  DATA_T_FLOAT
	#define POSITION_MAX_POS 	FLT_MAX
#elif 	DATA_T_DOUBLE
	#define POSITION_MAX_POS	DBL_MAX
#else
	#define POSITION_MAX_POS	DATA_T_ERR  // we really screwed up hard
#endif

#define POSITION_MAX_NEG	-POSITION_MAX_POS
#define POS_QUARTER_MARK	POSITION_MAX_POS/2.0 // use 2.0 for float and double data_t
#define NEG_QUARTER_MARK 	-POS_QUARTER_MARK

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

	data_t 	*fma_x;
	data_t 	*fma_y;
	data_t 	*fma_z;
} octant;

typedef struct pair  // for parent - child relations
{
	int 	parent;
	int 	child;
} pair;

octant*		octant_new(int lvl);
int			octree_rebuild(octant* root);

void 		center_of_mass_update(octant* root);

pair 		octant_locate(data_t x, data_t y, data_t z);
int 		octant_add_body(octant* root, int major, int minor, data_t mass, data_t pos_x, data_t pos_y, data_t pos_z, data_t vel_x, data_t vel_y, data_t vel_z);

void		force_zero(octant* root);

#endif