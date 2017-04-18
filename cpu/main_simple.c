#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "cpu_simple.h"

#define REBUILD_FREQ			5				// Rebuild after every X iterations
#define TIME_STEP 				30  			// in simulation time, in minutes
#define EXIT_COUNT				200 			// Number of iterations to do before exiting, -1 for infinite
#define FILENAME_LEN 			256
#define ERROR 					-1 				// Generic Error val for readability

#define SECS					TIME_STEP * 60	// seconds per time step

////////////////////////////////////////////////////////////////////
//
// Inputs: a file name of the form galaxy_###.csv, where the number
//		   portion may be arbitrarily long
//
// Program Outline
//
// Step 1:
//			Read the input file and generate an array of bodies
// Step 2:
//			Build tree from octant structs
//			Populate the Level 2 nodes with leaves, AKA suns, moons, &c
//			Calculate non-Root node mass attributes
// Step 3:
//			!! MAIN PROGRAM LOOP HERE !!
//			Check for rebuild of octree
//				rebuild if constraints met
//			calculate accelerations
//			update position and velocity arrays in L2 octants
//			zero out accelerations
//
////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
	/////////////////
	//
	// data_t check
	//
	/////////////////

	if(MAX_POS_POSITION == DATA_T_ERR)
	{
		printf("\nERROR: data_t not defined properly!\n");
		return 0;
	}

	/////////////////
	//
	//  PART 1
	//
	/////////////////

	//  grab and process the file from command line
	char*		filename = (char*) malloc(sizeof(char) * FILENAME_LEN);
	nbody*		bodies;
	int 		i, j, k;
	int 		num_bodies = 0;

	if(argc != 2)
	{
		printf("\nERROR: Comman line requires file name input!\n");
		exit(EXIT_FAILURE);
	}

	filename = argv[1];

	//  calculate of bodies
	//  NB, the loop here is designed with only galaxy_####.csv as an expected name
	//  No error checking done for variety of inputs
	char* p = filename;
	while(*p)	//  Still more characters to process
	{
		if(isdigit(*p))
		{
			num_bodies *= 10;
			num_bodies += strtol(p, &p, 10);
		}
		p++;
	}

	//  get bodies from file
	bodies = (nbody*) malloc(sizeof(nbody) * num_bodies);
	if( KILL == nbody_enum(bodies, filename)) return 0;  // exit on failure
	free(filename);  //  file will no longer be accessed

	/////////////////
	//
	//  PART 2
	//
	/////////////////

	p_octant root_oct = octant_new(-1, ROOT);
	if(!root_oct) return 0;

	p_octant root_children = root_oct->children;  // store pointer to reduce one layer of chasing

	// don't like the below loop, feels like it encourages pointer chasing
	for(i = 0; i < CHILD_COUNT; i++)
	{
		if(NULL == (root_children[i] = octant_new(i, LVL_1))) return 0;

		for(j = 0; j < CHILD_COUNT; j++)
			if(NULL == (root_children[i]->children[j] = octant_new(j, LVL_2))) return 0;
	}

	//  place the leaves in suboctants
	//  subroutine prosepctively designed
	data_t upper_x, lower_x, half_x upper_y, lower_y, half_y upper_z, lower_z, half_z;
	int octant, suboctant;
	data_t body_x, body_y, body_z;

	for (i = 0; i < num_bodies; i++)
	{
		octant 		= 0;
		suboctant 	= 0;

		body_x = bodies[i]->pos_x;
		body_y = bodies[i]->pos_y;
		body_z = bodies[i]->pos_z;

		// find location

		if(body_x >= 0) // even octant
		{
			octant 	+= 1;
			upper_x = MAX_POS_POSITION;
			half_x  = POS_QUARTER_MARK;
			lower_x = 0;
		}
		else
		{
			upper_x = 0;
			half_x  = NEG_QUARTER_MARK;
			lower_x = MAX_NEG_POSITION;
		}
		suboctant += (body_x >= half_x) 1 : 0;

		if(body_y >= 0)
		{
			octant 	+= 2;
			upper_y = MAX_POS_POSITION;
			half_y  = POS_QUARTER_MARK;
			lower_y = 0;
		}
		else
		{
			upper_y = 0;
			half_y  = NEG_QUARTER_MARK;
			lower_y = MAX_NEG_POSITION;
		}
		suboctant += (body_y >= half_y) 2 : 0;

		if(body_z >= 0)
		{
			octant 	+= 4;
			upper_z = MAX_POS_POSITION;
			half_z  = POS_QUARTER_MARK;
			lower_z = 0;
		}
		else
		{
			upper_z = 0;
			half_z  = NEG_QUARTER_MARK;
			lower_z = MAX_NEG_POSITION;
		}
		suboctant += (body_z >= half_z) 4 : 0;

		if(KILL == octant_add_body(root_children[octant]->children[suboctant], bodies[i]))
		{
			return 0;
		}
	}

	free(bodies);

	//  Center of mass calculations
	//  See below note on DRY issue here
	for(i = 0; i < CHILD_COUNT; i++)
	{
		// center of mass for LVL 2 nodes
		for(j = 0; j < CHILD_COUNT; j++)
			octant_center_of_mass(root_children[i]->children[j]);

		//  center of mass for parent of what was just done
		octant_center_of_mass(root_children[i]);
	}

	/////////////////
	//
	//  PART 3
	//
	/////////////////

	p_octant oct_focus;
	int index, comp, leaf_count, check;

	for(i = 0; i < EXIT_COUNT; i++)
	{
		if((i % REBUILD_FREQ) == 0) // time to rebuild tree!
		{
			check = octree_rebuild(root);
			if(KILL == check) return 0;
		}

		// calculate & clear accelerations
		// update position and velocities
		for(j = 0; j < CHILD_COUNT; j++)
		{
			for(k = 0; k < CHILDREN_PER_OCTANT; k++)
			{
				oct_focus  = root_children[j]->children[k];
				leaf_count = focus->leaf_count;

				// accumulate accelerations
				for(index = 0; index < leaf_count; index++)
				{
					// first bodies in the suboctant
					for(comp = index + 1; comp < leaf_count; comp++)
						body_body_accum_accel(index, comp, oct_focus);

					// then to local suboctants
					for(comp = 0; comp < CHILDREN_PER_OCTANT; comp++)
						if(comp != k) body_oct_accum_accel(oct_focus, index, root_children[j]->children[comp]);

					// then for distal LVL 1 octants
					for(comp = 0; comp < CHILDREN_PER_OCTANT; comp++)
						if(comp != j) body_oct_accum_accel(oct_focus, index, root_children[comp]);
				}

				//  update positions & velocities
				for(index = 0; index < leaf_count; index++)
				{
					body_pos_update(index, oct_focus, SECS);
					body_vel_update(index, oct_focus, SECS);
				}

				// clear accelerations
				octant_acceleration_zero(oct_focus);
			}
		}

		//recalculate centers of mass
		//  reevaluate loop
		//  NOT DRY with preloop
		for(j = 0; j < CHILD_COUNT; j++)
		{
			for(k = 0; k < CHILDREN_PER_OCTANT; k++)
				octant_center_of_mass(root_children[j]->children[k]);

			octant_center_of_mass(root_children[j]);
		}
	}

	return 0;
}