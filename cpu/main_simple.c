#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "cpu_simple.h"

#define REBUILD_FREQ			5		// Rebuild after every X iterations
#define TIME_STEP 				30  	// in simulation time, in minutes
#define EXIT_COUNT				200 	// Number of iterations to do before exiting, maybe 0 or -1 for infinite
#define FILENAME_LEN 			256
#define ERROR 					-1 		// Generic Error val for readability

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
	int 		i, j;
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
	for(i = 0; i < 8; i++)
	{
		if(NULL == (root_children[i] = octant_new(i, LVL_1))) return 0;

		for(j = 0; j < 8; j++)
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

	return 0;
}