#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "cpu_simple.h"

#define REBUILD_FREQ			5		// Rebuild after every X iterations
#define TIME_STEP 				30  	// in simulation time, in minutes
#define EXIT_COUNT				200 	// Number of iterations to do before exiting, maybe 0 or -1 for infinite
#define FILENAME_LEN 			256

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

	return 0;
}