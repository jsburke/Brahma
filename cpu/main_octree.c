#include "cpu_octree.h"

//  Time based defines
#define TIME_STEP				3*TIME_DAY  //time step to be used for calculations
//  General use
#define TIME_MIN 				60			// lowest granualarity is second
#define TIME_HOUR				60*TIME_MIN
#define TIME_DAY				24*TIME_HOUR
#define TIME_MONTH				30*TIME_DAY
#define TIME_YEAR				365*TIME_DAY

#define EXIT_COUNT				200			//  number of iterations in loop
#define FILENAME_LEN			256

////////////////////////////////////////////////////////////////////
//
// Inputs: a file name of the form galaxy_###.csv, where the number
//		   portion may be arbitrarily long
//
// Program Outline
//
// Step 1:
//			Get number of bodies from input file
//			Generate Tree Skeleton
//			Read file while populating suboctants
// Step 2:
//			Calculate centers of mass
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
	//  PART 1
	//
	/////////////////

	char 		*filename = (char*) malloc(sizeof(char) * FILENAME_LEN);
	int i, j, k;
	int num_bodies = 0;

	if(argc != 2)
	{
		printf("ERROR: Command line requires file input!\n");
		return 0;
	}

	filename   = argv[1];
	num_bodies = body_count(filename);
	//printf("Number of bodies: %d\n", num_bodies);

	//  Generate the empty tree

	octant *root 			 			 = octant_new(ROOT);

	for(i = 0; i < CHILD_COUNT; i++)
	{
		root->children[i] 	= octant_new(LEVEL_1);
		for(j = 0; j < CHILD_COUNT; j++)
			root->children[i]->children[j] = octant_new(LEVEL_2);
	}

	return 0;
}