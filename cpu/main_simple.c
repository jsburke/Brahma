#include "cpu_simple.h"

#define REBUILD_FREQ	5		// Rebuild after every X iterations
#define TIME_STEP 		30  	// in simulation time, in minutes
#define EXIT_COUNT		200 	// Number of iterations to do before exiting, maybe 0 or -1 for infinite

////////////////////////////////////////////////////////////////////
//
// Program Outline
//
// Step 1:
//			Read the input file and generate an array of bodies
// Step 2:
//			Build tree from octant structs from the bottom to top
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