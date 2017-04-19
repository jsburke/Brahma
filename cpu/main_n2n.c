#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cpu_n2n.h"

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

const data_t GRAV_CONST = 6.674e-11;

////////////////////////////////////////////////////////////////////
//
// Inputs: a file name of the form galaxy_###.csv, where the number
//		   portion may be arbitrarily long
//
// Program Outline
//
// Step 1:
//			Read the input file and build arrays for calculations
// Step 2:
//			!! MAIN PROGRAM LOOP HERE !!
//			calculate forces
//			update position and velocity arrays
//			zero out accelerations
//
////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
	////////////////
	//
	//  PART 1
	//
	////////////////

	char 		*filename = (char*) malloc(sizeof(char) * FILENAME_LEN);
	int i, j, k;
	int num_bodies = 0;

	if(argc != 2)
	{
		printf("ERROR: Command line requires file name input!\n");
		return 0;
	}

	filename  = argv[1];
	num_bodies = body_count(filename);
	//free(filename);
	printf("Num bodies: %d\n", num_bodies);

	//  generate the arrays we need

	return 0;
}