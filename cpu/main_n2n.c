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

	// for doing the calculations over
	data_t *mass;   //mass array

	data_t *pos_x;  //position arrays
	data_t *pos_y;
	data_t *pos_z;

	data_t *vel_x;	 //velocity arrays
	data_t *vel_y;
	data_t *vel_z;

	data_t *frc_x;  //force arrays
	data_t *frc_y;
	data_t *frc_z;

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
	mass  = (data_t*) malloc(sizeof(data_t) * num_bodies);

	pos_x = (data_t*) malloc(sizeof(data_t) * num_bodies);
	pos_y = (data_t*) malloc(sizeof(data_t) * num_bodies);
	pos_z = (data_t*) malloc(sizeof(data_t) * num_bodies);

	vel_x = (data_t*) malloc(sizeof(data_t) * num_bodies);
	vel_y = (data_t*) malloc(sizeof(data_t) * num_bodies);
	vel_z = (data_t*) malloc(sizeof(data_t) * num_bodies);

	frc_x = (data_t*) malloc(sizeof(data_t) * num_bodies);
	frc_y = (data_t*) malloc(sizeof(data_t) * num_bodies);
	frc_z = (data_t*) malloc(sizeof(data_t) * num_bodies);

	force_zero(frc_x, frc_y, frc_z, num_bodies);

	return 0;
}