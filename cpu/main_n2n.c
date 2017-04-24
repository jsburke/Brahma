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

#define TIMING_ACTIVE			1 			//  comment me out to disable timing in compile

#ifdef	TIMING_ACTIVE
	#include "timing.h"
	#define TIMING_MODE				CLOCK_PROCESS_CPUTIME_ID  // change me for parallel
#endif

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

	#ifdef TIMING_ACTIVE
		struct timespec time_start, time_end, time_elapse;
	#endif

	// for doing the calculations over
	data_t *mass;   //mass array

	data_t *pos_x;  //position arrays
	data_t *pos_y;
	data_t *pos_z;

	data_t *vel_x;	 //velocity arrays
	data_t *vel_y;
	data_t *vel_z;

	data_t *fma_x;  //force || acceleration arrays
	data_t *fma_y;
	data_t *fma_z;

	if(argc != 2)
	{
		printf("ERROR: Command line requires file name input!\n");
		return 0;
	}

	filename  = argv[1];
	num_bodies = body_count(filename);
	//free(filename);

	//  generate the arrays we need
	mass  = (data_t*) malloc(sizeof(data_t) * num_bodies);

	pos_x = (data_t*) malloc(sizeof(data_t) * num_bodies);
	pos_y = (data_t*) malloc(sizeof(data_t) * num_bodies);
	pos_z = (data_t*) malloc(sizeof(data_t) * num_bodies);

	vel_x = (data_t*) malloc(sizeof(data_t) * num_bodies);
	vel_y = (data_t*) malloc(sizeof(data_t) * num_bodies);
	vel_z = (data_t*) malloc(sizeof(data_t) * num_bodies);

	fma_x = (data_t*) malloc(sizeof(data_t) * num_bodies);
	fma_y = (data_t*) malloc(sizeof(data_t) * num_bodies);
	fma_z = (data_t*) malloc(sizeof(data_t) * num_bodies);

	if(!mass || !pos_x || !pos_y || !pos_z || !vel_x || !vel_y || !vel_z || !fma_x || !fma_y || !fma_z)
	{
		printf("ERROR: Array malloc issue!\n");
		return 0;
	}

	if(!fileread_build_arrays(filename, mass, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, num_bodies))
	{
		printf("ERROR: file read failed!\n");
		return 0;
	}

	////////////////
	//
	//  PART 2
	//
	////////////////

	#ifdef TIMING_ACTIVE
		measure_cps();
		clock_gettime(TIMING_MODE, &time_start);
	#endif

	for(i = 0; i < EXIT_COUNT; i++)
	{
		//printf("Position (x, y, z) of body 5: (%f, %f, %f)\n", pos_x[4], pos_y[4], pos_z[4]);
		force_zero(fma_x, fma_y, fma_z, num_bodies);		

		for(j = 0; j < num_bodies; j++)
		{
			for(k = j + 1; k < num_bodies; k++)
				force_accum(mass, pos_x, pos_y, pos_z, fma_x, fma_y, fma_z, j, k);
		}

		position_update(mass, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, fma_x, fma_y, fma_z, num_bodies, TIME_STEP);
		velocity_update(mass, vel_x, vel_y, vel_z, fma_x, fma_y, fma_z, num_bodies, TIME_STEP);
		//  if we get graphics in, update screen here
	}

	#ifdef TIMING_ACTIVE
		clock_gettime(TIMING_MODE, &time_end);
		time_elapse = ts_diff(time_start, time_end);
		double ns = ((double) time_elapse.tv_sec) * 1.0e9 + ((double) time_elapse.tv_nsec);
		printf("Time Elapsed was %lf ns.\n", ns);
	#endif

	return 0;
}