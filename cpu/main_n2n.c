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

#ifdef  CSV_ACTIVE
	#define TARGET_FILE ((const char*) "results.csv")
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
		struct timespec total_start, total_end, total_elapse;  // for total execution time
		#ifdef CPE_ACTIVE
			struct timespec iter_start[EXIT_COUNT], iter_end[EXIT_COUNT];
			double 			iter_avg;
		#endif
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
		clock_gettime(TIMING_MODE, &total_start);
	#endif

	#ifdef CSV_ACTIVE
		FILE *pTarget = fopen(TARGET_FILE, "a");
	#endif

	for(i = 0; i < EXIT_COUNT; i++)
	{
		#ifdef TIMING_ACTIVE
			#ifdef CPE_ACTIVE
				clock_gettime(TIMING_MODE, &iter_start[i]);
			#endif
		#endif

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

		#ifdef TIMING_ACTIVE
			#ifdef CPE_ACTIVE
				clock_gettime(TIMING_MODE, &iter_end[i]);
			#endif
		#endif
	}

	#ifdef TIMING_ACTIVE
		clock_gettime(TIMING_MODE, &total_end);
		total_elapse = ts_diff(total_start, total_end);
		double ns = ((double) total_elapse.tv_sec) * 1.0e9 + ((double) total_elapse.tv_nsec);

		#ifdef CPE_ACTIVE
			iter_avg = 0;
			for(i = 0; i < EXIT_COUNT; i++)
				iter_avg += double_diff(iter_start[i], iter_end[i]);  //saturation issues?
			iter_avg /= EXIT_COUNT;

			#ifdef CSV_ACTIVE
				fprintf(pTarget, "%.0lf, ", CPE_calculate(iter_avg, num_bodies));
			#else
				printf("CPE : %.0lf cycles\n", CPE_calculate(iter_avg, num_bodies));
			#endif
				
		#elif  CSV_ACTIVE
			fprintf(pTarget, "%.0lf, ", ns);
		#else
			printf("Time Elapsed was %.0lf ns.\n", ns);
		#endif
	#endif

	#ifdef CSV_ACTIVE
		fclose(pTarget);
	#endif

	return 0;
}