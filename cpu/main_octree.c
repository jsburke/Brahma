#ifdef THREAD_ACTIVE
	#include "cpu_octree_omp.h"
#else
	#include "cpu_octree.h"
#endif

//  Rebuild constraint
#define REBUILD_FREQ			5

//  Time based defines
#define TIME_STEP				3*TIME_HOUR  //time step to be used for calculations
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
	#ifndef	THREAD_ACTIVE
		#define TIMING_MODE				CLOCK_REALTIME  // change me for parallel
	#else	
		#define TIMING_MODE				CLOCK_PROCESS_CPUTIME_ID
	#endif
#endif

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
//			!! MAIN PROGRAM LOOP HERE !!
//			Check for rebuild of octree
//				rebuild if constraints met
//			Calculate centers of mass
//			calculate forces
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
	int i, j;
	int num_bodies = 0;
	
	#ifdef TIMING_ACTIVE
		struct timespec time_start, time_end, time_elapse;
	#endif

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
	if(!root) return 0;

	for(i = 0; i < CHILD_COUNT; i++)
	{
		root->children[i] 	= octant_new(LEVEL_1);
		if(!(root->children[i])) return 0;

		for(j = 0; j < CHILD_COUNT; j++)
			if(!(root->children[i]->children[j] = octant_new(LEVEL_2))) return 0;
	}

	//  test variables, comment if not testing
	//octant *test  = root->children[4]->children[3];
	//int test_leaf = 0;

	if(!fileread_build_tree(filename, root, num_bodies))
	{
		printf("ERROR:  Reading file failed!\n");
		return 0;
	}

	#ifdef THREAD_ACTIVE
		omp_set_dynamic(0);
		omp_set_num_threads(NUM_THREADS);
	#endif


	//printf("Body %d in octant(4, 3) has mass %.2lf kg and is at position (%.2lf, %.2lf, %.2lf).\n", test_leaf, test->mass[test_leaf], test->pos_x[test_leaf], test->pos_y[test_leaf], test->pos_z[test_leaf]);

	/////////////////
	//
	//  PART 2
	//
	/////////////////

	int check = 0;

	#ifdef TIMING_ACTIVE
		measure_cps();
		clock_gettime(TIMING_MODE, &time_start);
	#endif

	for(i = 0; i < EXIT_COUNT; i++)
	{
		//printf("iter: %d\n\n", i);
		if((i % REBUILD_FREQ) == 0)
			check = octree_rebuild(root);

		if(!check)
		{
			printf("ERROR: Octree Rebuild caused error, iteration %d\n", i);
			return 0;
		}

		force_zero(root);

		//printf("Body %d in octant(4, 3) has mass %.2lf kg and is at position (%.2lf, %.2lf, %.2lf).\n", test_leaf, test->mass[test_leaf], test->pos_x[test_leaf], test->pos_y[test_leaf], test->pos_z[test_leaf]);

		center_of_mass_update(root);
		force_accum(root);
		position_update(root, TIME_STEP);		
		velocity_update(root, TIME_STEP);
	}

	#ifdef TIMING_ACTIVE
		clock_gettime(TIMING_MODE, &time_end);
		time_elapse = ts_diff(time_start, time_end);
		double ns = ((double) time_elapse.tv_sec) * 1.0e9 + ((double) time_elapse.tv_nsec);
		printf("Time Elapsed was %lf ns.\n", ns);
	#endif

	return 0;
}