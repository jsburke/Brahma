#include "cpu_octree.h"

//  Rebuild constraint
#define REBUILD_FREQ			5

//  Time based defines
#ifndef TIME_STEP
	#define TIME_STEP			3*TIME_HOUR  //time step to be used for calculations
#endif
//  General use
#define TIME_MIN 				60			// lowest granualarity is second
#define TIME_HOUR				60*TIME_MIN
#define TIME_DAY				24*TIME_HOUR
#define TIME_MONTH				30*TIME_DAY
#define TIME_YEAR				365*TIME_DAY

#ifndef EXIT_COUNT
	#define EXIT_COUNT			200			//  number of iterations in loop
#endif

#define FILENAME_LEN			256

//  following structures depend on defines passed in at compile time

#ifdef TEST_PRINT
	#define TEST_MAJOR			1			//  Major octant for test printing
	#define TEST_MINOR			6 			//	Minor octant (suboctant) for prints
	#define TEST_LEAF 			0 			//  Leaf to test print
#endif

#ifdef	TIMING_ACTIVE
	#include "timing.h"
	#ifdef	THREAD_ACTIVE
		#define TIMING_MODE				CLOCK_REALTIME  
	#else	
		#define TIMING_MODE				CLOCK_PROCESS_CPUTIME_ID
	#endif
#endif

#ifdef  CSV_ACTIVE
	#define TARGET_FILE ((const char*) "results.csv")
#endif

void payload_calculations(octant *root);  //mainly used to help parallelism

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
		struct timespec total_start, total_end, total_elapse;
		#ifdef CPE_ACTIVE
			struct timespec iter_start[EXIT_COUNT], iter_end[EXIT_COUNT];
			double 			iter_avg;
		#endif
	#endif

	time_set_up(TIME_STEP);

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
	#ifdef TEST_PRINT
	  octant *test  = root->children[TEST_MAJOR]->children[TEST_MINOR];
	#endif

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

	#ifdef CSV_ACTIVE
		FILE *pTarget = fopen(TARGET_FILE, "a");
	#endif

	#ifdef TIMING_ACTIVE
		measure_cps();
		clock_gettime(TIMING_MODE, &total_start);
	#endif

	for(i = 0; i < CHILD_COUNT; i++)
		center_of_mass_update(root, i);

	for(i = 0; i < EXIT_COUNT; i++)
	{
		#ifdef TIMING_ACTIVE
			#ifdef CPE_ACTIVE
				clock_gettime(TIMING_MODE, &iter_start[i]);
			#endif
		#endif

		//printf("iter: %d\n\n", i);
		if((i % REBUILD_FREQ) == 0)
			check = octree_rebuild(root);

		if(!check)
		{
			printf("ERROR: Octree Rebuild caused error, iteration %d\n", i);
			return 0;
		}

		#ifdef TEST_PRINT
			printf("Body %d in octant(%d, %d) has mass %.2lf kg and is at position (%.2lf, %.2lf, %.2lf).\n", TEST_LEAF, TEST_MAJOR, TEST_MINOR, test->mass[TEST_LEAF], test->pos_x[TEST_LEAF], test->pos_y[TEST_LEAF], test->pos_z[TEST_LEAF]);
		#endif

		payload_calculations(root);
		//center_of_mass_update(root);  // can try to parallelize later

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

void payload_calculations(octant *root)
{
	int i;

	#ifdef THREAD_ACTIVE
		#pragma omp parallel for
	#endif
	for(i = 0; i < CHILD_COUNT; i++)
	{
		force_zero(root, i);
		force_accum(root, i);
		#ifdef THREAD_ACTIVE
			#pragma omp barrier
		#endif
		position_update(root, i);		
		velocity_update(root, i);
		center_of_mass_update(root, i);
	}
}