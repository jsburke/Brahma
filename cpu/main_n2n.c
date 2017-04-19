

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
	//  data_t
	//
	////////////////

	if(MAX_POS_POSITION == DATA_T_ERR)
	{
		printf("\nERROR: data_t not defined properly!\n");
		return 0;
	}

	////////////////
	//
	//  PART 1
	//
	////////////////
	return 0;
}