#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include "cuPrintf.cu"

#define REBUILD_FREQ		5			// Rebuild after every X iterations
#define TIME_STEP 		30  			// in simulation time, in minutes
#define EXIT_COUNT		200 			// Number of iterations to do before exiting, -1 for infinite
#define FILENAME_LEN	 	256
#define ERROR 			-1 			// Generic Error val for readability
#define SECS			TIME_STEP * 60		// seconds per time step
#define GRAV_CONST	  	6.67408e-11;
#define NUM_THREADS		64
#define NUM_BLOCKS		16
#define LINE_LEN		512
#define TOL			0.05
#define GIG 			1000000000
#define MI 			1000000
#define STR_TO_DATA_T(str) strtof(str, NULL)
typedef float data_t;

int fileread_build_arrays(char* filename, data_t* h_mass, data_t* h_pos_x, data_t* h_pos_y, data_t* h_pos_z, data_t* h_vel_x, data_t* h_vel_y, data_t* h_vel_z, int len);
//__global__ void kernel_add (int arrLen, data_t* x, data_t* y, data_t* result);
__global__ void kernel_force_accum(data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* vel_x, data_t* vel_y, data_t* vel_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int num_bodies, int time);

__global__ void kernel_force_zero(data_t* x, data_t* y, data_t* z, int len);

__global__ void kernel_position_update(data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* vel_x, data_t* vel_y, data_t* vel_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int num_bodies, int time);

__global__ void kernel_velocity_update(data_t* mass, data_t* vel_x, data_t* vel_y, data_t* vel_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int num_bodies, int time);
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct timespec diff(struct timespec start, struct timespec end){
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_force_accum(data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* vel_x, data_t* vel_y, data_t* vel_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int num_bodies, int time){

	int idx = (blockIdx.x * blockDim.x + threadIdx.x);	
	//cuPrintf("%d\n",idx);
	if(idx < num_bodies){
	int i;
	data_t r_x; //r_y, r_z, r;
	data_t r_y;
	data_t r_z;
	data_t r;
	data_t F_part;
		for(i = 0; i < num_bodies; i++){
			if(i != idx){
				r_x = pos_x[idx] - pos_x[i];
				r_y = pos_y[idx] - pos_y[i];
				r_z = pos_z[idx] - pos_z[i];
	
				r = sqrt( (r_x* r_x) + (r_y * r_y) + (r_z * r_z) );

				// force 
				F_part = 6.67408e-11 * (mass[idx] * mass[i])/(r * r *r);

				fma_x[idx]  += F_part * r_x;
				fma_y[idx]  += F_part * r_y;
				fma_z[idx]  += F_part * r_z;
				//cuPrintf("TeST");
				//cuPrintf("%f\n",fma_x[idx]);
			}
		}
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_force_zero(data_t* x, data_t* y, data_t* z, int len){
	int i;
int idx = (blockIdx.x * blockDim.x + threadIdx.x);
	//for(i = 0; i < len; i++)
	//{
		x[idx] = 0;
		y[idx] = 0;
		z[idx] = 0;
	//}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_position_update(data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* vel_x, data_t* vel_y, data_t* vel_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int num_bodies, int time){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < num_bodies){
		fma_x[i] /= mass[i];
		fma_y[i] /= mass[i];
		fma_z[i] /= mass[i];

		pos_x[i] += time * (vel_x[i] + (0.5 * fma_x[i] * time)); 
		pos_y[i] += time * (vel_y[i] + (0.5 * fma_y[i] * time)); 
		pos_z[i] += time * (vel_z[i] + (0.5 * fma_z[i] * time)); 
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_velocity_update(data_t* mass, data_t* vel_x, data_t* vel_y, data_t* vel_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int num_bodies, int time){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < num_bodies){
		vel_x[i] += fma_x[i] * time;
		vel_y[i] += fma_y[i] * time;
		vel_z[i] += fma_z[i] * time;
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void force_zero(data_t* x, data_t* y, data_t* z, int len){
	int i;

	for(i = 0; i < len; i++)
	{
		x[i] = 0;
		y[i] = 0;
		z[i] = 0;
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void force_accum(data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int focus, int comp){
	//  First the distance
	data_t r_x, r_y, r_z, r;

	r_x = pos_x[focus] - pos_x[comp];
	r_y = pos_y[focus] - pos_y[comp];
	r_z = pos_z[focus] - pos_z[comp];

	r = sqrt( (r_x * r_x) + (r_y * r_y) + (r_z * r_z) );

	//  then the force for the focus

	data_t F_part;

	//F_part = FORCE_PARTIAL(GRAV_CONST, mass[focus], mass[comp], r);
	F_part = 6.67408e-11 * (mass[focus] * mass[comp])/(r * r *r);

	//printf("F_part %lf | m1 %.2lf kg | m2 %.2lf kg | r %.2lf km\n", F_part, mass[focus], mass[comp], r);

	fma_x[focus]  += F_part * r_x;
	fma_y[focus]  += F_part * r_y;
	fma_z[focus]  += F_part * r_z;

	// force for the comparison
	// we know this by Newton's 3rd law

	fma_x[comp]   += -fma_x[focus];
	fma_y[comp]   += -fma_y[focus];
	fma_z[comp]   += -fma_z[focus];
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void position_update(data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* vel_x, data_t* vel_y, data_t* vel_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int len, int time){
	//  NB, when this is invoked, fma arrays will have forces built up in force_accum()
	int i;
	for(i = 0; i < len; i++){
		// convert forces to acceleration, saves a multiply later
		fma_x[i] /= mass[i];
		fma_y[i] /= mass[i];
		fma_z[i] /= mass[i];

		pos_x[i] += time * (vel_x[i] + (0.5 * fma_x[i] * time)); 
		pos_y[i] += time * (vel_y[i] + (0.5 * fma_y[i] * time)); 
		pos_z[i] += time * (vel_z[i] + (0.5 * fma_z[i] * time)); 
		//pos_x[i] += DISPLACE(vel_x[i], fma_x[i], time);
		//pos_y[i] += DISPLACE(vel_y[i], fma_y[i], time);
		//pos_z[i] += DISPLACE(vel_z[i], fma_z[i], time);
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void velocity_update(data_t* mass, data_t* vel_x, data_t* vel_y, data_t* vel_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int len, int time){
	// NB, when this is invoked, fma arrays should be accelerations set in position_update()
	int i;

	for(i = 0; i < len; i++){
		vel_x[i] += fma_x[i] * time;
		vel_y[i] += fma_y[i] * time;
		vel_z[i] += fma_z[i] * time;
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int fileread_build_arrays(char* filename, data_t* h_mass, data_t* h_pos_x, data_t* h_pos_y, data_t* h_pos_z, data_t* h_vel_x, data_t* h_vel_y, data_t* h_vel_z, int len){
	// returns true -- false
	FILE *fp = fopen(filename, "r");

	if(fp == NULL) return 0;

	int i = 0;
	char *buf = (char*) malloc(LINE_LEN);
	int buf_len = 0;
	char* tmp;

	while((i < len) && (fgets(buf, LINE_LEN - 1, fp) != NULL)){
		buf_len = strlen(buf);

		if((buf_len > 0) && (buf[buf_len - 1] == '\n'))
			buf[buf_len - 1] = '\0'; 

		// extract here
		tmp 		= strtok(buf, ",");
		h_mass[i] 	= STR_TO_DATA_T(tmp);

		tmp 		= strtok(NULL, ",");
		h_pos_x[i] 	= STR_TO_DATA_T(tmp);

		tmp 		= strtok(NULL, ",");
		h_pos_y[i] 	= STR_TO_DATA_T(tmp);

		tmp 		= strtok(NULL, ",");
		h_pos_z[i] 	= STR_TO_DATA_T(tmp);

		tmp 		= strtok(NULL, ",");
		h_vel_x[i] 	= STR_TO_DATA_T(tmp);

		tmp 		= strtok(NULL, ",");
		h_vel_y[i] 	= STR_TO_DATA_T(tmp);

		tmp 		= strtok(NULL, ",");
		h_vel_z[i] 	= STR_TO_DATA_T(tmp);

		i++;
	}
	free(buf);
	fclose(fp);
	return 1;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]){
	
	// Arrays on GPU global memory
	cudaPrintfInit();
	cudaEvent_t start, stop, start1, stop1;
	float elapsed_gpu, elapsed_gpu1;

	//  grab and process the file from command line
	char* filename = (char*) malloc(sizeof(char) * FILENAME_LEN);
	int i, j, k, errCount = 0;
	int num_bodies = 0;
	int allocSize = 0;

	struct timespec diff(struct timespec start, struct timespec end);
  	struct timespec time1, time2;
  	struct timespec time_stamp;
  	int clock_gettime(clockid_t clk_id, struct timespec *tp);

	/////////////////////////device variables//////////////////////////////
	data_t *d_mass;   //mass array

	data_t *d_pos_x;  //position arrays
	data_t *d_pos_y;
	data_t *d_pos_z;

	data_t *d_vel_x;  //velocity arrays
	data_t *d_vel_y;
	data_t *d_vel_z;

	data_t *d_fma_x;  //force || acceleration arrays
	data_t *d_fma_y;
	data_t *d_fma_z;
	/////////////////////////host variables/////////////////////////////
	data_t *h_mass;   //mass array

	data_t *h_pos_x;  //position arrays
	data_t *h_pos_y;
	data_t *h_pos_z;

	data_t *h_vel_x;  //velocity arrays
	data_t *h_vel_y;
	data_t *h_vel_z;

	data_t *h_fma_x;  //force || acceleration arrays
	data_t *h_fma_y;
	data_t *h_fma_z;
	data_t *h_result;

	if(argc != 2){
		printf("\nERROR: Command line requires file name input!\n");
		exit(EXIT_FAILURE);
	}

	filename = argv[1];
	
	//  calculate number of bodies
	//  NB, the loop here is designed with only galaxy_####.csv as an expected name
	//  No error checking done for variety of inputs
	char* p = filename;
	
	while(*p){	//  Still more characters to process{
		if(isdigit(*p)){
			num_bodies *= 10;
			num_bodies += strtol(p, &p, 10);
		}
		p++;
	}

	printf("Num bodies: %d\n", num_bodies);
	// total size required for allocation
	allocSize = sizeof(data_t) * num_bodies;
	printf("Allocating memory on host\n");
	// Allocate memory on CPU
	h_mass  = (data_t*) malloc(allocSize);
	h_pos_x = (data_t*) malloc(allocSize);
	h_pos_y = (data_t*) malloc(allocSize);
	h_pos_z = (data_t*) malloc(allocSize);
	h_vel_x = (data_t*) malloc(allocSize);
	h_vel_y = (data_t*) malloc(allocSize);
	h_vel_z = (data_t*) malloc(allocSize);
	h_fma_x = (data_t*) malloc(allocSize);
	h_fma_y = (data_t*) malloc(allocSize);
	h_fma_z = (data_t*) malloc(allocSize);
	h_result = (data_t*) malloc(allocSize);
	//h_result = (data_t*) malloc(allocSize);

	if(!h_mass || !h_pos_x || !h_pos_y || !h_pos_z || !h_vel_x || !h_vel_y || !h_vel_z || !h_fma_x || !h_fma_y || !h_fma_z || !h_result){
		printf("ERROR: Array malloc issue!\n");
		return 0;
	}
	printf("Done!\n");

	
	// read the file
	printf("Reading file and building arrays...\n");
	fileread_build_arrays(filename, h_mass, h_pos_x, h_pos_y, h_pos_z, h_vel_x, h_vel_y, h_vel_z, num_bodies);
	//printf("%lf\n",h_pos_x[4]);	
	printf("Done!\n");
	//accelerations
	//force_zero(h_fma_x, h_fma_y, h_fma_z, num_bodies); 

	// Select GPU
        CUDA_SAFE_CALL(cudaSetDevice(0));
	
	printf("Allocating memory on GPU...\n");	
	//Allocate memory on GPU
	CUDA_SAFE_CALL(cudaMalloc((data_t**)&d_mass, allocSize));	
	CUDA_SAFE_CALL(cudaMalloc((data_t**)&d_pos_x, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((data_t**)&d_pos_y, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((data_t**)&d_pos_z, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((data_t**)&d_vel_x, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((data_t**)&d_vel_y, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((data_t**)&d_vel_z, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((data_t**)&d_fma_x, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((data_t**)&d_fma_y, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((data_t**)&d_fma_z, allocSize));
	printf("Done!\n");

	// Create the cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	// Record event on the default stream
	cudaEventRecord(start, 0);

	printf("Copying data on GPU...\n");
	//Transfer the data to GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(d_mass, h_mass, allocSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_pos_x, h_pos_x, allocSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_pos_y, h_pos_y, allocSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_pos_z, h_pos_z, allocSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_vel_x, h_vel_x, allocSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_vel_y, h_vel_y, allocSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_vel_z, h_vel_z, allocSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_fma_x, h_fma_x, allocSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_fma_y, h_fma_y, allocSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_fma_z, h_fma_z, allocSize, cudaMemcpyHostToDevice));
	printf("Done!\n");

	// Launch the kernel
    	dim3 dimBlock(NUM_THREADS, 1, 1);
	printf("dimBlock\n");
	dim3 dimGrid(NUM_BLOCKS, 1, 1);
	printf("dimGrid\n");

	//kernel_force_zero<<<dimGrid, dimBlock>>>(h_fma_x, h_fma_y, h_fma_z, num_bodies);
	cudaEventRecord(start1, 0);

	for(i = 0; i < EXIT_COUNT; i++)
	{
		//printf("%d\n",i);
		kernel_force_zero<<<dimGrid, dimBlock>>>(d_fma_x, d_fma_y, d_fma_z, num_bodies);
		kernel_force_accum<<<dimGrid, dimBlock>>>(d_mass, d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, d_fma_x, d_fma_y, d_fma_z, num_bodies, TIME_STEP);
		kernel_position_update<<<dimGrid, dimBlock>>>(d_mass, d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, d_fma_x, d_fma_y, d_fma_z, num_bodies, TIME_STEP);
		kernel_velocity_update<<<dimGrid, dimBlock>>>(d_mass, d_vel_x, d_vel_y, d_vel_z, d_fma_x, d_fma_y, d_fma_z, num_bodies, TIME_STEP);
		
	}

	//CUDA_SAFE_CALL(cudaPeekAtLastError());
	cudaPrintfDisplay(stdout, true);
  	cudaPrintfEnd();

	CUDA_SAFE_CALL(cudaMemcpy(h_result, d_pos_x, allocSize, cudaMemcpyDeviceToHost));

	//printf("kernel call done\n");
	cudaEventRecord(stop1,0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&elapsed_gpu1, start1, stop1);
	printf("\nGPU time for kernel execution: %lf (msec)\n", elapsed_gpu1);
	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);
	cudaThreadSynchronize();

	// Stop and destroy the timer
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);
	printf("\nGPU time: %lf (msec)\n", elapsed_gpu);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


///////////////////////////CPU turn////////////////////////////////////////////////
	printf("CPU: 'My turn son!' \n");
	printf("GPU: 'I am gonna come back tomorrow then, slow timer!' \n");

     	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
	for(i = 0; i < EXIT_COUNT; i++)
	{
		//printf("Position (x, y, z) of body 5: (%f, %f, %f)\n", pos_x[4], pos_y[4], pos_z[4]);
		force_zero(h_fma_x, h_fma_y, h_fma_z, num_bodies);		

		for(j = 0; j < num_bodies; j++)
		{
			for(k = j + 1; k < num_bodies; k++)
				force_accum(h_mass, h_pos_x, h_pos_y, h_pos_z, h_fma_x, h_fma_y, h_fma_z, j, k);
		}

		position_update(h_mass, h_pos_x, h_pos_y, h_pos_z, h_vel_x, h_vel_y, h_vel_z, h_fma_x, h_fma_y, h_fma_z, num_bodies, TIME_STEP);
		velocity_update(h_mass, h_vel_x, h_vel_y, h_vel_z, h_fma_x, h_fma_y, h_fma_z, num_bodies, TIME_STEP);
		
	}

     	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
     	time_stamp = diff(time1,time2);
     	printf("CPU time is %ld (msec)", (long int)(GIG * time_stamp.tv_sec + time_stamp.tv_nsec)/MI);

	int N = num_bodies;
	// Compare the results
	for(i = 0; i < N; i++) 
		{
			if (abs(h_result[i] - h_pos_x[i]) > (TOL*h_pos_x[i]))  //h_result is the output of the GPU copied to host i.e. CPU
				errCount++;
		}
	//}
		
	if (errCount > 0) {
		printf("\nERROR: TEST FAILED: %d results did not match\n", errCount);
	}
//	else if (zeroCount > 0){
//		printf("\n@ERROR: TEST FAILED: %d results (from GPU) are zero\n", zeroCount);
//	}
	else {
		printf("\nTEST PASSED: All results matched\n");
	}


}
