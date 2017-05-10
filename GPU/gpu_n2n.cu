#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include "grav_gpu.hu"
#include "grav_cpu.h"

#define TIME_STEP 			30  			// in simulation time, in minutes
#define EXIT_COUNT			200 			// Number of iterations to do before exiting
#define FILENAME_LEN	 	256
#define GRAV_CONST	  		6.67408e-11;
#define NUM_THREADS			64
#define NUM_BLOCKS			16
#define LINE_LEN			512
#define TOL					0.05
#define GIG 				1000000000
#define MI 					1000000

struct timespec diff(struct timespec start, struct timespec end)
{
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

int main(int argc, char *argv[]){
	
	// Arrays on GPU global memory
	cudaEvent_t start, stop, start1, stop1;
	float elapsed_gpu, elapsed_gpu1;

	//  grab and process the file from command line
	char* filename = (char*) malloc(sizeof(char) * FILENAME_LEN);
	int i, j, k, errCount = 0;
	int num_bodies = 0;
	int allocSize = 0;

  	struct timespec time1, time2;
  	struct timespec time_stamp;

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
	num_bodies = body_count(filename);	

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


	//  CPU verification

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


	for(i = 0; i < num_bodies; i++) 
		{
			if (abs(h_result[i] - h_pos_x[i]) > (TOL*h_pos_x[i]))  //h_result is the output of the GPU copied to host i.e. CPU
				errCount++;
		}
	//}
		
	if (errCount > 0) {
		printf("\nERROR: TEST FAILED: %d results did not match\n", errCount);
	}
	else {
		printf("\nTEST PASSED: All results matched\n");
	}


}
