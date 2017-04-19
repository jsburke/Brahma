#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "cpu_simple.h"
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>

#define REBUILD_FREQ			5			// Rebuild after every X iterations
#define TIME_STEP 			30  			// in simulation time, in minutes
#define EXIT_COUNT			200 			// Number of iterations to do before exiting, -1 for infinite
#define FILENAME_LEN 			256
#define ERROR 				-1 			// Generic Error val for readability

#define SECS				TIME_STEP * 60		// seconds per time step
#define G    				6.67408e-11;
typedef float data_t;


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

int main(int argc, char *argv[])
{
	// data_t check
	if(MAX_POS_POSITION == DATA_T_ERR)
	{
		printf("\nERROR: data_t not defined properly!\n");
		return 0;
	}

	//  grab and process the file from command line
	char*		filename = (char*) malloc(sizeof(char) * FILENAME_LEN);
	//nbody*		h_bodies, d_bodies;
	int 		i, j, k;
	int 		num_bodies = 0;
	int 		allocSize = 0;
	//float* 		h_result, d_result;

	/////////////////////////device variables//////////////////////////////
	data_t *d_mass;   //mass array

	data_t *d_pos_x;  //position arrays
	data_t *d_pos_y;
	data_t *d_pos_z;

	data_t *d_vel_x;	 //velocity arrays
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

	data_t *h_vel_x;	 //velocity arrays
	data_t *h_vel_y;
	data_t *h_vel_z;

	data_t *h_fma_x;  //force || acceleration arrays
	data_t *h_fma_y;
	data_t *h_fma_z;

	if(argc != 2)
	{
		printf("\nERROR: Comman line requires file name input!\n");
		exit(EXIT_FAILURE);
	}

	filename = argv[1];
	
	//  calculate number of bodies
	//  NB, the loop here is designed with only galaxy_####.csv as an expected name
	//  No error checking done for variety of inputs
	char* p = filename;
	while(*p)	//  Still more characters to process
	{
		if(isdigit(*p))
		{
			num_bodies *= 10;
			num_bodies += strtol(p, &p, 10);
		}
		p++;
	}
	printf("Num bodies: %d\n", num_bodies);
	allocSize = sizeof(nbody) * num_bodies;

	//get bodies from file
/*	h_bodies = (nbody*) malloc(allocSize);
	h_result = (nbody*) malloc(allocSize);
	CUDA_SAFE_CALL(cudaMalloc((nbody*)&d_bodies, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((nbody*)&d_result, allocSize));

	//Transfer the bodies to GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(d_bodies, h_bodies, allocSize, cudaMemcpyHostToDevice));
*/

	h_mass  = (data_t*) malloc(sizeof(data_t) * num_bodies);

	h_pos_x = (data_t*) malloc(sizeof(data_t) * num_bodies);
	h_pos_y = (data_t*) malloc(sizeof(data_t) * num_bodies);
	h_pos_z = (data_t*) malloc(sizeof(data_t) * num_bodies);

	h_vel_x = (data_t*) malloc(sizeof(data_t) * num_bodies);
	h_vel_y = (data_t*) malloc(sizeof(data_t) * num_bodies);
	h_vel_z = (data_t*) malloc(sizeof(data_t) * num_bodies);

	h_fma_x = (data_t*) malloc(sizeof(data_t) * num_bodies);
	h_fma_y = (data_t*) malloc(sizeof(data_t) * num_bodies);
	h_fma_z = (data_t*) malloc(sizeof(data_t) * num_bodies);

	//Allocate memory on GPU
	CUDA_SAFE_CALL(cudaMalloc((data_t*)&d_mass, allocSize));
	
	CUDA_SAFE_CALL(cudaMalloc((data_t*)&d_pos_x, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((data_t*)&d_pos_y, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((data_t*)&d_pos_z, allocSize));

	CUDA_SAFE_CALL(cudaMalloc((data_t*)&d_vel_x, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((data_t*)&d_vel_y, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((data_t*)&d_vel_z, allocSize));

	CUDA_SAFE_CALL(cudaMalloc((data_t*)&d_fma_x, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((data_t*)&d_fma_y, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((data_t*)&d_fma_z, allocSize));

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
	
	if( KILL == nbody_enum(bodies, filename)) return 0;  // exit on failure
	free(filename);  //  file will no longer be accessed

	// Launch the kernel
        dim3 dimBlock(16,16,1);
	dim3 dimGrid(1,1,1);
	kernel_calculate_forces<<<dimGrid, dimBlock>>>(d_bodies, d_result, num_bodies);	
	
	// Check for errors during launch
	CUDA_SAFE_CALL(cudaPeekAtLastError());
	
	// Transfer the results back to the host
	CUDA_SAFE_CALL(cudaMemcpy(h_result, d_result, allocSize, cudaMemcpyDeviceToHost));

	// Free-up device and host memory
	CUDA_SAFE_CALL(cudaFree(d_bodies));
	CUDA_SAFE_CALL(cudaFree(d_result));
		   
	free(h_bodies);
	free(h_result);

	return 0;
}


__global__ void kernel_forceCalc(nbody* bodies,float* result,int num_bodies) {

	long int i;
	float 	F, r_x, r_y, r_z, r;
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
   	unsigned int index_y = blockIdx.y * blockDim.y + threadIdx.y;	
    	unsigned int index = index_y * num_bodies + index_x;

	for (int i = 0; i<num_bodies; i++){
        	for (int j = 0; j<num_bodies; j++){
            		r_x = (abs(bodies[j].xPosition - bodies[i].xPosition));
		        r_y = (abs(bodies[j].yPosition - bodies[i].yPosition));
            		r_z = (abs(bodies[j].zPosition - bodies[i].zPosition));
            		r = sqrt(r_x*r_x + r_y*r_y + r_z*r_z);
            		F = (G*bodies[i].mass*bodies[j].mass)/(r*r);
            		if(i != j){
                	printf ("Force between body %d and %d is:%fN\n",i,j,F);
            }
        }
	for (i=0;i<N;i++)
		sum += x[index_y*N+i]*y[index_x+i*N]; 
	
	result[index] = sum; 
}

