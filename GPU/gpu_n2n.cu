#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>

#define REBUILD_FREQ		5			// Rebuild after every X iterations
#define TIME_STEP 			30  			// in simulation time, in minutes
#define EXIT_COUNT			200 			// Number of iterations to do before exiting, -1 for infinite
#define FILENAME_LEN	 	256
#define ERROR 				-1 			// Generic Error val for readability

#define SECS				TIME_STEP * 60		// seconds per time step
#define GRAV_CONST	  	  	6.67408e-11;
#define NUM_THREADS			16
#define NUM_BLOCKS			1
#define LINE_LEN			512

typedef float data_t;

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) force_zero(fma_x, fma_y, fma_z, num_bodies);
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

int fileread_build_arrays(char* filename, data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* vel_x, data_t* vel_y, data_t* vel_z, int len);

__global__ void kernel_force_accum(data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int num_bodies, int time)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	data_t* r_x, r_y, r_z, r;
	data_t* F_part;
	
	for(i = 0; i < num_bodies; i++){
		//int idx = index * num_bodies + i;
		if(i != idx){
			r_x[idx] = pos_x[idx] - pos_x[i];
			r_y[idx] = pos_y[idx] - pos_y[i];
			r_z[idx] = pos_z[idx] - pos_z[i];
	
			r[idx] = sqrt((r_x[idx] * r_x[idx]) + (r_y[idx] * r_y[idx]) + (r_z * r_z[idx]));

			// force 
			F_part[idx] = GRAV_CONST * (mass[idx] * mass[i])/(r[idx] * r[idx] *r[idx]);

			fma_x[idx]  += F_part[idx] * r_x[idx];
			fma_y[idx]  += F_part[idx] * r_y[idx];
			fma_z[idx]  += F_part[idx] * r_z[idx];

			fma_x[i]   += -fma_x[idx];
			fma_y[i]   += -fma_y[idx];
			fma_z[i]   += -fma_z[idx];
			}
		__syncthreads();
	}
	position_update(mass[idx], pos_x[idx], pos_y[idx], pos_z[idx], vel_x[idx], vel_y[idx], vel_z[idx], fma_x[idx], fma_y[idx],fma_z[idx], num_bodies, time);
	velocity_update(mass[idx], vel_x[idx], vel_y[idx], vel_z[idx], fma_x[idx], fma_y[idx], fma_z[idx], num_bodies, TIME_STEP);
	
}

__device__ void position_update(data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* vel_x, data_t* vel_y, data_t* vel_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int num_bodies, int time)
{
	//  NB, when this is invoked, fma arrays will have forces built up in force_accum()
	int i;
	//int index = blockIdx.x * blockDim.x + threadIdx.x;

	for(i = 0; i < len; i++)
	{
		//int idx = index * num_bodies + i;

		// convert forces to acceleration, saves a multiply later
		fma_x[idx] /= mass[idx];
		fma_y[idx] /= mass[idx];
		fma_z[idx] /= mass[idx];

		pos_x[idx] += time * (vel_x[idx] + (0.5 * fma_x[idx] * time)); 
		pos_y[idx] += time * (vel_y[idx] + (0.5 * fma_y[idx] * time)); 
		pos_z[idx] += time * (vel_z[idx] + (0.5 * fma_z[idx] * time)); 
		__syncthreads();
	}
}

__device__ void velocity_update(data_t* mass, data_t* vel_x, data_t* vel_y, data_t* vel_z, data_t* fma_x, data_t* fma_y, data_t* fma_z, int num_bodies, int time)
{
	// NB, when this is invoked, fma arrays should be accelerations set in position_update()
	int i;
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	for(i = 0; i < num_bodies; i++)
	{
		int idx = index * num_bodies + i;
		vel_x[idx] += fma_x[idx] * time;
		vel_y[idx] += fma_y[idx] * time;
		vel_z[idx] += fma_z[idx] * time;
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
	int 		i, j, k;
	int 		num_bodies = 0;
	int 		allocSize = 0;

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
	
	// total size required for allocation
	allocSize = sizeof(data_t) * num_bodies;
	
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

	if(!h_mass || !h_pos_x || !h_pos_y || !h_pos_z || !h_vel_x || !h_vel_y || !h_vel_z || !h_fma_x || !h_fma_y || !h_fma_z)
	{
		printf("ERROR: Array malloc issue!\n");
		return 0;
	}
	
	// read the file
	fileread_build_arrays(filename, h_mass, h_pos_x, h_pos_y, h_pos_z, h_vel_x, h_vel_y, h_vel_z, num_bodies);

	//accelerations
	force_zero(fma_x, fma_y, fma_z, num_bodies); /////////////////// need to fix this ////////////////////////////

	// Select GPU
        CUDA_SAFE_CALL(cudaSetDevice(0));
	
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
	
	free(filename);  //  file will no longer be accessed

	// Launch the kernel
    	dim3 dimBlock(NUM_THREADS, NUM_THREADS, 1);
	dim3 dimGrid(NUM_BLOCKS, NUM_BLOCKS, 1);
	
	for(i = 0; i < EXIT_COUNT; i++)
	{
		kernel_force_accum<<<dimGrid, dimBlock>>>(mass, pos_x, pos_y, pos_z, fma_x, fma_y, fma_z,num_bodies,TIME_STEP);
	
	}

	// Check for errors during launch
	CUDA_SAFE_CALL(cudaPeekAtLastError());
	
	// Transfer the results back to the host
	CUDA_SAFE_CALL(cudaMemcpy(h_result, d_result, allocSize, cudaMemcpyDeviceToHost));

	// Free-up device and host memory
	CUDA_SAFE_CALL(cudaFree(d_mass));
	CUDA_SAFE_CALL(cudaFree(d_pos_x));
	CUDA_SAFE_CALL(cudaFree(d_pos_y));
	CUDA_SAFE_CALL(cudaFree(d_pos_z));
	CUDA_SAFE_CALL(cudaFree(d_vel_x));
	CUDA_SAFE_CALL(cudaFree(d_vel_y));
	CUDA_SAFE_CALL(cudaFree(d_vel_z));
	CUDA_SAFE_CALL(cudaFree(d_fma_x));
	CUDA_SAFE_CALL(cudaFree(d_fma_y));
	CUDA_SAFE_CALL(cudaFree(d_fma_z));

	free(h_mass);
	free(h_pos_x);
	free(h_pos_y);
	free(h_pos_z);
	free(h_vel_x);
	free(h_vel_y);
	free(h_vel_z);
	free(h_fma_x);
	free(h_fma_y);
	free(h_fma_z);

	return 0;
}

void	force_zero(data_t* x, data_t* y, data_t* z, int len)
{
	int i;

	for(i = 0; i < len; i++)
	{
		x[i] = 0;
		y[i] = 0;
		z[i] = 0;
	}
}

int fileread_build_arrays(char* filename, data_t* mass, data_t* pos_x, data_t* pos_y, data_t* pos_z, data_t* vel_x, data_t* vel_y, data_t* vel_z, int len)
{
	// returns true -- false
	FILE *fp = fopen(filename, "r");

	if(fp == NULL) return 0;

	int i = 0;
	char *buf = (char*) malloc(LINE_LEN);
	int buf_len = 0;

	while((i < len) && (fgets(buf, LINE_LEN - 1, fp) != NULL))
	{
		buf_len = strlen(buf);

		if((buf_len > 0) && (buf[buf_len - 1] == '\n'))
			buf[buf_len - 1] = '\0'; 

	tmp = strtok(buf, ",");
        tmp = strtok(NULL, ",");
        h_mass[i] = atof(tmp);

        tmp = strtok(NULL, ",");
        h_pos_x[i] = atof(tmp);

        tmp = strtok(NULL, ",");
        h_pos_y[i] = atof(tmp);

        tmp = strtok(NULL, ",");
        h_pos_z[i] = atof(tmp);

        tmp = strtok(NULL, ",");
        h_vel_x[i] = atof(tmp);

        tmp = strtok(NULL, ",");
        h_vel_y[i] = atof(tmp);

        tmp = strtok(NULL, ",");
        h_vel_z[i] = atof(tmp);

	i++;
	}
	free(buf);
	fclose(fp);
	return 1;
}
