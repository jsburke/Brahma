#include "grav_gpu.hu"

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
