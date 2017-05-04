#include "cpu_octree.h"

const	data_t GRAV_CONST = 6.674e-11;
#define LINE_LEN			512

int		body_count(char* filename)
{
	int count = 0;

	while(*filename) // still characters to process
	{
		if(isdigit(*filename))
		{
			count *= 10;
			count += strtol(filename, &filename, 10);
		}
		filename++;
	}

	return count;
}

int 	fileread_build_tree(char* filename, octant *root, int len)
{
	FILE *fp = fopen(filename, "r");

	if(!fp) return 0;

	//  file reading variables
	int i = 0;
	char *buf = (char*) malloc(LINE_LEN);
	int buf_len = 0;
	char* tmp;

	//  body variables to be placed into tree structure

	data_t 	mass;
	data_t 	pos_x, pos_y, pos_z;
	data_t 	vel_x, vel_y, vel_z;
	pair 	locus;
	int 	oct_major, oct_minor;

	//  read and assign loop

	while((i < len) && (fgets(buf, LINE_LEN - 1, fp) != NULL))
	{
		buf_len = strlen(buf);

		if((buf_len > 0) && (buf[buf_len - 1] == '\n'))
			buf[buf_len - 1] = '\0'; 

		// extract here
		tmp 		= strtok(buf, ",");
		mass	 	= STR_TO_DATA_T(tmp);

		tmp 		= strtok(NULL, ",");
		pos_x	 	= STR_TO_DATA_T(tmp);

		tmp 		= strtok(NULL, ",");
		pos_y	 	= STR_TO_DATA_T(tmp);

		tmp 		= strtok(NULL, ",");
		pos_z	 	= STR_TO_DATA_T(tmp);

		tmp 		= strtok(NULL, ",");
		vel_x	 	= STR_TO_DATA_T(tmp);

		tmp 		= strtok(NULL, ",");
		vel_y	 	= STR_TO_DATA_T(tmp);

		tmp 		= strtok(NULL, ",");
		vel_z	 	= STR_TO_DATA_T(tmp);

		//  placement code
		locus		= octant_locate(pos_x, pos_y, pos_z);
		oct_major	= locus.parent;
		oct_minor 	= locus.child;

		//printf("Add body to octant(%d, %d) at position(%lf, %lf, %lf)\n", oct_major, oct_minor, pos_x, pos_y, pos_z);

		if(!octant_add_body(root, oct_major, oct_minor, mass, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z))
		{
			printf("ERROR: Filled octant(%d, %d) beyond capacity!\n", oct_major, oct_minor);
			return 0;
		}

		//octant *local = root->children[oct_major]->children[oct_minor];
		//int    leaf = local->leaf_count - 1;
		//printf("Body %d in (%d, %d) with mass %.2lf kg was added at (%.2lf, %.2lf, %.2lf)\n", i, oct_major, oct_minor, local->mass[leaf], local->pos_x[leaf], local->pos_y[leaf], local->pos_z[leaf]);

		i++;
	}
	free(buf);
	fclose(fp);
	return 1;
}

void	force_zero(octant* root)
{
	int 		i, j, k, leaf_count;
	octant*		local;
	
	#ifdef THREAD_ACTIVE
		#pragma omp parallel for
	#endif

	for(i = 0; i < CHILD_COUNT; i++)
		for(j = 0; j < CHILD_COUNT; j++)
		{
			local 		= root->children[i]->children[j];
			leaf_count 	= local->leaf_count;
			for(k = 0; k < leaf_count; k++)
			{
				local->fma_x[k] = 0;
				local->fma_y[k] = 0;
				local->fma_z[k] = 0;
			}
		}
}

void 	body_body_force_accum(octant* oct, int focus, int comp)
{
	data_t r_x, r_y, r_z, r;

	r_x = oct->pos_x[focus] - oct->pos_x[comp];
	r_y = oct->pos_y[focus] - oct->pos_y[comp];
	r_z = oct->pos_z[focus] - oct->pos_z[comp];

	r 	= DISTANCE(r_x, r_y, r_z);

	data_t F_x, F_y, F_z, F_part;

	F_part 	= FORCE_PARTIAL(GRAV_CONST, oct->mass[focus], oct->mass[comp], r);

	// if(F_part > 0.0)
	// 	printf("F_part legit: %.15lf\n", F_part);
	// else
	// 	printf("F_part super small\n");

	F_x 	= F_part * r_x;
	F_y 	= F_part * r_y;
	F_z 	= F_part * r_z;

	oct->fma_x[focus] += F_x;
	oct->fma_y[focus] += F_y;
	oct->fma_z[focus] += F_z;

	oct->fma_x[comp] -= F_x;
	oct->fma_y[comp] -= F_y;
	oct->fma_z[comp] -= F_z;
}

void 	body_octant_force_accum(octant* local, int leaf, octant* distal)
{
	data_t r_x, r_y, r_z, r;

	r_x = local->pos_x[leaf] - distal->mass_center_x;
	r_y = local->pos_y[leaf] - distal->mass_center_y;
	r_z = local->pos_z[leaf] - distal->mass_center_z;

	if((r_x != r_x) || (r_y != r_y) || (r_z != r_z)) return;

	//printf("(%.4lf, %.4lf, %.4lf)\n", r_x, r_y, r_z);

	r 	= DISTANCE(r_x, r_y, r_z);

	data_t F_part 	= FORCE_PARTIAL(GRAV_CONST, local->mass[leaf], distal->mass_total, r);

	local->fma_x[leaf] += F_part * r_x;
	local->fma_y[leaf] += F_part * r_y;
	local->fma_z[leaf] += F_part * r_z;
}

#ifdef VECTOR_ACTIVE
void 	body_body_force_accum_vec(octant* oct, int focus, int comp)
{
	
	__m256d r_x, r_y, r_z, r, f_part;
	__m256d *comp_r_x = (__m256d*) &(oct->pos_x[comp]);
	__m256d *comp_r_y = (__m256d*) &(oct->pos_y[comp]);
	__m256d *comp_r_z = (__m256d*) &(oct->pos_z[comp]);

	//  broad cast for common subtraction
	__m256d focus_r_x = _mm256_set1_pd(oct->pos_x[focus]);
	__m256d focus_r_y = _mm256_set1_pd(oct->pos_y[focus]);
	__m256d focus_r_z = _mm256_set1_pd(oct->pos_z[focus]);

	//calculate the vector displacements
	r_x = _mm256_sub_pd(focus_r_x, *comp_r_x);
	r_y = _mm256_sub_pd(focus_r_y, *comp_r_y);
	r_z = _mm256_sub_pd(focus_r_z, *comp_r_z);

	// recycle focus for squaring
	focus_r_x = _mm256_mul_pd(r_x, r_x);
	focus_r_y = _mm256_mul_pd(r_y, r_y);
	focus_r_z = _mm256_mul_pd(r_z, r_z);

	// sum the squares into a recycled point
	r = _mm256_add_pd(focus_r_x, focus_r_y);
	r = _mm256_add_pd(r, focus_r_z);

	// squareroot of the sum of squares, focus_r_x contains four scalar distances
	r = _mm256_sqrt_pd(r);

	// collect masses for force calc
	__m256d mass1 = _mm256_set1_pd(oct->mass[focus]);
	__m256d *mass2 = (__m256d*) &(oct->mass[comp]);
	mass1 = _mm256_mul_pd(mass1, *mass2);

	focus_r_y = _mm256_mul_pd(r, r);  // first radius multiply

	__m256d g = _mm256_set1_pd(GRAV_CONST);

	__m256d force_top;
	force_top = _mm256_mul_pd(g, mass1);  //top half of divide
		
	r = _mm256_mul_pd(focus_r_y, r);      // r^3

	f_part = _mm256_div_pd(force_top,r);  // partial force calculated

	// calculation of vector forces
	__m256d f_part_x, f_part_y, f_part_z;

	f_part_x = _mm256_mul_pd(f_part, r_x);	
	f_part_y = _mm256_mul_pd(f_part, r_y);	
	f_part_z = _mm256_mul_pd(f_part, r_z);

	__m256d *fma_comp_x = (__m256d*) &(oct->fma_x[comp]);
	__m256d *fma_comp_y = (__m256d*) &(oct->fma_y[comp]);
	__m256d *fma_comp_z = (__m256d*) &(oct->fma_z[comp]);

	// apply newton'w third law to non-focus elements

	*fma_comp_x = _mm256_sub_pd(*fma_comp_x, focus_r_x);
	*fma_comp_y = _mm256_sub_pd(*fma_comp_y, focus_r_y);
	*fma_comp_z = _mm256_sub_pd(*fma_comp_z, focus_r_z);

	// half adds to sum the four calculations
	focus_r_x = _mm256_hadd_pd(focus_r_x, focus_r_x);	
	focus_r_y = _mm256_hadd_pd(focus_r_y, focus_r_y);	
	focus_r_z = _mm256_hadd_pd(focus_r_z, focus_r_z);

	focus_r_x = _mm256_hadd_pd(focus_r_x, focus_r_x);	
	focus_r_y = _mm256_hadd_pd(focus_r_y, focus_r_y);	
	focus_r_z = _mm256_hadd_pd(focus_r_z, focus_r_z);

	//  push lower sum into double for easy use
	data_t F_x, F_y, F_z;

	_mm_store_sd(&F_x, _mm256_castpd256_pd128(focus_r_x));
	_mm_store_sd(&F_y, _mm256_castpd256_pd128(focus_r_y));
	_mm_store_sd(&F_z, _mm256_castpd256_pd128(focus_r_z));


	oct->fma_x[focus] += F_x;
	oct->fma_y[focus] += F_y;
	oct->fma_z[focus] += F_z;

}
#endif
		// VECTOR_ACTIVE

void	force_accum(octant* root)
{
	int 	i, j, k, m, leaf_count;
	octant 	*local;

	#ifdef THREAD_ACTIVE
		#pragma omp parallel for
	#endif

	for(i = 0; i < CHILD_COUNT; i++)
		for(j = 0; j < CHILD_COUNT; j++)
		{
			local 		= root->children[i]->children[j];
			leaf_count  = local->leaf_count;
			//  force interactions in suboctant
			for(k = 0; k < leaf_count; k++)
			{
				#ifdef VECTOR_ACTIVE
					for(m = k + 1; m%4 != 0; m++)
						body_body_force_accum(local, k, m);

					for(;(m+3) < leaf_count;m += 4)
						body_body_force_accum_vec(local, k, m);

					for(;m < leaf_count; m++)
						body_body_force_accum(local, k, m);
				#else
					for(m = k + 1; m < leaf_count; m++)
						body_body_force_accum(local, k, m);
				#endif

				for(m = 0; m < CHILD_COUNT; m++)
					if(m != j)body_octant_force_accum(local, k, root->children[i]->children[m]);


				for(m = 0; m < CHILD_COUNT; m++)
					if(m != i)body_octant_force_accum(local, k, root->children[m]);

			}
		}
}

void	position_update(octant* root, int timestep)
{
	int 		i, j, k, leaf_count;
	octant* 	local;
	data_t 		mass;

	#ifdef THREAD_ACTIVE
		#pragma omp parallel for
	#endif

	for(i = 0; i < CHILD_COUNT; i++)
		for(j = 0; j < CHILD_COUNT; j++)
		{
			octant* local = root->children[i]->children[j];
			leaf_count    = local->leaf_count;

			for(k = 0; k < leaf_count; k++)
			{

				//printf("(%d, %d, %d) has %.5lf kg is at (%.3lf, %.3lf, %.3lf)\n", i, j, k, local->mass[k], local->fma_x[k], local->fma_y[k], local->fma_z[k]);

				mass 			  = local->mass[k];
				local->fma_x[k]  /= mass;
				local->fma_y[k]  /= mass;
				local->fma_z[k]  /= mass;

				//printf("(%d, %d, %d) has %.5lf kg is at (%.3lf, %.3lf, %.3lf)\n", i, j, k, local->mass[k], local->fma_x[k], local->fma_y[k], local->fma_z[k]);

				local->pos_x[k]	 += DISPLACE(local->vel_x[k], local->fma_x[k], timestep);
				local->pos_y[k]	 += DISPLACE(local->vel_y[k], local->fma_y[k], timestep);
				local->pos_z[k]	 += DISPLACE(local->vel_z[k], local->fma_z[k], timestep);

				//printf("(%d, %d, %d) is at (%.3lf, %.3lf, %.3lf)\n", i, j, k, local->pos_x[k], local->pos_y[k], local->pos_z[k]);
			}
		}
}

void	velocity_update(octant* root, int timestep)
{
	int 		i, j, k, leaf_count;
	octant* 	local;

	#ifdef THREAD_ACTIVE
		#pragma omp parallel for
	#endif
	
	for(i = 0; i < CHILD_COUNT; i++)
		for(j = 0; j < CHILD_COUNT; j++)
		{
			octant* local = root->children[i]->children[j];
			leaf_count    = local->leaf_count;

			for(k = 0; k < leaf_count; k++)
			{
				local->vel_x[k]	 += local->fma_x[k] * timestep;
				local->vel_y[k]	 += local->fma_y[k] * timestep;
				local->vel_z[k]	 += local->fma_z[k] * timestep;
			}
		}
}


#ifdef VECTOR_ACTIVE
void 	position_update_vec()
{

}

void 	velocity_update_vec()
{

}
#endif 
		// VECTOR_ACTIVE