#include "octree.h"

octant*		octant_new(int lvl)
{
	octant* oct = (octant*) malloc(sizeof(octant));
	int i;

	if(!oct)
	{
		printf("ERROR: failed to malloc octant, level %d\n", lvl);
		return NULL;
	}

	// inits consistent across all levels
	oct->mass_center_x	= 0;
	oct->mass_center_y  = 0;
	oct->mass_center_z  = 0;
	oct->mass_total		= 0;

	oct->leaf_count 	= 0;

	// level dependent code

	if(lvl == LEVEL_2)  // most common case first
	{
		oct->mass = (data_t*) calloc(AREA_CAPACITY, sizeof(data_t));
		if(!(oct->mass))
		{
			printf("ERROR: failed to calloc masses, level %d\n", lvl);
			return NULL;
		}

		/////////////////////////////////////////////////////////////
		//
		// alloc positions
		//
		/////////////////////////////////////////////////////////////

		oct->pos_x = (data_t*) calloc(AREA_CAPACITY, sizeof(data_t));
		if(!(oct->mass))
		{
			printf("ERROR: failed to calloc x positions, level %d\n", lvl);
			return NULL;
		}

		oct->pos_y = (data_t*) calloc(AREA_CAPACITY, sizeof(data_t));
		if(!(oct->mass))
		{
			printf("ERROR: failed to calloc y positions, level %d\n", lvl);
			return NULL;
		}

		oct->pos_z = (data_t*) calloc(AREA_CAPACITY, sizeof(data_t));
		if(!(oct->mass))
		{
			printf("ERROR: failed to calloc z positions, level %d\n", lvl);
			return NULL;
		}

		/////////////////////////////////////////////////////////////
		//
		// alloc velocities
		//
		/////////////////////////////////////////////////////////////

		oct->vel_x = (data_t*) calloc(AREA_CAPACITY, sizeof(data_t));
		if(!(oct->mass))
		{
			printf("ERROR: failed to calloc x velocities, level %d\n", lvl);
			return NULL;
		}

		oct->vel_y = (data_t*) calloc(AREA_CAPACITY, sizeof(data_t));
		if(!(oct->mass))
		{
			printf("ERROR: failed to calloc y velocities, level %d\n", lvl);
			return NULL;
		}

		oct->vel_z = (data_t*) calloc(AREA_CAPACITY, sizeof(data_t));
		if(!(oct->mass))
		{
			printf("ERROR: failed to calloc z velocities, level %d\n", lvl);
			return NULL;
		}	

		/////////////////////////////////////////////////////////////
		//
		// alloc force -- acceleration vectors
		//
		/////////////////////////////////////////////////////////////

		oct->fma_x = (data_t*) calloc(AREA_CAPACITY, sizeof(data_t));
		if(!(oct->mass))
		{
			printf("ERROR: failed to calloc x force -- acceleration vectors, level %d\n", lvl);
			return NULL;
		}

		oct->fma_y = (data_t*) calloc(AREA_CAPACITY, sizeof(data_t));
		if(!(oct->mass))
		{
			printf("ERROR: failed to calloc y force -- acceleration vectors, level %d\n", lvl);
			return NULL;
		}

		oct->fma_z = (data_t*) calloc(AREA_CAPACITY, sizeof(data_t));
		if(!(oct->mass))
		{
			printf("ERROR: failed to calloc z force -- acceleration vectors, level %d\n", lvl);
			return NULL;
		}	

		for(i = 0; i < CHILD_COUNT; i++)
			oct->children[i] = (octant*) NULL;
	}
	else  // ROOT or LEVEL 1
	{
		oct->mass  = (data_t*) NULL;

		oct->pos_x = (data_t*) NULL;
		oct->pos_y = (data_t*) NULL;
		oct->pos_z = (data_t*) NULL;

		oct->vel_x = (data_t*) NULL;
		oct->vel_y = (data_t*) NULL;
		oct->vel_z = (data_t*) NULL;

		oct->fma_x = (data_t*) NULL;
		oct->fma_y = (data_t*) NULL;
		oct->fma_z = (data_t*) NULL;

		for(i = 0; i < CHILD_COUNT; i++)
			oct->children[i] = (octant*) NULL;
	}

	return oct;
}

int 		body_move(octant* src, octant* dst, int src_index)
{
	int dst_index = dst->leaf_count;

	if(dst_index == AREA_CAPACITY)
	{
		printf("\n\tERROR: Suboctant body capacity exceeded in octant_move_leaf()!\n");
		printf("\tPLEASE rebuild with wider arrays in octant struct!\n");
		return 0;
	}
	else
	{

		// first copy from source to dst
		dst->mass[dst_index]  = src->mass[src_index];

		dst->pos_x[dst_index] = src->pos_x[src_index];
		dst->pos_y[dst_index] = src->pos_y[src_index];
		dst->pos_z[dst_index] = src->pos_z[src_index];

		dst->vel_x[dst_index] = src->vel_x[src_index];
		dst->vel_y[dst_index] = src->vel_y[src_index];
		dst->vel_z[dst_index] = src->vel_z[src_index];

		// don't copy acceleration vectors, will zero out later

		dst->leaf_count = dst_index + 1;  // note body addition

		//move last leaf to index of moved leaf
		//truncate array by decrementing leaf_count
		int src_end = src->leaf_count;

		src->mass[src_index]  = src->mass[src_end];

		src->pos_x[src_index] = src->pos_x[src_end];
		src->pos_y[src_index] = src->pos_y[src_end];
		src->pos_z[src_index] = src->pos_z[src_end];

		src->vel_x[src_index] = src->vel_x[src_end];
		src->vel_y[src_index] = src->vel_y[src_end];
		src->vel_z[src_index] = src->vel_z[src_end];
		// not going to zero out src_end, just consider as garbage
		// decremented leaf_count will handle it's access

		// again, not copying acceleration vectors

		src->leaf_count = src_end - 1;

		return 1;
	}
}

int			octree_rebuild(octant* root)
{
		int 	oct_major, oct_minor, leaf, leaf_count, safe;
		octant *local, *distal;
		pair 	check;

		for(oct_major = 0; oct_major < CHILD_COUNT; oct_major++)
			for(oct_minor = 0; oct_minor < CHILD_COUNT; oct_minor++)
			{
				local 		= root->children[oct_major]->children[oct_minor];
				leaf_count  = local->leaf_count;

				for(leaf = 0; leaf < leaf_count; leaf++)
				{
					check = octant_locate(local->pos_x[leaf], local->pos_y[leaf], local->pos_z[leaf]);
					if((check.parent != oct_major) || (check.child != oct_minor))
					{
						distal = root->children[check.parent]->children[check.child];
						safe = body_move(local, distal, leaf);

						if(!safe) return 0;
					}
				}
			}

		return 1;
}

void 		center_of_mass_update(octant* root)
{
	int 	i, j, k, leaf_count;
	octant* local;

	data_t mass_accum, x_accum, y_accum, z_accum;
	data_t mass_accum_L1, x_accum_L1, y_accum_L1, z_accum_L1;

	for(i = 0; i < CHILD_COUNT; i++)
	{
		mass_accum_L1   = 0;
		x_accum_L1		= 0;
		y_accum_L1		= 0;
		z_accum_L1		= 0;

		for(j = 0; j < CHILD_COUNT; j++)
		{
			local 		= root->children[i]->children[j];
			leaf_count 	= local->leaf_count;

			mass_accum  = 0;
			x_accum		= 0;
			y_accum		= 0;
			z_accum		= 0;

			for(k = 0; k < leaf_count; k++)
			{
				mass_accum	 += local->mass[k];
				x_accum      += (local->pos_x[i] * local->mass[i]);
				y_accum      += (local->pos_y[i] * local->mass[i]);
				z_accum      += (local->pos_z[i] * local->mass[i]);
			}

			//printf("Center of mass (%d, %d) -- %.3lf kg (%.4lf, %.4lf, %.4lf)\n", i, j, mass_accum, x_accum, y_accum, z_accum);

			local->mass_total 	 = mass_accum;
			local->mass_center_x = x_accum/mass_accum;
			local->mass_center_y = y_accum/mass_accum;
			local->mass_center_z = z_accum/mass_accum;

			// do stuff for higher level while here
			mass_accum_L1   += mass_accum;
			x_accum_L1		+= local->mass_center_x;
			y_accum_L1		+= local->mass_center_y;
			z_accum_L1		+= local->mass_center_z;
				
		}

		//Level 1 calculations

		//printf("Center of mass (%d) -- %.3lf kg (%.4lf, %.4lf, %.4lf)\n", i, mass_accum_L1, x_accum_L1, y_accum_L1, z_accum_L1);

		local = root->children[i];

		local->mass_total 	 = mass_accum_L1;
		local->mass_center_x = x_accum_L1/mass_accum_L1;
		local->mass_center_y = y_accum_L1/mass_accum_L1;
		local->mass_center_z = z_accum_L1/mass_accum_L1;
	}
}

pair 		octant_locate(data_t x, data_t y, data_t z)
{
	data_t half_x, half_y, half_z;
	pair locus;

	// inits so that this works properly
	locus.parent = 0;
	locus.child  = 0;

	if(x >= 0) 
	{
		locus.parent 	+= 1;
		half_x  		= POS_QUARTER_MARK;
	}
	else
	{
		half_x  		= NEG_QUARTER_MARK;
	}
	locus.child 		+= (x >= half_x) ? 1 : 0;

	if(y >= 0)
	{
		locus.parent 	+= 2;
		half_y  		= POS_QUARTER_MARK;
	}
	else
	{
		half_y  		= NEG_QUARTER_MARK;
	}
	locus.child 		+= (y >= half_y) ? 2 : 0;

	if(z >= 0)
	{
		locus.parent 	+= 4;
		half_z  		= POS_QUARTER_MARK;
	}
	else
	{
		half_z  		= NEG_QUARTER_MARK;
	}
	locus.child 		+= (z >= half_z) ? 4 : 0;

	return locus;
}

int 		octant_add_body(octant* root, int major, int minor, data_t mass, data_t pos_x, data_t pos_y, data_t pos_z, data_t vel_x, data_t vel_y, data_t vel_z)
{
	octant* local 	= root->children[major]->children[minor];
	int leaf_count 	= local->leaf_count;

	if(leaf_count < AREA_CAPACITY)  // still safe
	{
		local->mass[leaf_count]		= mass;

		local->pos_x[leaf_count] 	= pos_x;
		local->pos_y[leaf_count] 	= pos_y;
		local->pos_z[leaf_count] 	= pos_z;

		local->vel_x[leaf_count] 	= vel_x;
		local->vel_y[leaf_count] 	= vel_y;
		local->vel_z[leaf_count] 	= vel_z;

		local->leaf_count 			= leaf_count + 1;

		return 1;

		//  don't care about fma arrays since we zero them out at the start of every loop anyhow
	}
	else return 0;
}