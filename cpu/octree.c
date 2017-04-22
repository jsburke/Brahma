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

int			octree_rebuild(octant* root)
{
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
				x_accum      += (oct->pos_x[i] * oct->mass[i]);
				y_accum      += (oct->pos_y[i] * oct->mass[i]);
				z_accum      += (oct->pos_z[i] * oct->mass[i]);
			}

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