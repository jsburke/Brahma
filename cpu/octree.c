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

void 		center_of_mass_update(octant* root)
{
	
}

pair 		octant_locate(data_t x, data_t y, data_t z)
{
	pair locus;

	return locus;
}