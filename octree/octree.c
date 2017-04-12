#include <stdio.h>
#include "octree.h"

// code for body reading and generation

#define LINE_LEN 	  256
#define MASS_INVALID  -1
#define AREA_CAPACITY 10000  // trying to overallocate
#define CHILD_COUNT   8

// following are int returns in fashion of t-f
// skip means the func was called wrongly, but
// that the evaluation was skipped and may not
// harm execution
#define KILL 		  0
#define PASS 		  1
#define SKIP 		  2

int nbody_enum(nbody *body_array[], char* file)  //  True - False response
{
	FILE *fp;
	if((fp = fopen(file, "r")) == NULL)
	{
		printf("\nFile could not be opened\n");
		return 0;
	}

	int i = 0;
	char *buf = malloc(LINE_LEN);
	while(fgets(buf, LINE_LEN - 1, fp) != NULL)
	{
		if ((strlen(buf)>0) && (buf[strlen (buf) - 1] == '\n'))
            buf[strlen (buf) - 1] = '\0';       
          
        // tmp = strtok(buf, ",");
        // nbodies[i].category = tmp;

        tmp = strtok(NULL, ",");
        nbodies[i].mass = atof(tmp);

        tmp = strtok(NULL, ",");
        nbodies[i].pos_x = atof(tmp);

        tmp = strtok(NULL, ",");
        nbodies[i].pos_y = atof(tmp);

        tmp = strtok(NULL, ",");
        nbodies[i].pos_z = atof(tmp);

        tmp = strtok(NULL, ",");
        nbodies[i].vel_x = atof(tmp);

        tmp = strtok(NULL, ",");
        nbodies[i].vel_y = atof(tmp);

        tmp = strtok(NULL, ",");
        nbodies[i].vel_z = atof(tmp);

        i++;
	}
	free(buf);
	fclose(fp);
	return 1;
}

//  code for octants

p_octant octant_new(int oct_no, octype level)
{
	p_octant oct = (p_octant) malloc(sizeof(octant));
	
	if(!oct)
	{
		printf("\nFailed to allocate octant\n");
		return NULL;
	}

	//generic for all octant types
	oct->octant_no  = oct_no;
	oct->leaf_count = 0;
	oct->level      = level;

	oct->mass_center_x = 0;
	oct->mass_center_y = 0;
	oct->mass_center_z = 0;
	oct->mass_total    = 0;

	if(level == LVL_2)
	{
		oct->mass  = (data_t*) malloc(AREA_CAPACITY * sizeof(data_t));

		oct->pos_x = (data_t*) malloc(AREA_CAPACITY * sizeof(data_t));
		oct->pos_y = (data_t*) malloc(AREA_CAPACITY * sizeof(data_t));
		oct->pos_z = (data_t*) malloc(AREA_CAPACITY * sizeof(data_t));

		oct->vel_x = (data_t*) malloc(AREA_CAPACITY * sizeof(data_t));
		oct->vel_y = (data_t*) malloc(AREA_CAPACITY * sizeof(data_t));
		oct->vel_z = (data_t*) malloc(AREA_CAPACITY * sizeof(data_t));

		oct->children  = NULL;  // has leaves not child octants
	}else  // ROOT or LVL_1
	{
		if(level == ROOT) oct->octant_no = -1; //redundant for safety

		oct->mass  = (data_t*) NULL;

		oct->pos_x = (data_t*) NULL;
		oct->pos_y = (data_t*) NULL;
		oct->pos_z = (data_t*) NULL;

		oct->vel_x = (data_t*) NULL;
		oct->vel_y = (data_t*) NULL;
		oct->vel_z = (data_t*) NULL;

		oct->children = (p_octant) malloc(CHILD_COUNT * sizeof(octant));
	}

	return oct;
}

void octant_center_of_mass(p_octant oct)  // maybe center of gravity
{
	if(oct->octant_no > -1)  // not the root
	{
		int i;
		data_t mass_total = 0;
		data_t x_acc = 0;
		data_t y_acc = 0;
		data_t z_acc = 0;

		if(oct->level == LVL_2)  //  for nodes with leafs
		{
			int leaf_count;
			leaf_count = oct->leaf_count;

			// calc total mass, weight positions
			for(i = 0; i < leaf_count; i++)
			{
				mass_total += oct->mass[i];
				x_acc      += (oct->pos_x[i] * oct->mass[i]);
				y_acc      += (oct->pos_y[i] * oct->mass[i]);
				z_acc      += (oct->pos_z[i] * oct->mass[i]);
			}
		}
		else  // calculate from children
		{
			for(i = 0; i < 8; i++)  //octants with suboctants
			{
				mass_total += oct->children[i].mass_total;
				x_acc      += (oct->childre[i].mass_center_x * oct->children[i].mass_total);
				y_acc      += (oct->childre[i].mass_center_y * oct->children[i].mass_total);
				z_acc      += (oct->childre[i].mass_center_z * oct->children[i].mass_total);
			}
		}
		
		oct->mass_total    = mass_total; 
		oct->mass_center_x = x_acc/mass_total;
		oct->mass_center_y = y_acc/mass_total;
		oct->mass_center_z = z_acc/mass_total;
	}
}

void octant_add_child(p_octant oct, p_octant child)
{
	oct->children[child->octant_no] = child;  // will need to create suboctants carefully
}

int octant_add_body(p_octant oct, nbody *body)  //returns true-false that may terminate execution
{
	if(oct->level == LVL_2)  // allowed to add bodies to this level
	{
		index = oct->leaf_count;
		if(index > AREA_CAPACITY)
		{
			printf("\n\tERROR: Suboctant body capacity exceeded in octant_add_body()!\n");
			printf("\tPLEASE rebuild with wider arrays in octant struct!\n");
			return KILL;
		}
		else
		{
			// add the body to the suboctant arrays
			oct->mass[index]  = body->mass;

			oct->pos_x[index] = body->pos_x;
			oct->pos_y[index] = body->pos_y;
			oct->pos_z[index] = body->pos_z;

			oct->vel_x[index] = body->vel_x;
			oct->vel_y[index] = body->vel_y;
			oct->vel_z[index] = body->vel_z;

			oct->leaf_count = index + 1;  // increment leaf count, don't care if at capacity for this iteration

			return PASS;
		}
	}
	else  //  improper call
	{
		return SKIP;
	}
}

int octant_move_leaf(p_octant src, p_octant dst, int src_index)
{
	int dst_index = dst->leaf_count;

	// check dst octant
	if(dst->level == LVL_2)
	{	
		if(dst_index == AREA_CAPACITY)  // boundary check
		{
			printf("\n\tERROR: Suboctant body capacity exceeded in octant_move_leaf()!\n");
			printf("\tPLEASE rebuild with wider arrays in octant struct!\n");
			return KILL;
		}
		else
		{

			//first -- copy leaf from source to destination
			dst->mass[dst_index]  = src->mass[src_index];

			dst->pos_x[dst_index] = src->pos_x[src_index];
			dst->pos_y[dst_index] = src->pos_y[src_index];
			dst->pos_z[dst_index] = src->pos_z[src_index];

			dst->vel_x[dst_index] = src->vel_x[src_index];
			dst->vel_y[dst_index] = src->vel_y[src_index];
			dst->vel_z[dst_index] = src->vel_z[src_index];

			dst->leaf_count = dst_index + 1;  // note body addition

			//remove leaf from source and compress arrays
			int src_end = src->leaf_count;

			src->mass[src_index]  = src->mass[src_end];

			src->pos_x[src_index] = src->pos_x[src_end];
			src->pos_y[src_index] = src->pos_y[src_end];
			src->pos_z[src_index] = src->pos_z[src_end];

			src->vel_x[src_index] = src->vel_x[src_end];
			src->vel_y[src_index] = src->vel_y[src_end];
			src->vel_z[src_index] = src->vel_z[src_end];

			src->leaf_count = src_end - 1;
			return PASS;	
		}
	}
	else  // improper octant type passed
	return SKIP;
}