#include <stdio.h>
#include "octree.h"

// code for body reading and generation

#define LINE_LEN 256

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

p_octant octant_new(octant_type type, int oct_no)
{
	p_octant oct = (p_octant) malloc(sizeof(octant));
	
	if(!oct)
	{
		printf("\nFailed to allocate octant\n");
		return NULL;
	}

	if(type == ROOT)
	{
		oct->octant_no = -1;
	}
	else oct->octant_no = octant_no;

	return oct;
}

void octant_add_body(p_octant oct, nbody *body)
{
	
}

void octant_center_of_mass(p_octant oct)
{
	if(oct->octant_no > -1)  // am I not the root
	{
		if(oct->mass != NULL)  // do I have child octants or body arrays
		{
			// calculate center of mass from body arrays
			int i;

			for(i = 0; oct->leaf_count; i++)
			{
				oct->mass[i]
			}
		}
	}
}