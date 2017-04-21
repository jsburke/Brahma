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
	return 0;
}
