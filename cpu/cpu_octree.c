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



		i++;
	}
	free(buf);
	fclose(fp);
	return 1;

	return 1;
}
