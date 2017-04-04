#include <stdio.h>
#include "octree.h"

int nbody_enum(nbody **body_array, char* file)  //  True - False response
{
	FILE *fp;
	if((fp = fopen(file, "r")) == NULL)
	{
		printf("\nFile could not be opened\n");
		return 0;
	}

	int i = 0;
	char *buf = malloc(256);
	while(fgets(buf, 255, fp) != NULL)
	{
		if ((strlen(buf)>0) && (buf[strlen (buf) - 1] == '\n'))
            buf[strlen (buf) - 1] = '\0';       
          
        tmp = strtok(buf, ",");
        nbodies[i].category = tmp;

        tmp = strtok(NULL, ",");
        nbodies[i].mass = atof(tmp);

        tmp = strtok(NULL, ",");
        nbodies[i].xPosition = atof(tmp);

        tmp = strtok(NULL, ",");
        nbodies[i].yPosition = atof(tmp);

        tmp = strtok(NULL, ",");
        nbodies[i].zPosition = atof(tmp);

        tmp = strtok(NULL, ",");
        nbodies[i].xVelocity = atof(tmp);

        tmp = strtok(NULL, ",");
        nbodies[i].yVelocity = atof(tmp);

        tmp = strtok(NULL, ",");
        nbodies[i].zVelocity = atof(tmp);

        i++;
	}
	free(buf);
	fclose(fp);
	return 1;
}