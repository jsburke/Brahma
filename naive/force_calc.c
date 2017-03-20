/*
File: force_calc
To calculate force on ever heavenly body by every other body
Author: Sarthak Jagetia
Date created: 03/18/2017
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STR_LEN 256
#define MAX_BODIES 256

struct nbody{
        char *category; 
	double mass;
	double xPosition;
	double yPosition;
	double zPosition;
	double xVelocity;
	double yVelocity;
	double zVelocity;
};
struct nbody nbodies[MAX_BODIES];

int force_calc(&filename){
	csvfile = *filename;
	FILE *bodyGenFile;

    /* allocation of the buffer for every line in the File */
    char *buf = malloc(MAX_STR_LEN);
    char *tmp; 

    /* if the space could not be allocated, return an error */
    if (buf == NULL) {
        printf ("Memory Allocation Error\n");
        return 1;
    }
   
       
    if ( ( bodyGenFile = fopen( csvfile, "r" ) ) == NULL ) //Reading a file
    {
        printf( "File could not be opened.\n" );
    }


    int i = 0;
    while (fgets(buf, 255, bodyGenFile) != NULL)
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

        printf("index i= %i, %f\n, %f\n, %f\n, %f\n, %f\n, %f\n, %f \n",i, nbodies[i].mass , nbodies[i].xPosition, nbodies[i].yPosition , nbodies[i].zPosition, nbodies[i].xVelocity, nbodies[i].yVelocity, nbodies[i].zVelocity);

        i++;
    }
    free(buf);
    fclose(bodyGenFile);

}


int main(int argc, char **argv)
{   
    /* To Extract Number of Bodies from file name */
    int total_n = 0;
    int n;
    int j;
    while (1 == sscanf(argv[1] + total_n, "%*[^0123456789]%d%n", &j, &n))
    {
    	total_n += n;
        //printf("%d\n", total_n);
    	printf("%d\n", j);
    }
    int *filename;
    *filename = argv[1]; //storing the filename 
    force_calc(filename);


    return 0;
}