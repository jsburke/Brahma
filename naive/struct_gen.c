//This C program takes the csv file, generated by the python script "body-gen.py", from the command line and generates a struct array
//Version - 1.0 

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


int main(int argc, char **argv)
{   
    /* To Extract Number of Bodies from file name */
    int total_n = 0;
    int n;
    int j;
    while (1 == sscanf(argv[1] + total_n, "%*[^0123456789]%d%n", &j, &n))
    {
    	total_n += n;
    	//printf("%d\n", j);
    }
    
    
    FILE *bodyGenFile;

    /* allocation of the buffer for every line in the File */
    char *buf = malloc(MAX_STR_LEN);
    char *tmp; 

    /* if the space could not be allocated, return an error */
    if (buf == NULL) {
        printf ("Memory Allocation Error\n");
        return 1;
    }
   
       
    if ( ( bodyGenFile = fopen( argv[1], "r" ) ) == NULL ) //Reading a file
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

      //   printf("index i= %i, %f\n, %f\n, %f\n, %f\n, %f\n, %f\n, %f \n",i, nbodies[i].mass , nbodies[i].xPosition, nbodies[i].yPosition , nbodies[i].zPosition, nbodies[i].xVelocity, nbodies[i].yVelocity, nbodies[i].zVelocity);

        i++;
    }
    //free(buf);
    fclose(bodyGenFile);
    return 0;

}

