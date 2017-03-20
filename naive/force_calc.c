/*
gcc force_calc.c
./a.out galaxy_267.csv
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



int force_calc(FILE *s){
    char *buf = malloc(MAX_STR_LEN);
    char *tmp; 

    // if the space could not be allocated, return an error 
    if (buf == NULL) {
        printf ("Memory Allocation Error\n");
        return 1;
    }
   
       
    if ( s == NULL ) //Reading a file
    {
        printf( "File could not be opened.\n" );
    }


    int i = 0;
    while (fgets(buf, 255, s) != NULL)
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
    fclose(s);
    return 1;

}//*/

//////////////////////////////////////////////////////////////main/////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){   
    /* To Extract Number of Bodies from file name */
    int total_n = 0;
    int n;
    int j;
    while (1 == sscanf(argv[1] + total_n, "%*[^0123456789]%d%n", &j, &n))
    {
    	total_n += n;
        //printf("%d\n", total_n);
    	//printf("%d\n", j); //prints the number in the file name eg: prints 10 of galaxy_10.csv
    }
    FILE *filename;
    filename = fopen(argv[1],"r");

    //char a = argv[1];
    //filename = &a;
    force_calc(filename);
    //FILE a = argv[1]; //storing the filename 
    //filename = &a;
    //printf("%s",a);
    //printf("\n");
   // force_calc(filename);

    fclose(filename);
    return 0;
}