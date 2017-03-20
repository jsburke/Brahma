/*
gcc force_calc.c
./a.out galaxy_267.csv
File: force_calc
To calculate force on ever heavenly body by every other body
Author: Sarthak Jagetia
Date created: 03/18/2017

FORMULAE:
F_x = ((G*m1*m2)/(r*r*r))*r_x
F_y = ((G*m1*m2)/(r*r*r))*r_y
F_z = ((G*m1*m2)/(r*r*r))*r_z

where r_x = x2-x1;
      r_y = y2-y1;
      r_z = z2-z1;

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
    double r_x;
    double r_y;
    double r_z;
};
struct nbody nbodies[MAX_BODIES];



int force_calc(FILE *s, int noofobjects){
    double F, r_x, r_y, r_z, r;
    double G = 6.67408e-11;
    char *buf = malloc(MAX_STR_LEN);
    char *tmp; 
    int limit = noofobjects;
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

        //printf("index i= %i, %f\n, %f\n, %f\n, %f\n, %f\n, %f\n, %f \n",i, nbodies[i].mass , nbodies[i].xPosition, nbodies[i].yPosition , nbodies[i].zPosition, nbodies[i].xVelocity, nbodies[i].yVelocity, nbodies[i].zVelocity);

        i++;
    }
    for (int i = 0; i<limit; i++){
        //printf("index i= %i, %f\n, %f\n, %f\n, %f\n, %f\n, %f\n, %f \n",i, nbodies[i].mass , nbodies[i].xPosition, nbodies[i].yPosition , nbodies[i].zPosition, nbodies[i].xVelocity, nbodies[i].yVelocity, nbodies[i].zVelocity);
        for (int j = 0; j<limit; j++){
            r_x = (abs(nbodies[j].xPosition - nbodies[i].xPosition));
            r_y = (abs(nbodies[j].yPosition - nbodies[i].yPosition));
            r_z = (abs(nbodies[j].zPosition - nbodies[i].zPosition));
            r = sqrt(r_x*r_x + r_y*r_y + r_z*r_z);
            F = (G*nbodies[i].mass*nbodies[j].mass)/(r*r);
            if(i != j){
                printf ("Force between body %d and %d is:%fN\n",i,j,F);
            }
        }



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
    force_calc(filename, j);
    fclose(filename);



    return 0;
}