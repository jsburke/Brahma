#ifndef OCTREE_H
#define OCTREE_H

typedef data_t   float;

typedef struct octant
{
	int 	octant_no;

	// center of mass for the octant
	data_t 	mass_center_x;
	data_t 	mass_center_y;
	data_t 	mass_center_z;
	data_t 	mass_avg;

	// children
	octant 	children[8];  //  NULL if level above planets

	// leaf arrays
	// malloc to create array
	data_t 	*mass;

	data_t 	*pos_x;
	data_t 	*pos_y;
	data_t 	*pos_z;

	data_t 	*vel_x;
	data_t 	*vel_y;
	data_t 	*vel_z;

} octant, *p_octant;  //  octant is a cube divided into 8 cubes inside it of equal size

#endif