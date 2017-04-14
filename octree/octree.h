#ifndef OCTREE_H
#define OCTREE_H

typedef data_t   float;

typedef enum octype {ROOT, LVL_1, LVL_2} octant_type;  //  LVL_2 has no octant children, but uses body arrays

//  body struct for reading from file

typedef struct nbody{
    //char *category; 
	data_t	 mass;
	data_t	 pos_x;
	data_t	 pos_y;
	data_t	 pos_z;
	data_t	 vel_x;
	data_t	 vel_y;
	data_t	 vel_z;
} nbody;

void		nbody_enum(nbody **body_array, char* file);

//  octant struct and functions

typedef struct octant
{
	int 	octant_no;  // -1 if it is root node
	octype 	level;

	// center of mass for the octant
	data_t 	mass_center_x;
	data_t 	mass_center_y;
	data_t 	mass_center_z;
	data_t 	mass_total;

	// children
	octant 	*children[8];  //  NULL if level above planets

	// leaf arrays
	// malloc to create array
	// NULL if no leafs
	int 	leaf_count;  // index that will be written to, ie 2 means add 3rd body next or remove 2nd
	data_t 	*mass;

	data_t 	*pos_x;
	data_t 	*pos_y;
	data_t 	*pos_z;

	data_t 	*vel_x;
	data_t 	*vel_y;
	data_t 	*vel_z;

	data_t 	*acc_x;
	data_t 	*acc_y;
	data_t 	*acc_z;

} octant, *p_octant;  //  octant is a cube divided into 8 cubes inside it of equal size

p_octant 	octant_new(int oct_no, octype level);

void 		octant_center_of_mass(p_octant oct);
void		octant_add_child(p_octant oct, p_octant child);

int 		octant_add_body(p_octant oct, nbody* body);

int 		octant_move_leaf(p_octant src, p_octant dst, int offset);

#endif
