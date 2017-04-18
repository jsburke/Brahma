#ifndef OCTREE_H
#define OCTREE_H

typedef data_t   float; // change me below on type change!! SUPER IMPORTANT
#define DATA_T_FLOAT			1
#define DATA_T_ERR 				-1

//  below sets the range that he objects can exist in
//  restricted based on what data_t 
#ifdef  DATA_T_FLOAT
	#define MAX_POS_POSITION 	3.4e38
#elif 	DATA_T_DOUBLE
	#define MAX_POS_POSITION	1.7e308
#else
	#define MAX_POS_POSITION	DATA_T_ERR  // we really screwed up hard
#endif

#define MAX_NEG_POSITION 		-MAX_POS_POSITION
#define POS_QUARTER_MARK		MAX_POS_POSITION/2.0 // use 2.0 for float and double data_t
#define NEG_QUARTER_MARK 		-POS_QUARTER_MARK

typedef enum octype {ROOT, LVL_1, LVL_2} octant_type;  //  LVL_2 has no octant children, but uses body arrays
#define CHILD_COUNT			 	8

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
	octant 	*children[CHILD_COUNT];  //  NULL if level above planets

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

typedef struct octant_pair  // for parent - child relations
{
	int 	parent;
	int 	child;
} octant_pair;

p_octant 	octant_new(int oct_no, octype level);

void 		octant_center_of_mass(p_octant oct);
void		octant_add_child(p_octant oct, p_octant child);

int 		octant_add_body(p_octant oct, nbody* body);

int 		octant_move_leaf(p_octant src, p_octant dst, int offset);

void 		octant_acceleration_zero(p_octant oct);
int 		octree_rebuild(p_octant root);
octant_pair octant_locate(data_t pos_x, data_t pos_y, data_t pos_z);

#endif
