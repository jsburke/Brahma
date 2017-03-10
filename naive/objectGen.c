////////////////////////////////////////////////////////////////////////////////////////////////
//
// Program used to produce csv full of various celestial bodies to be simulated
//
// Command line interface:
//		objectGen [option] <number>
//
//			options:
//				
//				-t   ---	total      : number that follows is number of bodies to make
//				-s   ---	stars      : number of stars that should be included
//				-b   ---	blackholes : number of blackholes
//				-p   ---    planets    : number of earth sized planets
//				-g   ---    giants     : number of gas giants to be made
//				-m   ---	moons      : number of moon sized objects
//				-a   ---	asteroids  : nuber of asteroid, comet objects to include
//
//			each of these will have a general mass range and velocity ranges
//			if -t is specified in alone, then the output will be completely random
//			if various options are declared without -t, the total will be reflective of the sum
//
/////////////////////////////////////////////////////////////////////////////////////////////////