#!/usr/bin/env python
#-------------------------------------------------------

#-------------------------------------------------------
#
#	-h --help		Display this message
#	-v --verbose		verbose output
#	-q --quiet		no command line noise (default)
#
#	The following must be followed by an integer value
#
#	--total			No. of bodies to create, ignored if others are greater
#
#	--bhole			No. of black holes
#	--star			No. of star like bodies (may expand to different types later)
#	--planet			No. of earth like bodies (rocky inner planets)
#	--giant			No. of gas giants (satrun, jupiter)
#	--moon			No. of moon like bodies
#	--minor			No. of asteroid, comet like things to make
# 
#	Output file is named galaxy_###.csv
#		where ### is the total number of bodies in the csv
#
#		format of each row is as follows:
#			category, mass, x position, y position, z position, x velocity, y velocity, z velocity,
# 
#	All masses produced are kg
#	All positions are km offset from origin
#	All velocities are km/s referenced to origin
#
# Author  : John S. Burke
# Date	  : 3 March 2017
# Rev.    : 1.0
#

import os, sys, argparse, random, math

#-------------------------------------------------------
# Generic space numbers
#-------------------------------------------------------

SOLAR_MASS			= 1.99e30 	# in kg
EARTH_MASS			= 5.9722e24
JUPITER_MASS		= 317.8 * EARTH_MASS
COMET_SMALL_MASS	= 1e13
CERES_MASS			= 8.958e20

EARTH_VELOCITY		= 30		# km/s
								# around sun

MAX_POS_DOUBLE		= 1.5e200
MAX_NEG_DOUBLE		= -MAX_POS_DOUBLE
LOW_TOL				= 0.15		# tolerances so we don't start at screen edges
HIGH_TOL			= 0.85

#--------------------------------------------------------
# Relative Body masses and velocities
#
# Included to make sure we don't have something too wacky
#--------------------------------------------------------

BHOLE_MASS_MIN	= 5 * SOLAR_MASS
BHOLE_MASS_MAX	= 200 * SOLAR_MASS

STAR_MASS_MIN	= 0.08 * SOLAR_MASS
STAR_MASS_MAX	= 12 * SOLAR_MASS

PLANET_MASS_MIN	= .025 * EARTH_MASS 	# little less than 1/2 Mercury
PLANET_MASS_MAX	= 17 * EARTH_MASS		# Kepler-10c

GIANT_MASS_MIN	= 0.22 * JUPITER_MASS	# Bit smaller than Saturn 
GIANT_MASS_MAX	= 12 * JUPITER_MASS		# just below low mass dwarf 13

MOON_MASS_MIN	= 0.008 * EARTH_MASS	# Europa
MOON_MASS_MAX	= 0.026 * EARTH_MASS  	# a midge bigger than Ganymede

MINOR_MASS_MIN	= COMET_SMALL_MASS
MINOR_MASS_MAX	= CERES_MASS



BHOLE_VELOCITY_MIN	= 0					# setting to initially fixed
BHOLE_VELOCITY_MAX	= 0

STAR_VELOCITY_MIN	= 0.00001 * EARTH_VELOCITY
STAR_VELOCITY_MAX	= 2.15 * EARTH_VELOCITY

PLANET_VELOCITY_MIN	= 0.15 * EARTH_VELOCITY
PLANET_VELOCITY_MAX = 5 * EARTH_VELOCITY

GIANT_VELOCITY_MIN	= 0.088 * EARTH_VELOCITY
GIANT_VELOCITY_MAX	= 0.5 * EARTH_VELOCITY

MOON_VELOCITY_MIN	= 0.1 * EARTH_VELOCITY
MOON_VELOCITY_MAX	= EARTH_VELOCITY

MINOR_VELOCITY_MIN	= 0.002 * EARTH_VELOCITY
MINOR_VELOCITY_MAX	= 21 * EARTH_VELOCITY

#--------------------------------------------------------
# Random body generation
#--------------------------------------------------------

def pos_rand():
	return random.uniform(LOW_TOL * MAX_POS_DOUBLE, HIGH_TOL * MAX_POS_DOUBLE) 

def vel_scale(v):
	return math.sqrt(v)

def body_create(category, mass_min, mass_max, vel_min, vel_max):
	mass  = str(random.uniform(mass_min, mass_max))
	x_pos = str(pos_rand())
	y_pos = str(pos_rand())
	z_pos = str(pos_rand())
	x_vel = str(random.uniform(vel_min, vel_max))
	y_vel = str(random.uniform(vel_min, vel_max))
	z_vel = str(random.uniform(vel_min, vel_max))
	return category + ", " + mass + ", " + x_pos + ", " + y_pos + ", " + z_pos + ", " + x_vel + ", " + y_vel + ", " + z_vel

def bhole_create(outfile):
	outfile.write(body_create("bhole", BHOLE_MASS_MIN, BHOLE_MASS_MAX, BHOLE_VELOCITY_MIN, BHOLE_VELOCITY_MAX) + "\n")

def star_create(outfile):
	outfile.write(body_create("star", STAR_MASS_MIN, STAR_MASS_MAX, STAR_VELOCITY_MIN, STAR_VELOCITY_MAX) + "\n")

def planet_create(outfile):
	outfile.write(body_create("planet", PLANET_MASS_MIN, PLANET_MASS_MAX, PLANET_VELOCITY_MIN, PLANET_VELOCITY_MAX) + "\n")

def giant_create(outfile):
	outfile.write(body_create("giant", GIANT_MASS_MIN, GIANT_MASS_MAX, GIANT_VELOCITY_MIN, GIANT_VELOCITY_MAX) + "\n")

def moon_create(outfile):
	outfile.write(body_create("moon", MOON_MASS_MIN, MOON_MASS_MAX, MOON_VELOCITY_MIN, MOON_VELOCITY_MAX) + "\n")

def minor_create(outfile):
	outfile.write(body_create("minor", MINOR_MASS_MIN, MINOR_MASS_MAX, MINOR_VELOCITY_MIN, MINOR_VELOCITY_MAX) + "\n")
#--------------------------------------------------------
# Command Line
#--------------------------------------------------------

class parse_defined_error(argparse.ArgumentParser):
	def error(self, msg = ""):
		if(msg): print("\nERROR: %s\n" % msg)
		file = open(sys.argv[0])
		for (lineNum, line) in enumerate(file):
			if(line[0] != "#"): sys.exit(msg != "")
			if((lineNum == 2) or (lineNum >= 4)): print(line[1:].rstrip("\n"))

def parse_cmd():
	parser = parse_defined_error(add_help = False)

	# Normal Args

	parser.add_argument("-h", "--help", action = "store_true")

	# Quiet <--> Verbose
	response = parser.add_mutually_exclusive_group()

	response.add_argument("-v", "--verbose", action = "store_true")
	response.add_argument("-q", "--quiet",   action = "store_true")

	# Additional command line args

	parser.add_argument("--total",  type = int)
	parser.add_argument("--bhole",  type = int)
	parser.add_argument("--star",   type = int)
	parser.add_argument("--planet", type = int)
	parser.add_argument("--giant",  type = int)
	parser.add_argument("--moon",   type = int)
	parser.add_argument("--minor",  type = int)

	options = parser.parse_args()
	if options.help:
		parser.error()
		sys.exit()
	if not (options.total or options.bhole or options.star or options.planet or options.giant or options.moon or options.minor):
		print("\nERROR: at least one of these options must be added:")
		print("\t--total")
		print("\t--bhole")
		print("\t--star")
		print("\t--planet")
		print("\t--giant")
		print("\t--moon")
		print("\t--minor")
		sys.exit()
	else:
		return options

#------------------------------------------------
# Main
#------------------------------------------------

def main():
	options = parse_cmd()

	# set up body parameters from command line

	total 		= 0;	# zero will be used to show we don't care
	remainder	= 0;	# total - all others
	bhole 	  	= 0;
	star 		= 0;
	planet 		= 0;
	giant 		= 0;
	moon 		= 0;
	minor 		= 0;

	verbose_on	= 0;	# assume user doesn't want verbose output
	if(options.verbose):
		print("Verbose output enabled\n")
		verbose_on = 1

	# get number of each type

	if (options.total):
		total = options.total
	if (options.bhole):
		bhole = options.bhole
	if (options.star):
		star = options.star
	if (options.planet):
		planet = options.planet
	if (options.giant):
		giant = options.giant
	if (options.moon):
		moon = options.moon
	if (options.minor):
		minor = options.minor

	# calculate total and unspecified remainder		

	bodySum = bhole + star + planet + giant + moon + minor
	if(verbose_on): print("Total number of bodies specified: %d" % bodySum)

	if(total < bodySum):
		total = bodySum
		print(" ")
	else: 
		remainder = total - bodySum
		if(verbose_on): print("Number of bodies unspecified: %d\n" % remainder)

	# produce ouput file

	filename = "galaxy_" + str(total) + ".csv"
	if(verbose_on): print("Output file: %r\n" % filename)

	outfile = open(filename, 'w')
	outfile.truncate()

	# generate the bodies and write to file

	for i in range(0, bhole):
		bhole_create(outfile)

	for i in range(0, star):
		star_create(outfile)		

	for i in range(0, planet):
		planet_create(outfile)

	for i in range(0, giant):
		giant_create(outfile)

	for i in range(0, moon):
		moon_create(outfile)

	for i in range(0, minor):
		minor_create(outfile)

	for i in range(0, remainder):
		sel = random.randint(0,5)

		if   sel == 0 : bhole_create(outfile)
		elif sel == 1 : star_create(outfile)
		elif sel == 2 : planet_create(outfile)
		elif sel == 3 : giant_create(outfile)
		elif sel == 4 : moon_create(outfile)
		elif sel == 5 : minor_create(outfile)

	outfile.close()
	if(verbose_on): print("Complete Success!  Shoot for the stars!")


main()	
