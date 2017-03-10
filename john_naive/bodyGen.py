#!/usr/bin/env python
#-------------------------------------------------------
# bodyGen [-h] [-v|-q] [options]
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
#
# Author  : John S. Burke
# Date	  : 3 March 2017
# Rev.    : 0.1
#

import os, sys, argparse, random

#-------------------------------------------------------
# Generic space numbers
#-------------------------------------------------------

SOLAR_MASS			= 1.99e30 # in kg
EARTH_MASS			= 5.9722e24
JUPITER_MASS		= 317.8 * EARTH_MASS
COMET_SMALL_MASS	= 1e13
CERES_MASS			= 8.958e20

#--------------------------------------------------------
# Relative Body masses and velocities
#
# Included to make sure we don't have something too wacky
#--------------------------------------------------------

bhole_mass_min	= 5 * SOLAR_MASS
bhole_mass_max	= 200 * SOLAR_MASS

star_mass_min	= 0.08 * SOLAR_MASS
star_mass_max	= 12 * SOLAR_MASS

planet_mass_min	= .025 * EARTH_MASS 	# little less than 1/2 Mercury
planet_mass_max	= 17 * EARTH_MASS		# Kepler-10c

giant_mass_min	= 0.22 * JUPITER_MASS	# Bit smaller than Saturn 
giant_mass_max	= 12 * JUPITER_MASS		# just below low mass dwarf 13

moon_mass_min	= 0.008 * EARTH_MASS	# Europa
moon_mass_max	= 0.026 * EARTH_MASS  	# a midge bigger than Ganymede

minor_mass_min	= COMET_SMALL_MASS
minor_mass_max	= CERES_MASS



bhole_velocity_min = 0					# setting to initially fixed
bhole_velocity_max = 0
#--------------------------------------------------------
# Command Line
#--------------------------------------------------------

class ParseWithError(argparse.ArgumentParser):
	def error(self, msg = ""):
		if(msg): print("\nERROR: %s\n" % msg)
		file = open(sys.argv[0])
		for (lineNum, line) in enumerate(file):
			if(line[0] != "#"): sys.exit(msg != "")
			if((lineNum == 2) or (lineNum >= 4)): print(line[1:].rstrip("\n"))

def parse_cmd():
	parser = ParseWithError(add_help = False)

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

	for i in xrange(0, bhole):
		print(i)

	outfile.close()
	if(verbose_on): print("Complete Success!  Shoot for the stars!")


main()	
