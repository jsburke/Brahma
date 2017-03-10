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

import os, sys, subprocess, argparse, random

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

	print("\nhello\n")

main()	
