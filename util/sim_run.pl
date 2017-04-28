: # -*- perl -*-
  eval 'exec perl -S  $0 ${1+"$@"}' 
    if 0;  # if running under some shell
#
# Copied from /Lab3/scripts/combineBuild and modified

system("clear");

$myname = "sim_run";

$cwd = `pwd`; chomp $cwd;
if (!($cwd =~ m|/cpu$|)) {
  die "$myname: Curdir should be the 'cpu' directory.\n";
}

$n2n = "./N2N_CPU";
if (-e $n2n) {
  unlink($n2n);
}

$oct = "./OCTREE_CPU";
if (-e $oct) {
  unlink($oct);
}

$octomp = "./OCTREE_OMP";
if (-e $octomp) {
  unlink($octomp);
}

$bodyGen = "../util/bodyGen.py";
@objectCounts = (100, 500, 1000, 2000, 4000, 10000, 25000);

print "building $n2n, $oct, and $octomp\n";
system("make");
if (!(-x $n2n)) {
  die "$myname: No binary $n2n, compile error?\n";
}

if (!(-x $oct)) {
  die "$myname: No binary $oct, compile error?\n";
}

if (!(-x $octomp)) {
  die "$myname: No binary $octomp, compile error?\n";
}

print "\nProduce CSV files for execution\n";

foreach $count(@objectCounts)
{
	system("$bodyGen --total $count");
}

# for spacing
print "  BEGIN EXECUTION \n\n";

# run a series of tests
# harvest data

foreach $count(@objectCounts)
{
	print "N2N $count:\n";
	system("$n2n galaxy_$count.csv");
	print "\n";
}

foreach $count(@objectCounts)
{
	print "OCTREE $count:\n";
	system("$oct galaxy_$count.csv");
	print "\n";
}

foreach $count(@objectCounts)
{
	print "OCTREE_OMP $count:\n";
	system("$octomp galaxy_$count.csv");
	print "\n";
}

print "clean up\n";
system("rm *.csv");
system("make clean");

print "run_sim complete\n";
