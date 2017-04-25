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

print "building $n2n and $oct\n";
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

print "Produce CSV files for execution";
# 100
system("../util/bodyGen.py --star 5 --planet 25 --giant 10 --moon 30 --minor 30");

# 500
system("../util/bodyGen.py --star 50 --planet 250 --giant 50 --moon 50 --minor 100");

# 1000
system("../util/bodyGen.py --star 150 --planet 300 --giant 350 --moon 100 --minor 100");

# 2000
system("../util/bodyGen.py --star 300 --planet 400 --giant 300 --moon 200 --minor 800");

# 7000
system("../util/bodyGen.py --star 400 --planet 1250 --giant 750 --moon 1600 --minor 3000");

# for spacing
print "\n\n  BEGIN EXECUTION \n\n";

# run a series of tests
# harvest data
print "N2N 100:\n";
system("$n2n galaxy_100.csv");
print "\n";

print "N2N 500:\n";
system("$n2n galaxy_500.csv");
print "\n";

print "N2N 1000:\n";
system("$n2n galaxy_1000.csv");
print "\n";

print "N2N 2000:\n";
system("$n2n galaxy_2000.csv");
print "\n";

print "N2N 7000:\n";
system("$n2n galaxy_7000.csv");
print "\n";

print "OCTREE 100:\n";
system("$oct galaxy_100.csv");
print "\n";

print "OCTREE 500:\n";
system("$oct galaxy_500.csv");
print "\n";

print "OCTREE 1000:\n";
system("$oct galaxy_1000.csv");
print "\n";

print "OCTREE 2000:\n";
system("$oct galaxy_2000.csv");
print "\n";

print "OCTREE 7000:\n";
system("$oct galaxy_7000.csv");
print "\n";

print "OCTREE_OMP 100:\n";
system("$octomp galaxy_100.csv");
print "\n";

print "OCTREE_OMP 500:\n";
system("$octomp galaxy_500.csv");
print "\n";

print "OCTREE_OMP 1000:\n";
system("$octomp galaxy_1000.csv");
print "\n";

print "OCTREE_OMP 2000:\n";
system("$octomp galaxy_2000.csv");
print "\n";

print "OCTREE_OMP 7000:\n";
system("$octomp galaxy_7000.csv");
print "\n";

print "clean up\n";
system("rm *.csv");
system("make clean");

print "run_sim complete\n";