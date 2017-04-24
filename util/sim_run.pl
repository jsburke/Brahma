: # -*- perl -*-
  eval 'exec perl -S  $0 ${1+"$@"}' 
    if 0;  # if running under some shell
#
# Copied from /Lab3/scripts/combineBuild and modified

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

print "building $n2n and $oct\n";
system("make");
if (!(-x $n2n)) {
  die "$myname: No binary $n2n, compile error?\n";
}

if (!(-x $oct)) {
  die "$myname: No binary $oct, compile error?\n";
}

# run a series of tests
# harvest data
print "N2N 105:\n";
system("$n2n galaxy_105.csv");
print "\n";

print "N2N 500:\n";
system("$n2n galaxy_500.csv");
print "\n";

print "N2N 1000:\n";
system("$n2n galaxy_1000.csv");
print "\n";

print "N2N 7000:\n";
system("$n2n galaxy_7000.csv");
print "\n";

print "OCTREE 105:\n";
system("$oct galaxy_105.csv");
print "\n";

print "OCTREE 500:\n";
system("$oct galaxy_500.csv");
print "\n";

print "OCTREE 1000:\n";
system("$oct galaxy_1000.csv");
print "\n";

print "OCTREE 7000:\n";
system("$oct galaxy_7000.csv");
print "\n";

print "clean up\n";
system("make clean");

print "run_sim complete\n";