: # -*- perl -*-
  eval 'exec perl -S  $0 ${1+"$@"}' 
    if 0;  # if running under some shell
#
# Copied from /Lab3/scripts/combineBuild and modified

system("clear");

$myname = "threads_compare";
$final  = "threads_compare.csv";

$cwd = `pwd`; chomp $cwd;
if (!($cwd =~ m|/cpu$|)) {
  die "$myname: Curdir should be the 'cpu' directory.\n";
}

# make binaries

print "Making binaries\n";
@binaries = qw(OCTREE_T1 OCTREE_T2 OCTREE_T4 OCTREE_T8);
@threads  = (1, 2, 4, 8);
$make_name = "OCTREE_OMP";
$bin_pref  = "OCTREE_T";

system("rm $bin_pref*");

foreach $thread(@threads)
{
	system("make octree_omp THREADS=$thread");
	system("mv $make_name $bin_pref$thread");
}

# make files for simulation and result capture

print "\nCreate CSV files for exectuion and capturing results\n";

$bodyGen = "../util/bodyGen.py";
@objectCounts = (100, 500, 1000, 2000, 4000, 6000, 10000, 15000, 25000);

$file = "results.csv";
if(-e $file){
	system("rm $file");
}
system("touch $file");
open($fp, '>>', $file) or die "perl failed to open $file.";
print $fp "Threads, ";

foreach $count(@objectCounts)
{
	system("$bodyGen --total $count");
	print $fp "$count, ";
}

close $fp;

# spacing on screen
print "  BEGIN EXECUTION \n\n";

foreach $binary(@binaries)
{
	open($fp, ">>", $file) or die "perl failed to open $file in loop.\n";
	print $fp "\n$binary, ";
	close $fp;
	foreach $count(@objectCounts)
	{
		print "$binary $count:\n";
		system("./$binary galaxy_$count.csv");
		print "\n";
	}
}


print "clean up\n";
system("make clean");
system("mv $file $final");
system("rm galaxy_*.csv");
system("rm $bin_pref*");

print "$myname complete\n";
