TEST_PRINT    	 = off
TIMING_ACTIVE 	 = on
CPE_ACTIVE    	 = on
CSV_ACTIVE	  	 = off
VALGRIND_ACTIVE  = off

CC = gcc
OPT = -O3
FLAGS = -Wall
LINK  = -lrt -lm

N2N = N2N_CPU
N2N_OBJS = cpu_n2n.o timing.o

OCT = OCTREE_CPU
OCT_OBJS = cpu_octree.o octree.o timing.o

OCT_OMP = OCTREE_OMP
OCT_OMP_OBJS = cpu_octree_omp.o octree.o timing.o

THREAD_DEFINES = -DTHREAD_ACTIVE

ifeq ($(TEST_PRINT), on)
	THREAD_DEFINES += -DTEST_PRINT
	SERIAL_DEFINES += -DTEST_PRINT
endif

ifeq ($(TIMING_ACTIVE), on)
	THREAD_DEFINES += -DTIMING_ACTIVE
	SERIAL_DEFINES += -DTIMING_ACTIVE
endif

ifeq ($(CPE_ACTIVE), on)
	THREAD_DEFINES += -DCPE_ACTIVE
	SERIAL_DEFINES += -DCPE_ACTIVE
endif

ifeq ($(CSV_ACTIVE), on)
	THREAD_DEFINES += -DCSV_ACTIVE
	SERIAL_DEFINES += -DCSV_ACTIVE
endif

ifeq ($(VALGRIND_ACTIVE), on)
	FLAGS += -g
endif

all: n2n octree octree_omp

n2n: main_n2n.c cpu_n2n.o timing.o
	$(CC) $(OPT) $(FLAGS) main_n2n.c $(N2N_OBJS) -o $(N2N) $(LINK) $(SERIAL_DEFINES)

octree: main_octree.c cpu_octree.o octree.o timing.o
	$(CC) $(OPT) $(FLAGS) main_octree.c $(OCT_OBJS) -o $(OCT) $(LINK) $(SERIAL_DEFINES)

octree_omp: main_octree.c cpu_octree_omp.o octree.o timing.o
	$(CC) $(OPT) $(FLAGS) $(INTEL) -fopenmp main_octree.c $(OCT_OMP_OBJS) -o $(OCT_OMP) $(LINK) $(THREAD_DEFINES)

cpu_n2n.o: cpu_n2n.c
	$(CC) $(OPT) $(FLAGS) -c cpu_n2n.c

octree.o: octree.c
	$(CC) $(OPT) $(FLAGS) -c octree.c

cpu_octree.o: octree.o cpu_octree.c
	$(CC) $(OPT) $(FLAGS) octree.o -c cpu_octree.c

cpu_octree_omp.o: octree.o cpu_octree.c
	$(CC) $(OPT) $(FLAGS) -fopenmp octree.o -c cpu_octree.c -o cpu_octree_omp.o $(THREAD_DEFINES)

timing.o: timing.c
	$(CC) $(OPT) $(FLAGS) -c timing.c -lrt

rebuild: clean all

clean:
	rm *.o $(N2N) $(OCT) $(OCT_VEC) $(OCT_OMP)