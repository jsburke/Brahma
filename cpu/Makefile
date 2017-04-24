CC = gcc
OPT = -O3
FLAGS = -Wall
LINK  = -lrt -lm

N2N = N2N_CPU
N2N_OBJS = cpu_n2n.o timing.o

OCT = OCTREE_CPU
OCT_OBJS = cpu_octree.o octree.o timing.o

all: n2n octree

n2n: main_n2n.c cpu_n2n.o timing.o
	$(CC) $(OPT) $(FLAGS) main_n2n.c $(N2N_OBJS) -o $(N2N) $(LINK)

cpu_n2n.o: cpu_n2n.c
	$(CC) $(OPT) $(FLAGS) -c cpu_n2n.c

octree: main_octree.c cpu_octree.o octree.o timing.o
	$(CC) $(OPT) $(FLAGS) main_octree.c $(OCT_OBJS) -o $(OCT) $(LINK)

octree.o: octree.c
	$(CC) $(OPT) $(FLAGS) -c octree.c

cpu_octree.o: octree.o cpu_octree.c
	$(CC) $(OPT) $(FLAGS) octree.o -c cpu_octree.c

timing.o: timing.c
	$(CC) $(OPT) $(FLAGS) -c timing.c -lrt

rebuild:
	clean all

clean:
	rm *.o $(N2N) $(OCT)