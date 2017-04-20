CC = gcc
OPT = -O1
FLAGS = -Wall
LINK  = -lrt -lm

N2N = N2N_CPU
N2N_OBJS = cpu_n2n.o 

OCT = OCTREE_CPU
OCT_OBJS = cpu_simple.o octree.o

all: n2n octree

n2n: main_n2n.c cpu_n2n.o
	$(CC) $(OPT) $(FLAGS) main_n2n.c $(N2N_OBJS) -o $(N2N) $(LINK)

cpu_n2n.o: cpu_n2n.c
	$(CC) $(OPT) $(FLAGS) -c cpu_n2n.c

octree: main_simple.c cpu_simple.o octree.o
	$(CC) $(OPT) $(FLAGS) main_simple.c $(OCT_OBJS) -o $(OCT)

octree.o: octree.c
	$(CC) $(OPT) $(FLAGS) -c octree.o

cpu_simple.o: octree.o cpu_simple.c
	$(CC) $(OPT) $(FLAGS) cpu_simple.c -c cpu_simple.o

rebuild:
	clean all

clean:
	rm *.o $(N2N) $(OCT)