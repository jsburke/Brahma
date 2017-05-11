DEFAULT_TIME_STEP  = off
DEFAULT_EXIT_COUNT = off
CPU_VERIFY         = off

ITERS			   = 100
TIME 			   = 5*TIME_HOUR

CC                 = nvcc
OPT 			   = -O3
LINK 			   = -lrt -lm

GPU = GRAV_GPU
GPU_OBJS = grav_gpu.o

TIME_STEP          = -DTIME_STEP=$(TIME)
EXIT_COUNT         = -DEXIT_COUNT=$(ITERS)

ifeq ($(DEFAULT_TIME_STEP), off)
	ANCILLARY += $(TIME_STEP)
endif

ifeq ($(DEFAULT_EXIT_COUNT), off)
	ANCILLARY += $(EXIT_COUNT)
endif

ifeq ($(CPU_VERIFY), on)
	ANCILLARY += -DCPU_ON
endif

all: gpu

gpu: grav_gpu.o
	$(CC) $(OPT) $(GPU_OBJS) -o $(GPU) gpu_n2n.cu $(ANCILLARY)

grav_gpu.o: grav_gpu.cu
	$(CC) $(OPT) grav_gpu.cu -c $(LINK)

rebuild: clean all

clean:
	rm *.o $(GPU)