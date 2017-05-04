#ifndef TIMING_H
#define TIMING_H

#include <time.h>
#include <stdio.h>


// rdtsc setup
typedef union {
  unsigned long long int64;
  struct {unsigned int lo, hi;} int32;
} mcps_tctr;

#define MCPS_RDTSC(cpu_c) __asm__ __volatile__ ("rdtsc" : \
                     "=a" ((cpu_c).int32.lo), "=d"((cpu_c).int32.hi))

int    			clock_gettime(clockid_t clk_id, struct timespec *tp);
struct timespec ts_diff(struct timespec start, struct timespec end);
double          double_diff(struct timespec start, struct timespec end);
double 			measure_cps(void);
double 			CPE_calculate(double ns_iter, int elements);

#endif