#ifndef _FDECLS_H
#define _FDECLS_H

#include "types.h"

static inline double now_ms(void);

void handle_sigint(int signum);

int move_obj_to_gpu(gpu_mem_t* m);

void cleanup_gpu_buf(gpu_mem_t* m);

static inline void advance_rotation(fvec3 axis, mat_3x3f_t* m, float theta);

void main_loop(const gpu_mem_t m);

int allocate_gpu_mem(gpu_mem_t* m);

#endif
