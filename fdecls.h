#ifndef _FDECLS_H
#define _FDECLS_H

#include "types.h"

static inline double now_ms(void);

void handle_sigint(int signum);

void move_obj_to_gpu(void** d_v, void** d_f);

void cleanup_gpu_buf(void** d_v, void** d_f);

static inline void advance_rotation(fvec3 axis, mat_3x3f_t* m, float theta);

void main_loop();

#endif
