#ifndef _TYPES_H
#define _TYPES_H

#include <stdint.h>

typedef struct gpu_mem_t
{
        void* d_verts;
        void* d_faces;
        void* d_drawn_edges;
        void* d_verts_transform;
        void* d_frame_buf;
} gpu_mem_t;

typedef struct quat_f_t
{
        float w;
        float x;
        float y;
        float z;
} quat_f_t;

typedef struct mat_3x3f_t
{
        float data[3][3];
} mat_3x3f_t;

typedef struct fvec3
{
        float x;
        float y;
        float z;
} fvec3;

typedef struct pixel_t
{
        size_t x;
        size_t y;
} pixel_t;

typedef uint32_t color_t;

#endif
