#ifndef _TYPES_H
#define _TYPES_H

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

#endif
