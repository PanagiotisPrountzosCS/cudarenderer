#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <time.h>

#include "fdecls.h"
#include "penger.h"
#include "types.h"

#define PI 3.141592f
#define PI_2 6.283185f
#define ANGULAR_VELOCITY 1  // rotations per second

volatile sig_atomic_t should_run = 1;
double global_start;
double global_end;
size_t frame_cnt;

void handle_sigint(int signum)
{
        should_run = 0;
        global_end = now_ms();
        double duration = (global_end - global_start) / 1000;
        double fps = frame_cnt / duration;
        printf("Computed %ld frames in %lf seconds, averaging %lf FPS\n",
               frame_cnt, duration, fps);
}

void move_obj_to_gpu(void** d_v, void** d_f)
{
        cudaError_t ret;
        ret = cudaMalloc(d_v, sizeof(model_v));
        if (ret != cudaSuccess) printf("Error allocating GPU vertex buffer\n");

        ret = cudaMalloc(d_f, sizeof(model_f));
        if (ret != cudaSuccess) printf("Error allocating GPU face buffer\n");

        ret =
            cudaMemcpy(*d_v, model_v, sizeof(model_v), cudaMemcpyHostToDevice);
        if (ret != cudaSuccess) printf("Error copying vertex buffer to GPU\n");

        ret =
            cudaMemcpy(*d_f, model_f, sizeof(model_f), cudaMemcpyHostToDevice);
        if (ret != cudaSuccess) printf("Error copying face buffer to GPU\n");
}

void cleanup_gpu_buf(void** d_v, void** d_f)
{
        cudaFree(*d_v);
        cudaFree(*d_f);
}

static inline void advance_rotation(fvec3 axis, mat_3x3f_t* m, float theta)
{
        quat_f_t q;

        float c = cos(theta / 2);
        float s = sin(theta / 2);

        q.w = c;
        q.x = s * axis.x;
        q.y = s * axis.y;
        q.z = s * axis.z;

        float xx = q.x * q.x;
        float yy = q.y * q.y;
        float zz = q.z * q.z;
        float xy = q.x * q.y;
        float xz = q.x * q.z;
        float yz = q.y * q.z;
        float wx = q.w * q.x;
        float wy = q.w * q.y;
        float wz = q.w * q.z;

        m->data[0][0] = 1.0f - 2.0f * (yy + zz);
        m->data[0][1] = 2.0f * (xy - wz);
        m->data[0][2] = 2.0f * (xz + wy);

        m->data[1][0] = 2.0f * (xy + wz);
        m->data[1][1] = 1.0f - 2.0f * (xx + zz);
        m->data[1][2] = 2.0f * (yz - wx);

        m->data[2][0] = 2.0f * (xz - wy);
        m->data[2][1] = 2.0f * (yz + wx);
        m->data[2][2] = 1.0f - 2.0f * (xx + yy);
}

static inline double now_ms(void)
{
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double ms = ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
        return ms;
}

void main_loop()
{
        float angle = 0.0f;
        fvec3 rot_axis{1, 0, 0};  // would be nice to rotate this too!
        mat_3x3f_t m;

        double start_ms = now_ms();
        double end_ms = now_ms();
        while (should_run)
        {
                start_ms = now_ms();
                advance_rotation(rot_axis, &m, angle);

				frame_cnt++;

                // call the vertex transformation kernel
                //
                // call the line draw kernel
                //
                // maybe display, we're mostly after frames for now?

                end_ms = now_ms();
                angle += (end_ms - start_ms) * PI_2 * ANGULAR_VELOCITY / 1000;
                angle = fmodf(angle, PI_2);
        }
}

int main(void)
{
        void* dev_vertices = NULL;
        void* dev_faces = NULL;
        move_obj_to_gpu(&dev_vertices, &dev_faces);

        signal(SIGINT, handle_sigint);
        global_start = now_ms();
        main_loop();

        cleanup_gpu_buf(&dev_vertices, &dev_faces);

        return 0;
}
