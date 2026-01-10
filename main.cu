#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <time.h>

#include "fdecls.h"
#include "penger.h"
#include "types.h"

#define WIDTH 1280
#define HEIGHT 720
#define PI 3.141592f
#define PI_2 6.283185f
#define ANGULAR_VELOCITY 1
#define LINE_COLOR 0x00ff00ff

volatile sig_atomic_t should_run = 1;
double global_start_ms;
double global_end_ms;
size_t frame_cnt;

__global__ void k_transform_vertices(mat_3x3f_t m, gpu_mem_t data_buf,
                                     size_t num_verts)
{
        const Vec3* old_v = (const Vec3*)data_buf.d_verts;
        Vec3* new_v = (Vec3*)data_buf.d_verts_transform;
        size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        size_t total_threads = blockDim.x * gridDim.x;

        for (size_t idx = tid; idx < num_verts; idx += total_threads)
        {
                Vec3 in = old_v[idx];
                Vec3 out;

                out.x = m.data[0][0] * in.x + m.data[0][1] * in.y +
                        m.data[0][2] * in.z;
                out.y = m.data[1][0] * in.x + m.data[1][1] * in.y +
                        m.data[1][2] * in.z;
                out.z = m.data[2][0] * in.x + m.data[2][1] * in.y +
                        m.data[2][2] * in.z;

                new_v[idx] = out;
        }
}

__global__ void k_render_lines(gpu_mem_t data_buf, size_t num_faces)
{
        const Vec3* v = (const Vec3*)data_buf.d_verts_transform;
        const Tri* f = (const Tri*)data_buf.d_faces;
        size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        size_t total_threads = blockDim.x * gridDim.x;

        for (size_t idx = tid; idx < num_faces; idx += total_threads)
        {
                Tri tri_indices = f[idx];
                Vec3 tri_vertices[3];
                pixel_t tri_projection[3];

                tri_vertices[0] = v[tri_indices.a];
                tri_vertices[1] = v[tri_indices.b];
                tri_vertices[2] = v[tri_indices.c];

                float scale = 1.0f;  // adjust to fit the model in screen

                // camera is at 2,0,0, direction -1,0,0
                // screen is at (1,-1,-1) to (1,1,1)

                // project the 3 vertices onto the screen
                //
                // Fill in the 3 lines defined (Bresenham's)
                //
                // Write directly to the output buffer?
        }
}

void handle_sigint(int signum)
{
        should_run = 0;
        global_end_ms = now_ms();
        double duration = (global_end_ms - global_start_ms) / 1000;
        double fps = frame_cnt / duration;
        printf("Computed %ld frames in %lf seconds, averaging %lf FPS\n",
               frame_cnt, duration, fps);
}

int allocate_gpu_mem(gpu_mem_t* m)
{
        cudaError_t ret;
        ret = cudaMalloc(&m->d_verts, sizeof(model_v));
        if (ret != cudaSuccess)
        {
                printf("Failed to allocate vertex buffer on GPU!\n");
                return -1;
        }

        ret = cudaMalloc(&m->d_faces, sizeof(model_f));
        if (ret != cudaSuccess)
        {
                printf("Failed to allocate face buffer on GPU!\n");
                return -1;
        }

        ret = cudaMalloc(&m->d_verts_transform, sizeof(model_v));
        if (ret != cudaSuccess)
        {
                printf(
                    "Failed to allocate transformed vertex buffer on GPU!\n");
                return -1;
        }

        ret = cudaMalloc(&m->d_frame_buf, WIDTH * HEIGHT * sizeof(color_t));
        if (ret != cudaSuccess)
        {
                printf("Failed to allocate frame buffer on GPU!\n");
                return -1;
        }
        return 0;
}

int move_obj_to_gpu(gpu_mem_t* m)
{
        cudaError_t ret;
        ret = cudaMemcpy(m->d_verts, model_v, sizeof(model_v),
                         cudaMemcpyHostToDevice);
        if (ret != cudaSuccess)
        {
                printf("Failed to copy vertex buffer to GPU!\n");
                return -1;
        }

        ret = cudaMemcpy(m->d_faces, model_f, sizeof(model_f),
                         cudaMemcpyHostToDevice);
        if (ret != cudaSuccess)
        {
                printf("Failed to copy face buffer to GPU!\n");
                return -1;
        }
        return 0;
}

void cleanup_gpu_buf(gpu_mem_t* m)
{
        cudaFree(m->d_frame_buf);
        cudaFree(m->d_faces);
        cudaFree(m->d_verts);
        cudaFree(m->d_verts_transform);
        *m = {NULL, NULL, NULL, NULL};
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

void main_loop(const gpu_mem_t m_context)
{
        float angle = 0.0f;
        fvec3 rot_axis{1, 0, 0};  // would be nice to rotate this too!
        mat_3x3f_t m;

        size_t num_verts = sizeof(model_v) / sizeof(model_v[0]);
        size_t num_faces = sizeof(model_f) / sizeof(model_f[0]);

        double start_ms = now_ms();
        double end_ms = now_ms();
        while (should_run)
        {
                start_ms = now_ms();
                advance_rotation(rot_axis, &m, angle);

                frame_cnt++;

                // launch the vertex transformation kernel
                size_t threads_per_block = 256;
                size_t blocks_v =
                    (num_verts + threads_per_block - 1) / threads_per_block;
                k_transform_vertices<<<blocks_v, threads_per_block>>>(
                    m, m_context, num_verts);

                cudaDeviceSynchronize();

                // launch the line draw kernel
                // k_render_lines<<<1, 1>>>();

                // cudaDeviceSynchronize();

                // maybe display, we're mostly after frames for now?

                end_ms = now_ms();
                angle += (end_ms - start_ms) * PI_2 * ANGULAR_VELOCITY / 1000;
                // angle = fmodf(angle, PI_2);
        }
}

int main(void)
{
        gpu_mem_t m_context{NULL, NULL, NULL, NULL};
        if (allocate_gpu_mem(&m_context) == -1) exit(-1);
        if (move_obj_to_gpu(&m_context) == -1) exit(-1);

        signal(SIGINT, handle_sigint);
        global_start_ms = now_ms();
        main_loop(m_context);

        cleanup_gpu_buf(&m_context);

        return 0;
}
