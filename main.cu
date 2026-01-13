#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <time.h>

#include "fdecls.h"
#include "snake.h"
#include "types.h"

#define WIDTH 1280
#define HEIGHT 720
#define MODEL_SCALE 1.0f
#define PI 3.141592f
#define PI_2 6.283185f
#define ANGULAR_VELOCITY 1
#define LINE_COLOR 0x00FF00FF
#define BLACK 0x00000000

volatile sig_atomic_t should_run = 1;
double global_start_ms;
double global_end_ms;
size_t frames_total;
gpu_mem_t gpu_context;

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

__host__ __device__ inline size_t get_edge_index(size_t v1, size_t v2)
{
        // szudsik
        if (v1 > v2)
        {
                size_t temp = v1;
                v1 = v2;
                v2 = temp;
        }

        return v1 + v2 * v2;
}

// No need for this to be __host__, but I added it simply because it CAN be
__host__ __device__ void draw_line(int x0, int y0, int x1, int y1,
                                   color_t* frame_buf, color_t color)
{
        // bresenham's
        int dx = abs(x1 - x0);
        int dy = abs(y1 - y0);
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        int err = dx - dy;

        for (;;)
        {
                frame_buf[y0 * WIDTH + x0] = color;
                if (x0 == x1 && y0 == y1) break;

                int e2 = 2 * err;
                if (e2 > -dy)
                {
                        err -= dy;
                        x0 += sx;
                }
                if (e2 < dx)
                {
                        err += dx;
                        y0 += sy;
                }
        }
}

__global__ void k_render_lines(gpu_mem_t data_buf, size_t num_faces)
{
        const Vec3* v = (const Vec3*)data_buf.d_verts_transform;
        const Tri* f = (const Tri*)data_buf.d_faces;
        size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        size_t total_threads = blockDim.x * gridDim.x;
        unsigned int* drawn = (unsigned int*)data_buf.d_drawn_edges;

        for (size_t idx = tid; idx < num_faces; idx += total_threads)
        {
                Tri tri_indices = f[idx];
                Vec3 tri_vertices[3];
                pixel_t tri_projection[3];

                for (int i = 0; i < 3; i++)
                {
                        tri_vertices[i] = v[tri_indices.vi[i]];
                }

                for (int i = 0; i < 3; i++)
                {
                        // Assumes vertices are normalized, and centered around
                        // the origin. The front direction is z. The camera is
                        // at 0, 0, 2 with direction 0, 0, -1.
                        float d = (2 - tri_vertices[i].z);
                        tri_projection[i].x = MODEL_SCALE * WIDTH *
                                              (1 + tri_vertices[i].x / d) / 2;
                        tri_projection[i].y = MODEL_SCALE * HEIGHT *
                                              (1 - tri_vertices[i].y / d) / 2;
                }

                for (int i = 0; i < 3; i++)
                {
                        // Also assumes that all faces are triangles
                        pixel_t p0 = tri_projection[i];
                        pixel_t p1 = tri_projection[(i + 1) % 3];
                        size_t edge_idx = get_edge_index(
                            tri_indices.vi[i], tri_indices.vi[(i + 1) % 3]);
                        if (atomicExch(&drawn[edge_idx], 1) == 0)
                        {
                                draw_line(p0.x, p0.y, p1.x, p1.y,
                                          (color_t*)data_buf.d_frame_buf,
                                          LINE_COLOR);
                        }
                }
        }
}

void handle_sigint(int signum)
{
        should_run = 0;
        global_end_ms = now_ms();
        double fps_val = fps(frames_total, global_start_ms, global_end_ms);
        printf("\nComputed %ld frames in %lf seconds, averaging %lf FPS\n",
               frames_total, (global_end_ms - global_start_ms) / 1000, fps_val);
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

        size_t drawn_edges_size = sizeof(model_v) / sizeof(model_v[0]);
        drawn_edges_size *= drawn_edges_size;
        drawn_edges_size *= sizeof(unsigned int);
        // V^2 Memory allocation. This is obviously wasteful, but it is the
        // simplest implementation. Besides, cuda processes allocate enough
        // memory for this, so it shouldn't be a problem for small objects. An
        // alternative would be to allocate 3 * F * sizeof(unsigned int) bytes
        // since that is the max amount of edges...
        ret = cudaMalloc(&m->d_drawn_edges, drawn_edges_size);
        if (ret != cudaSuccess)
        {
                printf("Failed to allocate drawn edge buffer on GPU!\n");
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

int move_obj_to_gpu(const gpu_mem_t* m)
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
        cudaFree(m->d_drawn_edges);
        *m = {nullptr, nullptr, nullptr, nullptr, nullptr};
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

static inline double fps(size_t frames, double start_ms, double end_ms)
{
        double duration = (end_ms - start_ms) / 1000;
        return frames / duration;
}

static inline double now_ms(void)
{
        // TODO
        // Find a way to make this cross platform!
        // Maybe use STL
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double ms = ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
        return ms;
}

void save_frame_buffer(void* gpu_frame, const char* filename)
{
        color_t* frame_buf = (color_t*)malloc(WIDTH * HEIGHT * sizeof(color_t));
        if (!frame_buf)
        {
                printf("Failed to allocate output framebuffer\n");
                return;
        }

        cudaError_t ret =
            cudaMemcpy(frame_buf, gpu_frame, WIDTH * HEIGHT * sizeof(color_t),
                       cudaMemcpyDeviceToHost);
        if (ret != cudaSuccess)
        {
                printf("Failed to copy framebuffer from GPU\n");
                free(frame_buf);
                return;
        }

        // TODO
        // cross platform :/
        FILE* fp = fopen(filename, "wb");
        if (!fp)
        {
                printf("Failed to open %s file\n", filename);
                free(frame_buf);
                return;
        }

        // header
        fprintf(fp, "P6\n%d %d \n255\n", WIDTH, HEIGHT);

        // data
        for (size_t y = 0; y < HEIGHT; y++)
        {
                for (size_t x = 0; x < WIDTH; x++)
                {
                        color_t p = frame_buf[y * WIDTH + x];
                        unsigned char r = (p >> 24) & 0xFF;
                        unsigned char g = (p >> 16) & 0xFF;
                        unsigned char b = (p >> 8) & 0xFF;
                        fwrite(&r, 1, 1, fp);
                        fwrite(&g, 1, 1, fp);
                        fwrite(&b, 1, 1, fp);
                }
        }
        fclose(fp);
        free(frame_buf);
}

void main_loop(const gpu_mem_t* m_context)
{
        float angle = 0.0f;
        fvec3 rot_axis{0, 1, 0};  // TODO make this vector rotate too!
        mat_3x3f_t m;

        size_t num_verts = sizeof(model_v) / sizeof(model_v[0]);
        size_t num_faces = sizeof(model_f) / sizeof(model_f[0]);

        double start_ms = now_ms();
        double end_ms = now_ms();
        double last_print = now_ms();
        size_t frames_temp = 0;

        size_t threads_per_block = 256;
        size_t blocks_v =
            (num_verts + threads_per_block - 1) / threads_per_block;
        size_t blocks_f =
            (num_faces + threads_per_block - 1) / threads_per_block;

        while (should_run)
        {
                start_ms = now_ms();
                advance_rotation(rot_axis, &m, angle);

                frames_total++;
                frames_temp++;

                // clear the frame buffer
                cudaMemset(m_context->d_frame_buf, BLACK,
                           WIDTH * HEIGHT * sizeof(color_t));

                // clear the drawn edges buffer
                size_t drawn_edges_size = sizeof(model_v) / sizeof(model_v[0]);
                drawn_edges_size *= drawn_edges_size;
                drawn_edges_size *= sizeof(unsigned int);
                cudaMemset(m_context->d_drawn_edges, 0, drawn_edges_size);

                // launch the vertex transformation kernel
                k_transform_vertices<<<blocks_v, threads_per_block>>>(
                    m, *m_context, num_verts);
                cudaDeviceSynchronize();

                // launch the line draw kernel
                k_render_lines<<<blocks_f, threads_per_block>>>(*m_context,
                                                                num_faces);
                cudaDeviceSynchronize();

                // TODO add live display somehow

                end_ms = now_ms();

                if (end_ms - last_print >= 1000)
                {
                        double fps_val = fps(frames_temp, last_print, end_ms);
                        printf("\rFPS: %.2f", fps_val);
                        fflush(stdout);
                        last_print = end_ms;
                        frames_temp = 0;
                }

                angle += (end_ms - start_ms) * PI_2 * ANGULAR_VELOCITY / 1000;
                // angle = fmodf(angle, PI_2);
        }
}

int main(void)
{
        if (allocate_gpu_mem(&gpu_context) == -1) exit(-1);
        if (move_obj_to_gpu(&gpu_context) == -1) exit(-1);

        signal(SIGINT, handle_sigint);
        global_start_ms = now_ms();

        main_loop(&gpu_context);

        if (gpu_context.d_frame_buf)
                save_frame_buffer(gpu_context.d_frame_buf, "output.ppm");

        cleanup_gpu_buf(&gpu_context);

        return 0;
}
