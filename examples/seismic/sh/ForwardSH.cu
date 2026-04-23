/*
 * ForwardSH.cu
 * CUDA port of Devito-generated OpenACC ForwardSH operator.
 *
 * Ported from: Devito OpenACC output (nvidiaX platform, space_order=4)
 *
 * Compile (Linux/WSL2):
 *   nvcc -O3 -arch=sm_75 -shared -Xcompiler -fPIC -o ForwardSH.so ForwardSH.cu
 *
 * Compile (Windows, MSVC host):
 *   nvcc -O3 -arch=sm_75 --shared -o ForwardSH.dll ForwardSH.cu
 *
 * NOTE: sm_75 = Turing (RTX 20xx). Adjust -arch for your GPU:
 *   sm_61 = Pascal (GTX 10xx)
 *   sm_75 = Turing (RTX 20xx)
 *   sm_86 = Ampere (RTX 30xx)
 *   sm_89 = Ada    (RTX 40xx)
 *
 * CORRECTNESS STATUS: compilable, not yet validated against CPU reference.
 * Known issues to verify:
 *   - Kernel launch bounds (BLOCK_X * BLOCK_Y should be <= 1024)
 *   - Source injection atomicAdd is float — fine for single source,
 *     review if p_src_M > 0 with overlapping stencil footprints
 *   - Profiler uses QueryPerformanceCounter (Windows) or clock_gettime (Linux)
 */

#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

/* -------------------------------------------------------------------------
 * Platform timing  (replaces gettimeofday which doesn't exist on Windows)
 * ---------------------------------------------------------------------- */
#ifdef _WIN32
  #include <windows.h>
  static double _qpc_freq = 0.0;
  static void _init_timer() {
      LARGE_INTEGER f; QueryPerformanceFrequency(&f); _qpc_freq = (double)f.QuadPart;
  }
  static double _now_sec() {
      LARGE_INTEGER c; QueryPerformanceCounter(&c); return (double)c.QuadPart / _qpc_freq;
  }
#else
  #include <time.h>
  static void _init_timer() {}
  static double _now_sec() {
      struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
      return ts.tv_sec + ts.tv_nsec * 1e-9;
  }
#endif

#define START(S) double _t0_##S = _now_sec();
#define STOP(S,T) T->S += _now_sec() - _t0_##S;

/* -------------------------------------------------------------------------
 * Devito ABI structs  (must match the Python ctypes layout exactly)
 * ---------------------------------------------------------------------- */
struct dataobj {
    void * __restrict data;
    int *             size;
    unsigned long     nbytes;
    unsigned long *   npsize;
    unsigned long *   dsize;
    int *             hsize;
    int *             hofs;
    int *             oofs;
    void *            dmap;
};

struct profiler {
    double section0;
    double section1;
    double section2;
};

/* -------------------------------------------------------------------------
 * CUDA error helper
 * ---------------------------------------------------------------------- */
#define CUDA_CHECK(call) do {                                        \
    cudaError_t _e = (call);                                         \
    if (_e != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(_e));         \
        return -1;                                                   \
    }                                                                \
} while(0)

/* -------------------------------------------------------------------------
 * Kernel launch config
 * ---------------------------------------------------------------------- */
#define BLOCK_X 16
#define BLOCK_Y 16

/* -------------------------------------------------------------------------
 * Flat index helpers for 3-D rolling buffer: [t][x][y]
 * shape: (2, nx_total, ny_total)
 * ---------------------------------------------------------------------- */
__host__ __device__ __forceinline__
int idx3(int t, int x, int y, int nx, int ny) {
    return t * nx * ny + x * ny + y;
}

/* -------------------------------------------------------------------------
 * Kernel 1: velocity update
 *   v[t1][x+4][y+4] = ...
 * ---------------------------------------------------------------------- */
__global__ void kernel_update_v(
    float * __restrict__ v,
    const float * __restrict__ tau_xy,
    const float * __restrict__ tau_zy,
    const float * __restrict__ b,
    const float * __restrict__ damp,
    const int x_m, const int x_M,
    const int y_m, const int y_M,
    const int nx_total, const int ny_total,
    const int t0, const int t1,
    const float dt, const float r1)
{
    int x = x_m + blockIdx.x * BLOCK_X + threadIdx.x;
    int y = y_m + blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x > x_M || y > y_M) return;

    int xs = x + 4;   /* shifted index into padded array */
    int ys = y + 4;

    float val =
        dt * (
            r1 * v[idx3(t0, xs, ys, nx_total, ny_total)]
            + (
                1.04166667e-1F * (
                      tau_xy[idx3(t0, xs-2, ys,   nx_total, ny_total)]
                    - tau_xy[idx3(t0, xs+1, ys,   nx_total, ny_total)]
                    + tau_zy[idx3(t0, xs,   ys-2, nx_total, ny_total)]
                    - tau_zy[idx3(t0, xs,   ys+1, nx_total, ny_total)]
                )
                + 2.81250F * (
                    - tau_xy[idx3(t0, xs-1, ys,   nx_total, ny_total)]
                    + tau_xy[idx3(t0, xs,   ys,   nx_total, ny_total)]
                    - tau_zy[idx3(t0, xs,   ys-1, nx_total, ny_total)]
                    + tau_zy[idx3(t0, xs,   ys,   nx_total, ny_total)]
                )
            ) * b[xs * ny_total + ys]
        ) * damp[xs * ny_total + ys];

    v[idx3(t1, xs, ys, nx_total, ny_total)] = val;
}

/* -------------------------------------------------------------------------
 * Kernel 2: stress update
 *   tau_xy[t1][x+4][y+4] = ...
 *   tau_zy[t1][x+4][y+4] = ...
 * Both written in one kernel to avoid a second launch overhead.
 * Reads v[t1] written by kernel_update_v — must be launched AFTER it.
 * ---------------------------------------------------------------------- */
__global__ void kernel_update_tau(
    float * __restrict__ tau_xy,
    float * __restrict__ tau_zy,
    const float * __restrict__ v,
    const float * __restrict__ mu_x,
    const float * __restrict__ mu_z,
    const float * __restrict__ damp,
    const int x_m, const int x_M,
    const int y_m, const int y_M,
    const int nx_total, const int ny_total,
    const int t0, const int t1,
    const float dt, const float r1)
{
    int x = x_m + blockIdx.x * BLOCK_X + threadIdx.x;
    int y = y_m + blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x > x_M || y > y_M) return;

    int xs = x + 4;
    int ys = y + 4;

    /* tau_xy */
    tau_xy[idx3(t1, xs, ys, nx_total, ny_total)] =
        5.0e-1F * dt * (
            r1 * tau_xy[idx3(t0, xs, ys, nx_total, ny_total)]
            + (
                1.04166667e-1F * (
                      v[idx3(t1, xs-1, ys, nx_total, ny_total)]
                    - v[idx3(t1, xs+2, ys, nx_total, ny_total)]
                )
                + 2.81250F * (
                    - v[idx3(t1, xs,   ys, nx_total, ny_total)]
                    + v[idx3(t1, xs+1, ys, nx_total, ny_total)]
                )
            ) * mu_x[xs * ny_total + ys]
        ) * (damp[xs * ny_total + ys] + damp[(xs+1) * ny_total + ys]);

    /* tau_zy */
    tau_zy[idx3(t1, xs, ys, nx_total, ny_total)] =
        5.0e-1F * dt * (
            r1 * tau_zy[idx3(t0, xs, ys, nx_total, ny_total)]
            + (
                1.04166667e-1F * (
                      v[idx3(t1, xs, ys-1, nx_total, ny_total)]
                    - v[idx3(t1, xs, ys+2, nx_total, ny_total)]
                )
                + 2.81250F * (
                    - v[idx3(t1, xs, ys,   nx_total, ny_total)]
                    + v[idx3(t1, xs, ys+1, nx_total, ny_total)]
                )
            ) * mu_z[xs * ny_total + ys]
        ) * (damp[xs * ny_total + ys] + damp[xs * ny_total + (ys+1)]);
}

/* -------------------------------------------------------------------------
 * Source injection  (CPU loop — typically 1 source, 4 bilinear points)
 * Runs on CPU with a small cudaMemcpy of the affected v slice.
 * For simplicity we do a full v[t1] D2H, update, H2D each timestep.
 * TODO: optimise with a tiny GPU kernel + atomicAdd if n_src is large.
 * ---------------------------------------------------------------------- */
static void inject_sources_cpu(
    float * v_host,       /* full v host buffer, shape (2, nx, ny) */
    float * v_dev,
    const float * src_host,
    const float * src_coords_host,
    const int p_src_m, const int p_src_M,
    const int x_m, const int x_M,
    const int y_m, const int y_M,
    const int nx_total, const int ny_total,
    const int t1,
    const float dt, const float o_x, const float o_y,
    const int time)
{
    /* Pull v[t1] from device */
    size_t slice = (size_t)nx_total * ny_total * sizeof(float);
    cudaMemcpy(v_host + t1 * nx_total * ny_total,
               v_dev  + t1 * nx_total * ny_total,
               slice, cudaMemcpyDeviceToHost);

    for (int p_src = p_src_m; p_src <= p_src_M; p_src++) {
        for (int rsrcx = 0; rsrcx <= 1; rsrcx++) {
            for (int rsrcy = 0; rsrcy <= 1; rsrcy++) {
                int posx = (int)floorf(2.50f * (-o_x + src_coords_host[p_src * 2 + 0]));
                int posy = (int)floorf(2.50f * (-o_y + src_coords_host[p_src * 2 + 1]));
                float px = 2.50f*(-o_x + src_coords_host[p_src*2+0])
                         - floorf(2.50f*(-o_x + src_coords_host[p_src*2+0]));
                float py = 2.50f*(-o_y + src_coords_host[p_src*2+1])
                         - floorf(2.50f*(-o_y + src_coords_host[p_src*2+1]));
                if (rsrcx+posx >= x_m-1 && rsrcy+posy >= y_m-1 &&
                    rsrcx+posx <= x_M+1 && rsrcy+posy <= y_M+1)
                {
                    /* src layout: (nt, n_src) row-major */
                    float r0 = dt
                        * (rsrcx*px + (1-rsrcx)*(1-px))
                        * (rsrcy*py + (1-rsrcy)*(1-py))
                        * src_host[time * (p_src_M - p_src_m + 1) + p_src];
                    int xi = rsrcx + posx + 4;
                    int yi = rsrcy + posy + 4;
                    v_host[idx3(t1, xi, yi, nx_total, ny_total)] += r0;
                }
            }
        }
    }

    /* Push v[t1] back */
    cudaMemcpy(v_dev  + t1 * nx_total * ny_total,
               v_host + t1 * nx_total * ny_total,
               slice, cudaMemcpyHostToDevice);
}

/* -------------------------------------------------------------------------
 * Receiver extraction  (CPU loop)
 * ---------------------------------------------------------------------- */
static void extract_receivers_cpu(
    float * rec_host,
    const float * v_host,
    const float * rec_coords_host,
    const int p_rec_m, const int p_rec_M,
    const int x_m, const int x_M,
    const int y_m, const int y_M,
    const int nx_total, const int ny_total,
    const int t0,
    const float o_x, const float o_y,
    const int time)
{
    for (int p_rec = p_rec_m; p_rec <= p_rec_M; p_rec++) {
        float r4 = 2.50f * (-o_x + rec_coords_host[p_rec*2+0]);
        float r2 = floorf(r4);
        int posx = (int)r2;
        float r5 = 2.50f * (-o_y + rec_coords_host[p_rec*2+1]);
        float r3 = floorf(r5);
        int posy = (int)r3;
        float px = -r2 + r4;
        float py = -r3 + r5;
        float sum = 0.0f;

        for (int rrecx = 0; rrecx <= 1; rrecx++) {
            for (int rrecy = 0; rrecy <= 1; rrecy++) {
                if (rrecx+posx >= x_m-1 && rrecy+posy >= y_m-1 &&
                    rrecx+posx <= x_M+1 && rrecy+posy <= y_M+1)
                {
                    sum += (rrecx*px + (1-rrecx)*(1-px))
                         * (rrecy*py + (1-rrecy)*(1-py))
                         * v_host[idx3(t0, rrecx+posx+4, rrecy+posy+4, nx_total, ny_total)];
                }
            }
        }
        rec_host[time * (p_rec_M - p_rec_m + 1) + p_rec] = sum;
    }
}

/* =========================================================================
 * Public entry point — matches the Python ctypes wrapper signature
 * (deviceid / devicerm added to match OpenACC version)
 * ======================================================================= */
extern "C" int ForwardSH(
    struct dataobj * __restrict__ b_vec,
    struct dataobj * __restrict__ damp_vec,
    struct dataobj * __restrict__ mu_x_vec,
    struct dataobj * __restrict__ mu_z_vec,
    struct dataobj * __restrict__ rec_vec,
    struct dataobj * __restrict__ rec_coords_vec,
    struct dataobj * __restrict__ src_vec,
    struct dataobj * __restrict__ src_coords_vec,
    struct dataobj * __restrict__ tau_xy_vec,
    struct dataobj * __restrict__ tau_zy_vec,
    struct dataobj * __restrict__ v_vec,
    const int x_M, const int x_m,
    const int y_M, const int y_m,
    const float dt, const float o_x, const float o_y,
    const int p_rec_M, const int p_rec_m,
    const int p_src_M, const int p_src_m,
    const int time_M, const int time_m,
    const int deviceid, const int devicerm,
    struct profiler * timers)
{
    _init_timer();

    /* --- select GPU ---------------------------------------------------- */
    if (deviceid >= 0) {
        CUDA_CHECK(cudaSetDevice(deviceid));
    }

    /* --- unpack host pointers ------------------------------------------ */
    float * b_h          = (float*)b_vec->data;
    float * damp_h       = (float*)damp_vec->data;
    float * mu_x_h       = (float*)mu_x_vec->data;
    float * mu_z_h       = (float*)mu_z_vec->data;
    float * rec_h        = (float*)rec_vec->data;
    float * rec_coords_h = (float*)rec_coords_vec->data;
    float * src_h        = (float*)src_vec->data;
    float * src_coords_h = (float*)src_coords_vec->data;
    float * tau_xy_h     = (float*)tau_xy_vec->data;
    float * tau_zy_h     = (float*)tau_zy_vec->data;
    float * v_h          = (float*)v_vec->data;

    /* --- array dimensions ---------------------------------------------- */
    /* 2-D fields (2, nx_total, ny_total) */
    int nx_total = v_vec->size[1];
    int ny_total = v_vec->size[2];
    size_t n3    = (size_t)2 * nx_total * ny_total * sizeof(float);
    /* 2-D model fields (nx_total, ny_total) */
    size_t n2    = (size_t)nx_total * ny_total * sizeof(float);

    /* --- allocate device memory ---------------------------------------- */
    float *b_d, *damp_d, *mu_x_d, *mu_z_d;
    float *tau_xy_d, *tau_zy_d, *v_d;
    float *rec_d, *rec_coords_d, *src_d, *src_coords_d;

    CUDA_CHECK(cudaMalloc(&b_d,          n2));
    CUDA_CHECK(cudaMalloc(&damp_d,       n2));
    CUDA_CHECK(cudaMalloc(&mu_x_d,       n2));
    CUDA_CHECK(cudaMalloc(&mu_z_d,       n2));
    CUDA_CHECK(cudaMalloc(&tau_xy_d,     n3));
    CUDA_CHECK(cudaMalloc(&tau_zy_d,     n3));
    CUDA_CHECK(cudaMalloc(&v_d,          n3));

    size_t rec_bytes      = (size_t)rec_vec->size[0]       * rec_vec->size[1]       * sizeof(float);
    size_t rcoord_bytes   = (size_t)rec_coords_vec->size[0]* rec_coords_vec->size[1]* sizeof(float);
    size_t src_bytes      = (size_t)src_vec->size[0]       * src_vec->size[1]       * sizeof(float);
    size_t scoord_bytes   = (size_t)src_coords_vec->size[0]* src_coords_vec->size[1]* sizeof(float);

    CUDA_CHECK(cudaMalloc(&rec_d,        rec_bytes));
    CUDA_CHECK(cudaMalloc(&rec_coords_d, rcoord_bytes));
    CUDA_CHECK(cudaMalloc(&src_d,        src_bytes));
    CUDA_CHECK(cudaMalloc(&src_coords_d, scoord_bytes));

    /* --- H2D copies ----------------------------------------------------- */
    CUDA_CHECK(cudaMemcpy(b_d,          b_h,          n2,           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(damp_d,       damp_h,       n2,           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mu_x_d,       mu_x_h,       n2,           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mu_z_d,       mu_z_h,       n2,           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tau_xy_d,     tau_xy_h,     n3,           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tau_zy_d,     tau_zy_h,     n3,           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(v_d,          v_h,          n3,           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(rec_d,        rec_h,        rec_bytes,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(rec_coords_d, rec_coords_h, rcoord_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_d,        src_h,        src_bytes,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_coords_d, src_coords_h, scoord_bytes, cudaMemcpyHostToDevice));

    /* --- kernel launch config ------------------------------------------ */
    int nx_work = x_M - x_m + 1;
    int ny_work = y_M - y_m + 1;
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((nx_work + BLOCK_X - 1) / BLOCK_X,
              (ny_work + BLOCK_Y - 1) / BLOCK_Y);

    const float r1 = 1.0f / dt;

    /* --- time loop ------------------------------------------------------ */
    for (int time = time_m; time <= time_M; time++) {
        int t0 = time       % 2;
        int t1 = (time + 1) % 2;

        /* section0: stencil kernels */
        START(section0)

        kernel_update_v<<<grid, block>>>(
            v_d, tau_xy_d, tau_zy_d, b_d, damp_d,
            x_m, x_M, y_m, y_M, nx_total, ny_total,
            t0, t1, dt, r1);

        /* tau kernels read v[t1] — must sync after velocity kernel */
        cudaDeviceSynchronize();

        kernel_update_tau<<<grid, block>>>(
            tau_xy_d, tau_zy_d, v_d, mu_x_d, mu_z_d, damp_d,
            x_m, x_M, y_m, y_M, nx_total, ny_total,
            t0, t1, dt, r1);

        cudaDeviceSynchronize();
        STOP(section0, timers)

        /* section1: source injection (CPU path) */
        START(section1)
        if (p_src_M >= p_src_m) {
            inject_sources_cpu(
                v_h, v_d, src_h, src_coords_h,
                p_src_m, p_src_M, x_m, x_M, y_m, y_M,
                nx_total, ny_total, t1, dt, o_x, o_y, time);
        }
        STOP(section1, timers)

        /* section2: receiver extraction (CPU path — needs v[t0] on host) */
        START(section2)
        if (p_rec_M >= p_rec_m) {
            size_t slice = (size_t)nx_total * ny_total * sizeof(float);
            CUDA_CHECK(cudaMemcpy(
                v_h + t0 * nx_total * ny_total,
                v_d + t0 * nx_total * ny_total,
                slice, cudaMemcpyDeviceToHost));

            extract_receivers_cpu(
                rec_h, v_h, rec_coords_h,
                p_rec_m, p_rec_M, x_m, x_M, y_m, y_M,
                nx_total, ny_total, t0, o_x, o_y, time);
        }
        STOP(section2, timers)
    }

    /* --- D2H copyout ---------------------------------------------------- */
    CUDA_CHECK(cudaMemcpy(v_h,      v_d,      n3,        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tau_xy_h, tau_xy_d, n3,        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tau_zy_h, tau_zy_d, n3,        cudaMemcpyDeviceToHost));
    /* rec_h is written directly by extract_receivers_cpu (CPU path) — rec_d
       is never updated on the device, so we must NOT copy it back here. */

    /* --- free device memory -------------------------------------------- */
    if (devicerm) {
        cudaFree(b_d);  cudaFree(damp_d);
        cudaFree(mu_x_d); cudaFree(mu_z_d);
        cudaFree(tau_xy_d); cudaFree(tau_zy_d); cudaFree(v_d);
        cudaFree(rec_d); cudaFree(rec_coords_d);
        cudaFree(src_d); cudaFree(src_coords_d);
    }

    return 0;
}
