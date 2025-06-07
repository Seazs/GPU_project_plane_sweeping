#include "main.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cmath>


#define ZPlanes 256
#define ZNear 0.3f
#define ZFar 1.1f

// Camera parameters
__constant__ CamParams ref;
__constant__ CamParams cam;

#define CHK(code) \
do { \
    if ((code) != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s %s %i\n", \
                        cudaGetErrorString((code)), __FILE__, __LINE__); \
        goto Error; \
    } \
} while (0)

#define INDEX_2D(y, x, width) ((y) * (width) + (x))
#define INDEX_3D(z, y, x, height, width) ((z) * (height) * (width) + (y) * (width) + (x))



__global__ void sweep_kernel_double_Naive(
    float* cost_vol,             // [ZPlanes * H * W]
    const uint8_t* ref_img,      // [H * W]
    const uint8_t* tgt_img,
    int W, int H,
    int window)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= W || y >= H || z >= ZPlanes) return;

    double zval = ZNear * ZFar / (ZNear + ((double(z) / double(ZPlanes)) * (ZFar - ZNear)));

    // Backproject to 3D
    double Xr = (ref.K_inv[0] * x + ref.K_inv[1] * y + ref.K_inv[2]) * zval;
    double Yr = (ref.K_inv[3] * x + ref.K_inv[4] * y + ref.K_inv[5]) * zval;
    double Zr = (ref.K_inv[6] * x + ref.K_inv[7] * y + ref.K_inv[8]) * zval;

    // Ref 3D → World
    double X = ref.R_inv[0] * Xr + ref.R_inv[1] * Yr + ref.R_inv[2] * Zr - ref.t_inv[0];
    double Y = ref.R_inv[3] * Xr + ref.R_inv[4] * Yr + ref.R_inv[5] * Zr - ref.t_inv[1];
    double Z = ref.R_inv[6] * Xr + ref.R_inv[7] * Yr + ref.R_inv[8] * Zr - ref.t_inv[2];

    // World → Camera
    double Xp = cam.R[0] * X + cam.R[1] * Y + cam.R[2] * Z - cam.t[0];
    double Yp = cam.R[3] * X + cam.R[4] * Y + cam.R[5] * Z - cam.t[1];
    double Zp = cam.R[6] * X + cam.R[7] * Y + cam.R[8] * Z - cam.t[2];

    double x_proj = (cam.K[0] * Xp / Zp + cam.K[1] * Yp / Zp + cam.K[2]);
    double y_proj = (cam.K[3] * Xp / Zp + cam.K[4] * Yp / Zp + cam.K[5]);

    // Verification it's not out of bounds
	x_proj = x_proj < 0 || x_proj >= W ? 0 : round(x_proj);
	y_proj = y_proj < 0 || y_proj >= H ? 0 : round(y_proj);

    int x_p = (int)x_proj;
    int y_p = (int)y_proj;

    double cost = 0.0;
    int count = 0;
    int half = window / 2;

    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {
            int xr = x + dx, yr = y + dy;
            int xc = x_p + dx, yc = y_p + dy;
            if (xr >= 0 && xr < W && yr >= 0 && yr < H &&
                xc >= 0 && xc < W && yc >= 0 && yc < H) {
                int idx_ref = yr * W + xr;
                int idx_cam = yc * W + xc;
                cost += fabs((double)ref_img[idx_ref] - (double)tgt_img[idx_cam]);
                count++;
            }
        }
    }

    if (count > 0) cost /= count;

    int idx = z * H * W + y * W + x;
    cost_vol[idx] = fminf(cost_vol[idx], (float)cost);  // Simple min, not atomic
}


__global__ void sweep_kernel_float_Naive(
    float* cost_vol,             // [ZPlanes * H * W]
    const uint8_t* ref_img,      // [H * W]
    const uint8_t* tgt_img,
    int W, int H,
    int window)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= W || y >= H || z >= ZPlanes) return;

    float zval = ZNear * ZFar / (ZNear + ((float(z) / float(ZPlanes)) * (ZFar - ZNear)));

    // Backproject to 3D
    float Xr = (ref.K_inv[0] * x + ref.K_inv[1] * y + ref.K_inv[2]) * zval;
    float Yr = (ref.K_inv[3] * x + ref.K_inv[4] * y + ref.K_inv[5]) * zval;
    float Zr = (ref.K_inv[6] * x + ref.K_inv[7] * y + ref.K_inv[8]) * zval;

    // Ref 3D → World
    float X = ref.R_inv[0] * Xr + ref.R_inv[1] * Yr + ref.R_inv[2] * Zr - ref.t_inv[0];
    float Y = ref.R_inv[3] * Xr + ref.R_inv[4] * Yr + ref.R_inv[5] * Zr - ref.t_inv[1];
    float Z = ref.R_inv[6] * Xr + ref.R_inv[7] * Yr + ref.R_inv[8] * Zr - ref.t_inv[2];

    // World → Camera
    float Xp = cam.R[0] * X + cam.R[1] * Y + cam.R[2] * Z - cam.t[0];
    float Yp = cam.R[3] * X + cam.R[4] * Y + cam.R[5] * Z - cam.t[1];
    float Zp = cam.R[6] * X + cam.R[7] * Y + cam.R[8] * Z - cam.t[2];

    float x_proj = (cam.K[0] * Xp / Zp + cam.K[1] * Yp / Zp + cam.K[2]);
    float y_proj = (cam.K[3] * Xp / Zp + cam.K[4] * Yp / Zp + cam.K[5]);

    x_proj = x_proj < 0 || x_proj >= W ? 0 : roundf(x_proj);
	y_proj = y_proj < 0 || y_proj >= H ? 0 : roundf(y_proj);

    int x_p = (int)x_proj;
    int y_p = (int)y_proj;

    float cost = 0.0f;
    float count = 0.0f;
    int half = window / 2;

    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {
            int xr = x + dx, yr = y + dy;
            int xc = x_p + dx, yc = y_p + dy;
            if (xr >= 0 && xr < W && yr >= 0 && yr < H &&
                xc >= 0 && xc < W && yc >= 0 && yc < H) {
                int idx_ref = yr * W + xr;
                int idx_cam = yc * W + xc;
                cost += fabsf((float)ref_img[idx_ref] - (float)tgt_img[idx_cam]);
                count += 1.0f;
            }
        }
    }

    if (count > 0.0f) cost /= count;

    int idx = z * H * W + y * W + x;
    cost_vol[idx] = fminf(cost_vol[idx], cost);  // Simple min, not atomic
}



// Compute projection from (x, y, z) to (x_proj, y_proj)
__device__ void compute_projection(
    int x, int y, int z, int W, int H,
    float& x_proj, float& y_proj)
{
    float zval = ZNear * ZFar / (ZNear + ((float(z) / float(ZPlanes)) * (ZFar - ZNear)));
    float Xr = (ref.K_inv[0] * x + ref.K_inv[1] * y + ref.K_inv[2]) * zval;
    float Yr = (ref.K_inv[3] * x + ref.K_inv[4] * y + ref.K_inv[5]) * zval;
    float Zr = (ref.K_inv[6] * x + ref.K_inv[7] * y + ref.K_inv[8]) * zval;
    float X = ref.R_inv[0] * Xr + ref.R_inv[1] * Yr + ref.R_inv[2] * Zr - ref.t_inv[0];
    float Y = ref.R_inv[3] * Xr + ref.R_inv[4] * Yr + ref.R_inv[5] * Zr - ref.t_inv[1];
    float Z = ref.R_inv[6] * Xr + ref.R_inv[7] * Yr + ref.R_inv[8] * Zr - ref.t_inv[2];
    float Xp = cam.R[0] * X + cam.R[1] * Y + cam.R[2] * Z - cam.t[0];
    float Yp = cam.R[3] * X + cam.R[4] * Y + cam.R[5] * Z - cam.t[1];
    float Zp = cam.R[6] * X + cam.R[7] * Y + cam.R[8] * Z - cam.t[2];
    x_proj = (cam.K[0] * Xp / Zp + cam.K[1] * Yp / Zp + cam.K[2]);
    y_proj = (cam.K[3] * Xp / Zp + cam.K[4] * Yp / Zp + cam.K[5]);
}




// shared memory kernel for reference image only
__global__ void sweep_kernel_float_shared_REF(
    float* cost_vol,
    const uint8_t* ref_img,
    const uint8_t* tgt_img,
    int W, int H,
    int window)
{
    extern __shared__ uint8_t shared_mem[];
    
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int gid_x = blockIdx.x * blockDim.x + tid_x;
    const int gid_y = blockIdx.y * blockDim.y + tid_y;
    const int depth_idx = blockIdx.z;

    // Early bounds check
    if (gid_x >= W || gid_y >= H || depth_idx >= ZPlanes) return;

    const int radius = window >> 1;  // Using bit shift instead of division
    const int tile_width = blockDim.x + (radius << 1);
    const int tile_height = blockDim.y + (radius << 1);
    const int local_x = tid_x + radius;
    const int local_y = tid_y + radius;

    // Collaborative loading of shared memory with improved coalescing
    const int total_threads = blockDim.x * blockDim.y;
    const int thread_id = tid_y * blockDim.x + tid_x;
    const int total_elements = tile_width * tile_height;
    
    // Load reference image tile with better memory coalescing
    for (int elem_id = thread_id; elem_id < total_elements; elem_id += total_threads) {
        int tile_y = elem_id / tile_width;
        int tile_x = elem_id % tile_width;
        
        int src_x = blockIdx.x * blockDim.x + tile_x - radius;
        int src_y = blockIdx.y * blockDim.y + tile_y - radius;
        
        // Boundary handling with clamping
        src_x = max(0, min(src_x, W - 1));
        src_y = max(0, min(src_y, H - 1));
        
        shared_mem[elem_id] = ref_img[src_y * W + src_x];
    }

    __syncthreads();

    // Compute projection for this pixel and depth
    float proj_x, proj_y;
    compute_projection(gid_x, gid_y, depth_idx, W, H, proj_x, proj_y);

    
    if (proj_x < 0.0f || proj_x >= W || proj_y < 0.0f || proj_y >= H) {
        return;  // Skip this pixel entirely like CPU version
    }

    const int center_x = __float2int_rn(proj_x);  
    const int center_y = __float2int_rn(proj_y);

    // Optimized cost computation with loop unrolling hints
    float total_cost = 0.0f;
    int valid_pixels = 0;
    
    // Process window in a more cache-friendly pattern
    #pragma unroll 4
    for (int dy = -radius; dy <= radius; dy++) {
        const int ref_row = local_y + dy;
        const int tgt_row = center_y + dy;
        
        if (tgt_row >= 0 && tgt_row < H) {
            #pragma unroll 4
            for (int dx = -radius; dx <= radius; dx++) {
                const int ref_col = local_x + dx;
                const int tgt_col = center_x + dx;
                
                if (tgt_col >= 0 && tgt_col < W) {
                    const int ref_idx = ref_row * tile_width + ref_col;
                    const int tgt_idx = tgt_row * W + tgt_col;
                    
                    const float ref_val = __uint2float_rn(shared_mem[ref_idx]);
                    const float tgt_val = __uint2float_rn(tgt_img[tgt_idx]);
                    
                    total_cost += fabsf(ref_val - tgt_val);
                    valid_pixels++;
                }
            }
        }
    }

    // Compute final cost with improved numerical stability
    float final_cost = (valid_pixels > 0) ? __fdividef(total_cost, valid_pixels) : 255.0f;

    // Update cost volume with atomic min for safety (optional: can be removed if single-camera)
    const int output_idx = INDEX_3D(depth_idx, gid_y, gid_x, H, W);
    cost_vol[output_idx] = fminf(cost_vol[output_idx], final_cost);
}



// shared memory kernel for reference and target images
__global__ void sweep_kernel_float_shared_REF_TGT(
    float* cost_vol,
    const uint8_t* ref_img,
    const uint8_t* tgt_img,
    int W, int H,
    int window)
{
    const int radius = window >> 1;
    const int shared_width = blockDim.x + 2 * radius;
    const int shared_height = blockDim.y + 2 * radius;
    const int shared_size = shared_width * shared_height;

    extern __shared__ uint8_t shared_mem[];
    uint8_t* shared_ref = shared_mem;
    uint8_t* shared_tgt = shared_mem + shared_size;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int thread_id = ty * blockDim.x + tx;
    const int total_threads = blockDim.x * blockDim.y;

    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;
    const int zi = blockIdx.z;

    if (x >= W || y >= H || zi >= ZPlanes) return;

    // Compute projection for this pixel and depth
    float proj_x, proj_y;
    compute_projection(x, y, zi, W, H, proj_x, proj_y);

    // Clamp projected coordinates for safety
    int px = (proj_x < 0 || proj_x >= W) ? 0 : __float2int_rn(proj_x);
    int py = (proj_y < 0 || proj_y >= H) ? 0 : __float2int_rn(proj_y);

    // Collaborative loading of shared_ref and shared_tgt
    const int total_elements = shared_size;
    for (int idx = thread_id; idx < total_elements; idx += total_threads) {
        int local_y = idx / shared_width;
        int local_x = idx % shared_width;

        int global_rx = blockIdx.x * blockDim.x + local_x - radius;
        int global_ry = blockIdx.y * blockDim.y + local_y - radius;

        int global_tx = px + (local_x - (tx + radius));
        int global_ty = py + (local_y - (ty + radius));

        // Clamp to image boundaries
        global_rx = max(0, min(global_rx, W - 1));
        global_ry = max(0, min(global_ry, H - 1));
        global_tx = max(0, min(global_tx, W - 1));
        global_ty = max(0, min(global_ty, H - 1));

        shared_ref[local_y * shared_width + local_x] = ref_img[global_ry * W + global_rx];
        shared_tgt[local_y * shared_width + local_x] = tgt_img[global_ty * W + global_tx];
    }

    __syncthreads();

    // Compute SAD cost
    const int lx = tx + radius;
    const int ly = ty + radius;

    float cost = 0.0f;
    float count = 0.0f;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int ref_idx = (ly + dy) * shared_width + (lx + dx);
            int tgt_idx = ref_idx;  // same coords in shared memory

            cost += fabsf((float)shared_ref[ref_idx] - (float)shared_tgt[tgt_idx]);
            count += 1.0f;
        }
    }

    cost = (count > 0) ? cost / count : 255.0f;

    // Write to cost volume
    int out_idx = INDEX_3D(zi, y, x, H, W);
    cost_vol[out_idx] = fminf(cost_vol[out_idx], cost);
}


// Kernel for texture memory access
__global__ void sweep_kernel_float_texture(
    float* cost_vol,             // [ZPlanes * H * W]
    const uint8_t* ref_img,      // [H * W]
    cudaTextureObject_t texTgt,
    int W, int H,
    int window)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= W || y >= H || z >= ZPlanes) return;

    int half = window / 2;

    // Shared memory tile for reference image
    extern __shared__ uint8_t ref_tile[];
    int tile_w = blockDim.x + window - 1;
    int tile_h = blockDim.y + window - 1;

    // Load tile into shared memory
    for (int dy = threadIdx.y; dy < tile_h; dy += blockDim.y) {
        for (int dx = threadIdx.x; dx < tile_w; dx += blockDim.x) {
            int global_x = blockIdx.x * blockDim.x + dx - half;
            int global_y = blockIdx.y * blockDim.y + dy - half;
            if (global_x >= 0 && global_x < W && global_y >= 0 && global_y < H)
                ref_tile[dy * tile_w + dx] = ref_img[global_y * W + global_x];
            else
                ref_tile[dy * tile_w + dx] = 0;
        }
    }
    __syncthreads();

    float zval = ZNear * ZFar / (ZNear + ((float(z) / float(ZPlanes)) * (ZFar - ZNear)));

    // Backproject to 3D
    float Xr = (ref.K_inv[0] * x + ref.K_inv[1] * y + ref.K_inv[2]) * zval;
    float Yr = (ref.K_inv[3] * x + ref.K_inv[4] * y + ref.K_inv[5]) * zval;
    float Zr = (ref.K_inv[6] * x + ref.K_inv[7] * y + ref.K_inv[8]) * zval;

    // Ref 3D → World
    float X = ref.R_inv[0] * Xr + ref.R_inv[1] * Yr + ref.R_inv[2] * Zr - ref.t_inv[0];
    float Y = ref.R_inv[3] * Xr + ref.R_inv[4] * Yr + ref.R_inv[5] * Zr - ref.t_inv[1];
    float Z = ref.R_inv[6] * Xr + ref.R_inv[7] * Yr + ref.R_inv[8] * Zr - ref.t_inv[2];

    // World → Camera
    float Xp = cam.R[0] * X + cam.R[1] * Y + cam.R[2] * Z - cam.t[0];
    float Yp = cam.R[3] * X + cam.R[4] * Y + cam.R[5] * Z - cam.t[1];
    float Zp = cam.R[6] * X + cam.R[7] * Y + cam.R[8] * Z - cam.t[2];

    float x_proj = (cam.K[0] * Xp / Zp + cam.K[1] * Yp / Zp + cam.K[2]);
    float y_proj = (cam.K[3] * Xp / Zp + cam.K[4] * Yp / Zp + cam.K[5]);

    x_proj = x_proj < 0 || x_proj >= W ? 0 : roundf(x_proj);
	y_proj = y_proj < 0 || y_proj >= H ? 0 : roundf(y_proj);

    int x_p = (int)x_proj;
    int y_p = (int)y_proj;

    float cost = 0.0f;
    float count = 0.0f;

    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {
            int tile_x = threadIdx.x + dx + half;
            int tile_y = threadIdx.y + dy + half;
            int xc = x_p + dx, yc = y_p + dy;
            if (tile_x >= 0 && tile_x < tile_w && tile_y >= 0 && tile_y < tile_h &&
                xc >= 0 && xc < W && yc >= 0 && yc < H) {
                int idx_ref = tile_y * tile_w + tile_x;
                int idx_cam = yc * W + xc;
                float tgt_val = tex2D<uint8_t>(texTgt, xc + 0.5f, yc + 0.5f);
                cost += fabsf((float)ref_tile[idx_ref] - tgt_val);
                count += 1.0f;
            }
        }
    }

    if (count > 0.0f) cost /= count;

    int idx = z * H * W + y * W + x;
    cost_vol[idx] = fminf(cost_vol[idx], cost);
}


// ============================================================================
// ========================== WRAPPER FUNCTIONS =========================
// ============================================================================



// Wrapper function for naive implementation (double precision and float)
// This function launches the naive CUDA kernel for plane sweeping stereo.
// It allocates device memory, copies input data, launches the kernel for each camera,
// and copies the result back to host memory.
void sweeping_plane_gpu_device_Naive(
    const uint8_t* ref_Y, const CamParams& ref_params,
    const std::vector<const uint8_t*>& cam_Ys,
    const std::vector<CamParams>& cam_params,
    float* h_cost_vol,
    int W, int H, int window
) {
    clock_t function_start = clock();

    size_t img_size = W * H;

    // Set up CUDA grid and block dimensions
    dim3 threads(16, 16, 1);
    dim3 blocks(
        (W + threads.x - 1) / threads.x,
        (H + threads.y - 1) / threads.y,
        (ZPlanes + threads.z - 1) / threads.z
    );

    uint8_t* d_ref;
    float* d_cost;

    cudaError_t cudaStatus;
    cudaEvent_t start_gpu, stop_gpu; // cudaEvent are used to time the kernel

    std::vector<float> execution_times(cam_Ys.size());

    // Create CUDA events for timing
    CHK(cudaEventCreate(&start_gpu));
    CHK(cudaEventCreate(&stop_gpu));

    // Allocate and copy reference image to device
    CHK(cudaMalloc(&d_ref, img_size));
    CHK(cudaMemcpy(d_ref, ref_Y, img_size, cudaMemcpyHostToDevice));

    // Allocate and copy initial cost volume to device
    CHK(cudaMalloc(&d_cost, ZPlanes * img_size * sizeof(float)));
    CHK(cudaMemcpy(d_cost, h_cost_vol, ZPlanes * img_size * sizeof(float), cudaMemcpyHostToDevice));
    
    uint8_t* d_cam;
    CHK(cudaMalloc(&d_cam, img_size));

    // Copy reference camera parameters to constant memory
    cudaMemcpyToSymbol(ref, &ref_params, sizeof(CamParams));

    // Loop over all target cameras
    for (size_t i = 0; i < cam_Ys.size(); ++i) {

        // Copy target camera parameters to constant memory
        cudaMemcpyToSymbol(cam, &cam_params[i], sizeof(CamParams));

        // Copy target image to device
        CHK(cudaMemcpy(d_cam, cam_Ys[i], img_size, cudaMemcpyHostToDevice));

        // Start timing
        CHK(cudaEventRecord(start_gpu, 0));

        // Launch the naive float kernel (can switch to double if needed)
        // sweep_kernel_double_Naive<<<blocks, threads>>>(
        //     d_cost, d_ref, d_cam, W, H, window
        // );
        sweep_kernel_float_Naive<<<blocks, threads>>>(
            d_cost, d_ref, d_cam, W, H, window
        );

        // Stop timing
        CHK(cudaEventRecord(stop_gpu, 0));
        CHK(cudaGetLastError());
        CHK(cudaDeviceSynchronize());
        CHK(cudaEventSynchronize(stop_gpu));
        float milliseconds = 0.0f;
        CHK(cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu));
        execution_times[i] = milliseconds / 1000.0f;
    }

    // Free target camera buffer
    CHK(cudaFree(d_cam));

    // Print execution times for each camera
    for (size_t i = 0; i < cam_Ys.size(); ++i) {
        std::cout << "Execution time for camera " << i << " kernel : " << execution_times[i] << " seconds" << std::endl;
    }

    // Copy result cost volume back to host
    CHK(cudaMemcpy(h_cost_vol, d_cost, ZPlanes * img_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHK(cudaFree(d_ref));
    CHK(cudaFree(d_cost));
    CHK(cudaEventDestroy(start_gpu));
    CHK(cudaEventDestroy(stop_gpu));

    return;

Error:
    // Cleanup in case of error
    cudaFree(d_ref);
    cudaFree(d_cost);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    std::cerr << "CUDA error occurred in sweeping_plane_gpu_device." << std::endl;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }

    return;
}



// Wrapper function for shared memory kernel using only reference image in shared memory
void sweeping_plane_gpu_device_Shared_REF(
    const uint8_t* ref_Y, const CamParams& ref_params,
    const std::vector<const uint8_t*>& cam_Ys,
    const std::vector<CamParams>& cam_params,
    float* h_cost_vol,
    int W, int H, int window
) {
    clock_t function_start = clock();

    size_t img_size = W * H;

    // Set up CUDA grid and block dimensions
    dim3 threads(16, 16, 1);
    dim3 blocks(
        (W + threads.x - 1) / threads.x,
        (H + threads.y - 1) / threads.y,
        (ZPlanes + threads.z - 1) / threads.z
    );

    uint8_t* d_ref;
    float* d_cost;

    cudaError_t cudaStatus;
    cudaEvent_t start_gpu, stop_gpu; // CUDA events for timing

    std::vector<float> execution_times(cam_Ys.size());

    // Shared memory size for reference image tile
    size_t shared_mem_size = (threads.x + window - 1) * (threads.y + window - 1) * sizeof(uint8_t);

    CHK(cudaEventCreate(&start_gpu));
    CHK(cudaEventCreate(&stop_gpu));

    // Allocate and copy reference image to device
    CHK(cudaMalloc(&d_ref, img_size));
    CHK(cudaMemcpy(d_ref, ref_Y, img_size, cudaMemcpyHostToDevice));

    // Allocate and copy initial cost volume to device
    CHK(cudaMalloc(&d_cost, ZPlanes * img_size * sizeof(float)));
    CHK(cudaMemcpy(d_cost, h_cost_vol, ZPlanes * img_size * sizeof(float), cudaMemcpyHostToDevice));
    
    uint8_t* d_cam;
    CHK(cudaMalloc(&d_cam, img_size));

    // Copy reference camera parameters to constant memory
    cudaMemcpyToSymbol(ref, &ref_params, sizeof(CamParams));

    // Loop over all target cameras
    for (size_t i = 0; i < cam_Ys.size(); ++i) {

        // Copy target camera parameters to constant memory
        cudaMemcpyToSymbol(cam, &cam_params[i], sizeof(CamParams));

        // Copy target image to device
        CHK(cudaMemcpy(d_cam, cam_Ys[i], img_size, cudaMemcpyHostToDevice));
        clock_t kernel_start = clock();
        std::cout << "time kernel start: " << (double)(kernel_start - function_start)/CLOCKS_PER_SEC << std::endl;
        
        // Start timing
        CHK(cudaEventRecord(start_gpu, 0));
        // Launch the shared memory kernel (reference image only)
        sweep_kernel_float_shared_REF<<<blocks, threads, shared_mem_size>>>(
            d_cost, d_ref, d_cam, W, H, window
        );
        CHK(cudaEventRecord(stop_gpu, 0));

        CHK(cudaGetLastError());
        CHK(cudaDeviceSynchronize());

        // Stop timing and record execution time
        CHK(cudaEventSynchronize(stop_gpu));
        float milliseconds = 0.0f;
        CHK(cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu));
        execution_times[i] = milliseconds / 1000.0f;
    }
    CHK(cudaFree(d_cam));
    // Print execution times for each camera
    for (size_t i = 0; i < cam_Ys.size(); ++i) {
        std::cout << "Execution time for camera " << i << " kernel : " << execution_times[i] << " seconds" << std::endl;
    }

    // Copy result cost volume back to host
    CHK(cudaMemcpy(h_cost_vol, d_cost, ZPlanes * img_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHK(cudaFree(d_ref));
    CHK(cudaFree(d_cost));
    CHK(cudaEventDestroy(start_gpu));
    CHK(cudaEventDestroy(stop_gpu));

    return;

Error:
    // Cleanup in case of error
    cudaFree(d_ref);
    cudaFree(d_cost);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    std::cerr << "CUDA error occurred in sweeping_plane_gpu_device." << std::endl;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }

    return;
}





// Wrapper function for shared memory kernel using both reference and target images in shared memory
void sweeping_plane_gpu_device_Shared_REF_TGT(
    const uint8_t* ref_Y, const CamParams& ref_params,
    const std::vector<const uint8_t*>& cam_Ys,
    const std::vector<CamParams>& cam_params,
    float* h_cost_vol,
    int W, int H, int window
) {
    clock_t function_start = clock();

    size_t img_size = W * H;

    // Use 16x16 block for good occupancy and shared memory usage
    dim3 threads(16, 16, 1);
    dim3 blocks(
        (W + threads.x - 1) / threads.x,
        (H + threads.y - 1) / threads.y,
        (ZPlanes + threads.z - 1) / threads.z
    );

    uint8_t* d_ref = nullptr;
    float* d_cost = nullptr;
    uint8_t* d_cam = nullptr;
    cudaEvent_t start_gpu = nullptr, stop_gpu = nullptr;
    cudaError_t cudaStatus;

    std::vector<float> execution_times(cam_Ys.size());

    // Calculate shared memory size for both reference and target tiles
    const int radius = window / 2;
    const int shared_width = threads.x + 2 * radius;
    const int shared_height = threads.y + 2 * radius;
    const size_t shared_mem_size = 2 * shared_width * shared_height * sizeof(uint8_t);

    // Create CUDA events for timing
    CHK(cudaEventCreate(&start_gpu));
    CHK(cudaEventCreate(&stop_gpu));

    // Allocate and copy reference image to device
    CHK(cudaMalloc(&d_ref, img_size));
    CHK(cudaMemcpy(d_ref, ref_Y, img_size, cudaMemcpyHostToDevice));

    // Allocate and copy initial cost volume to device
    CHK(cudaMalloc(&d_cost, ZPlanes * img_size * sizeof(float)));
    CHK(cudaMemcpy(d_cost, h_cost_vol, ZPlanes * img_size * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate device memory for target image
    CHK(cudaMalloc(&d_cam, img_size));

    // Copy reference camera parameters to constant memory
    CHK(cudaMemcpyToSymbol(ref, &ref_params, sizeof(CamParams)));

    // Loop over all target cameras
    for (size_t i = 0; i < cam_Ys.size(); ++i) {
        // Copy target camera parameters to constant memory
        CHK(cudaMemcpyToSymbol(cam, &cam_params[i], sizeof(CamParams)));
        // Copy target image to device
        CHK(cudaMemcpy(d_cam, cam_Ys[i], img_size, cudaMemcpyHostToDevice));

        // Start timing
        CHK(cudaEventRecord(start_gpu, 0));
        // Launch the shared memory kernel (reference and target images)
        sweep_kernel_float_shared_REF_TGT<<<blocks, threads, shared_mem_size>>>(
            d_cost, d_ref, d_cam, W, H, window
        );
        CHK(cudaEventRecord(stop_gpu, 0));
        CHK(cudaGetLastError());
        CHK(cudaEventSynchronize(stop_gpu));

        float milliseconds = 0.0f;
        CHK(cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu));
        execution_times[i] = milliseconds / 1000.0f;
    }

    // Copy result cost volume back to host
    CHK(cudaMemcpy(h_cost_vol, d_cost, ZPlanes * img_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Print execution times for each camera
    for (size_t i = 0; i < cam_Ys.size(); ++i) {
        std::cout << "Execution time for camera " << i << " kernel : " << execution_times[i] << " seconds" << std::endl;
    }

    // Cleanup
    cudaFree(d_cam);
    cudaFree(d_ref);
    cudaFree(d_cost);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    return;

Error:
    // Cleanup in case of error
    cudaFree(d_cam);
    cudaFree(d_ref);
    cudaFree(d_cost);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    std::cerr << "CUDA error occurred in sweeping_plane_gpu_device_Shared_REF_TGT." << std::endl;
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
    return;
}




// Wrapper function for texture memory kernel
// This function launches the CUDA kernel that uses texture memory for the target image.
// It sets up the CUDA texture object, allocates/copies device memory, launches the kernel for each camera,
// and copies the result back to host memory.
void sweeping_plane_gpu_device_texture(
    const uint8_t* ref_Y, const CamParams& ref_params,
    const std::vector<const uint8_t*>& cam_Ys,
    const std::vector<CamParams>& cam_params,
    float* h_cost_vol,
    int W, int H, int window
) {
    clock_t function_start = clock();

    size_t img_size = W * H;

    // Set up CUDA grid and block dimensions
    dim3 threads(16, 16, 1);
    dim3 blocks(
        (W + threads.x - 1) / threads.x,
        (H + threads.y - 1) / threads.y,
        (ZPlanes + threads.z - 1) / threads.z
    );

    uint8_t* d_ref;
    float* d_cost;

    cudaError_t cudaStatus;
    cudaEvent_t start_gpu, stop_gpu; // CUDA events for timing

    std::vector<float> execution_times(cam_Ys.size());

    // Allocate CUDA array for target image (for texture memory)
    cudaArray_t tgt_array;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8_t>();
    CHK(cudaMallocArray(&tgt_array, &channelDesc, W, H));

    // Set up resource descriptor for texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = tgt_array;

    // Set up texture descriptor
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = false;

    // Create texture object for target image
    cudaTextureObject_t texTgt = 0;
    CHK(cudaCreateTextureObject(&texTgt, &resDesc, &texDesc, NULL));
    
    // Create CUDA events for timing
    CHK(cudaEventCreate(&start_gpu));
    CHK(cudaEventCreate(&stop_gpu));

    // Allocate and copy reference image to device
    CHK(cudaMalloc(&d_ref, img_size));
    CHK(cudaMemcpy(d_ref, ref_Y, img_size, cudaMemcpyHostToDevice));

    // Allocate and copy initial cost volume to device
    CHK(cudaMalloc(&d_cost, ZPlanes * img_size * sizeof(float)));
    CHK(cudaMemcpy(d_cost, h_cost_vol, ZPlanes * img_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Copy reference camera parameters to constant memory
    cudaMemcpyToSymbol(ref, &ref_params, sizeof(CamParams));

    // Loop over all target cameras
    for (size_t i = 0; i < cam_Ys.size(); ++i) {

        // Copy target camera parameters to constant memory
        cudaMemcpyToSymbol(cam, &cam_params[i], sizeof(CamParams));

        // Copy target image to CUDA array for texture access
        CHK(cudaMemcpyToArray(tgt_array, 0, 0, cam_Ys[i], img_size * sizeof(uint8_t), cudaMemcpyHostToDevice));

        // Start timing
        CHK(cudaEventRecord(start_gpu, 0));
        // Launch the kernel using texture memory for the target image
        sweep_kernel_float_texture<<<blocks, threads, (threads.x + window - 1) * (threads.y + window - 1) * sizeof(uint8_t)>>>(
            d_cost, d_ref, texTgt, W, H, window
        );
        CHK(cudaEventRecord(stop_gpu, 0));

        CHK(cudaGetLastError());
        CHK(cudaDeviceSynchronize());

        // Stop timing and record execution time
        CHK(cudaEventSynchronize(stop_gpu));
        float milliseconds = 0.0f;
        CHK(cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu));
        execution_times[i] = milliseconds / 1000.0f;
    }
    // Print execution times for each camera
    for (size_t i = 0; i < cam_Ys.size(); ++i) {
        std::cout << "Execution time for camera " << i << " kernel : " << execution_times[i] << " seconds" << std::endl;
    }

    // Copy result cost volume back to host
    CHK(cudaMemcpy(h_cost_vol, d_cost, ZPlanes * img_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CHK(cudaDestroyTextureObject(texTgt));
    CHK(cudaFree(d_ref));
    CHK(cudaFree(d_cost));
    CHK(cudaEventDestroy(start_gpu));
    CHK(cudaEventDestroy(stop_gpu));

    return;

Error:
    // Cleanup in case of error
    if (texTgt) {
        cudaDestroyTextureObject(texTgt);
    }
    cudaFree(d_ref);
    cudaFree(d_cost);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    std::cerr << "CUDA error occurred in sweeping_plane_gpu_device." << std::endl;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }

    return;
}






