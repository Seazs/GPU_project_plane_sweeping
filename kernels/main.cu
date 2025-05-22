#include "main.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>


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





// Load reference image tile into shared memory
__device__ void load_shared_ref(
    uint8_t* shared_ref, const uint8_t* ref_img,
    int x, int y, int tx, int ty, int lx, int ly,
    int W, int H, int pad, int shared_w, int shared_h)
{
    // Center pixel
    shared_ref[INDEX_2D(ly, lx, shared_w)] = ref_img[INDEX_2D(y, x, W)];

    // Left & right borders
    if (tx < pad) {
        int x_left = x - pad;
        int x_right = x + blockDim.x;
        shared_ref[INDEX_2D(ly, tx, shared_w)] =
            (x_left >= 0) ? ref_img[INDEX_2D(y, x_left, W)] : 0;
        shared_ref[INDEX_2D(ly, tx + blockDim.x + pad, shared_w)] =
            (x_right < W) ? ref_img[INDEX_2D(y, x_right, W)] : 0;
    }

    // Top & bottom borders
    if (ty < pad) {
        int y_top = y - pad;
        int y_bot = y + blockDim.y;
        shared_ref[INDEX_2D(ty, lx, shared_w)] =
            (y_top >= 0) ? ref_img[INDEX_2D(y_top, x, W)] : 0;
        shared_ref[INDEX_2D(ty + blockDim.y + pad, lx, shared_w)] =
            (y_bot < H) ? ref_img[INDEX_2D(y_bot, x, W)] : 0;
    }

    // Corners
    if (tx < pad && ty < pad) {
        int x_left = x - pad;
        int x_right = x + blockDim.x;
        int y_top = y - pad;
        int y_bot = y + blockDim.y;
        shared_ref[INDEX_2D(ty, tx, shared_w)] =
            (x_left >= 0 && y_top >= 0) ? ref_img[INDEX_2D(y_top, x_left, W)] : 0;
        shared_ref[INDEX_2D(ty, tx + blockDim.x + pad, shared_w)] =
            (x_right < W && y_top >= 0) ? ref_img[INDEX_2D(y_top, x_right, W)] : 0;
        shared_ref[INDEX_2D(ty + blockDim.y + pad, tx, shared_w)] =
            (x_left >= 0 && y_bot < H) ? ref_img[INDEX_2D(y_bot, x_left, W)] : 0;
        shared_ref[INDEX_2D(ty + blockDim.y + pad, tx + blockDim.x + pad, shared_w)] =
            (x_right < W && y_bot < H) ? ref_img[INDEX_2D(y_bot, x_right, W)] : 0;
    }
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

// Compute SAD cost
__device__ float compute_cost(
    const uint8_t* shared_ref, const uint8_t* tgt_img,
    int lx, int ly, int x_p, int y_p,
    int pad, int shared_w, int W, int H)
{
    float cost = 0.0f;
    float count = 0.0f;
    for (int dy = -pad; dy <= pad; dy++) {
        for (int dx = -pad; dx <= pad; dx++) {
            int rx = lx + dx;
            int ry = ly + dy;
            int px = x_p + dx;
            int py = y_p + dy;
            if (px >= 0 && px < W && py >= 0 && py < H) {
                float ref_val = (float)shared_ref[INDEX_2D(ry, rx, shared_w)];
                float tgt_val = (float)tgt_img[INDEX_2D(py, px, W)];
                cost += fabsf(ref_val - tgt_val);
                count += 1.0f;
            }
        }
    }
    return (count > 0.0f) ? cost / count : 255.0f;
}

// Sweep kernel using shared memory for reference image only
__global__ void sweep_kernel_float_shared(
    float* cost_vol,
    const uint8_t* ref_img,
    const uint8_t* tgt_img,
    int W, int H,
    int window)
{
    extern __shared__ uint8_t shared_ref[];

    const int pad = window / 2;
    const int shared_w = blockDim.x + 2 * pad;
    const int shared_h = blockDim.y + 2 * pad;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;
    const int z = blockIdx.z;

    if (x >= W || y >= H || z >= ZPlanes) return;

    const int lx = tx + pad;
    const int ly = ty + pad;

    // Load shared memory
    load_shared_ref(shared_ref, ref_img, x, y, tx, ty, lx, ly, W, H, pad, shared_w, shared_h);

    __syncthreads();

    // Projection
    float x_proj, y_proj;
    compute_projection(x, y, z, W, H, x_proj, y_proj);

    if (x_proj < 0 || x_proj >= W || y_proj < 0 || y_proj >= H) return;

    int x_p = (int)roundf(x_proj);
    int y_p = (int)roundf(y_proj);

    // Cost computation
    float cost = compute_cost(shared_ref, tgt_img, lx, ly, x_p, y_p, pad, shared_w, W, H);

    int idx = INDEX_3D(z, y, x, H, W);
    cost_vol[idx] = fminf(cost_vol[idx], cost);
}


// Optimized shared memory kernel with improved memory access patterns
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

    // Compute projection using the existing function (unchanged as requested)
    float proj_x, proj_y;
    compute_projection(gid_x, gid_y, depth_idx, W, H, proj_x, proj_y);

    // Bounds checking - match CPU behavior exactly
    if (proj_x < 0.0f || proj_x >= W || proj_y < 0.0f || proj_y >= H) {
        return;  // Skip this pixel entirely like CPU version
    }

    const int center_x = __float2int_rn(proj_x);  // Using intrinsic for rounding
    const int center_y = __float2int_rn(proj_y);

    // Cost computation - match CPU logic exactly
    float total_cost = 0.0f;
    float valid_pixels = 0.0f;  // Use float to match CPU behavior
    
    // Match CPU nested loop order exactly
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            // Reference pixel coordinates (using shared memory)
            int ref_x_local = local_x + dx;
            int ref_y_local = local_y + dy;
            
            // Target pixel coordinates (global)
            int tgt_x_global = center_x + dx;
            int tgt_y_global = center_y + dy;
            
            // Bounds checking - match CPU continue logic
            if (tgt_x_global < 0 || tgt_x_global >= W || 
                tgt_y_global < 0 || tgt_y_global >= H) {
                continue;
            }
            
            const int ref_idx = ref_y_local * tile_width + ref_x_local;
            const int tgt_idx = tgt_y_global * W + tgt_x_global;
            
            // Use regular float conversion instead of intrinsics for consistency
            const float ref_val = (float)shared_mem[ref_idx];
            const float tgt_val = (float)tgt_img[tgt_idx];
            
            total_cost += fabsf(ref_val - tgt_val);
            valid_pixels += 1.0f;
        }
    }

    // Match CPU division behavior exactly
    float final_cost = (valid_pixels > 0.0f) ? (total_cost / valid_pixels) : 255.0f;

    // Update cost volume with atomic min for safety (optional: can be removed if single-camera)
    const int output_idx = INDEX_3D(depth_idx, gid_y, gid_x, H, W);
    cost_vol[output_idx] = fminf(cost_vol[output_idx], final_cost);
}




__global__ void sweep_kernel_float_shared_REF_TGT(
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

    const int radius = window >> 1;
    const int tile_width = blockDim.x + (radius << 1);
    const int tile_height = blockDim.y + (radius << 1);
    
    // For target image, we use a larger tile to account for projection spread
    // This is a heuristic - you might need to adjust based on your camera setup
    const int tgt_expand = max(radius * 2, 32); // Expand target tile more than reference
    const int tgt_tile_width = blockDim.x + (tgt_expand << 1);
    const int tgt_tile_height = blockDim.y + (tgt_expand << 1);
    
    // Memory layout: [reference_tile][target_tile]
    uint8_t* ref_shared = shared_mem;
    uint8_t* tgt_shared = shared_mem + tile_width * tile_height;
    
    const int total_threads = blockDim.x * blockDim.y;
    const int thread_id = tid_y * blockDim.x + tid_x;
    
    // Load reference image tile (same as before)
    const int ref_elements = tile_width * tile_height;
    for (int elem_id = thread_id; elem_id < ref_elements; elem_id += total_threads) {
        int tile_y = elem_id / tile_width;
        int tile_x = elem_id % tile_width;
        
        int src_x = blockIdx.x * blockDim.x + tile_x - radius;
        int src_y = blockIdx.y * blockDim.y + tile_y - radius;
        
        src_x = max(0, min(src_x, W - 1));
        src_y = max(0, min(src_y, H - 1));
        
        ref_shared[elem_id] = ref_img[src_y * W + src_x];
    }
    
    // Load target image tile (larger area to account for projections)
    const int tgt_elements = tgt_tile_width * tgt_tile_height;
    for (int elem_id = thread_id; elem_id < tgt_elements; elem_id += total_threads) {
        int tile_y = elem_id / tgt_tile_width;
        int tile_x = elem_id % tgt_tile_width;
        
        int src_x = blockIdx.x * blockDim.x + tile_x - tgt_expand;
        int src_y = blockIdx.y * blockDim.y + tile_y - tgt_expand;
        
        src_x = max(0, min(src_x, W - 1));
        src_y = max(0, min(src_y, H - 1));
        
        tgt_shared[elem_id] = tgt_img[src_y * W + src_x];
    }

    __syncthreads();

    // Compute projection
    float proj_x, proj_y;
    compute_projection(gid_x, gid_y, depth_idx, W, H, proj_x, proj_y);

    if (proj_x < 0.0f || proj_x >= W || proj_y < 0.0f || proj_y >= H) {
        return;
    }

    const int center_x = __float2int_rn(proj_x);
    const int center_y = __float2int_rn(proj_y);
    
    // Calculate base coordinates for target tile access
    const int tgt_base_x = blockIdx.x * blockDim.x - tgt_expand;
    const int tgt_base_y = blockIdx.y * blockDim.y - tgt_expand;

    // Cost computation using both shared memory tiles
    float total_cost = 0.0f;
    float valid_pixels = 0.0f;
    
    const int local_x = tid_x + radius;
    const int local_y = tid_y + radius;
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            // Reference pixel (from shared memory)
            int ref_x_local = local_x + dx;
            int ref_y_local = local_y + dy;
            
            // Target pixel coordinates
            int tgt_x_global = center_x + dx;
            int tgt_y_global = center_y + dy;
            
            // Bounds checking for global coordinates
            if (tgt_x_global < 0 || tgt_x_global >= W || 
                tgt_y_global < 0 || tgt_y_global >= H) {
                continue;
            }
            
            // Convert global target coordinates to local tile coordinates
            int tgt_x_local = tgt_x_global - tgt_base_x;
            int tgt_y_local = tgt_y_global - tgt_base_y;
            
            // Check if target pixel is within our cached tile
            bool use_shared_tgt = (tgt_x_local >= 0 && tgt_x_local < tgt_tile_width &&
                                  tgt_y_local >= 0 && tgt_y_local < tgt_tile_height);
            
            const int ref_idx = ref_y_local * tile_width + ref_x_local;
            const float ref_val = (float)ref_shared[ref_idx];
            
            float tgt_val;
            if (use_shared_tgt) {
                // Use shared memory for target
                const int tgt_idx = tgt_y_local * tgt_tile_width + tgt_x_local;
                tgt_val = (float)tgt_shared[tgt_idx];
            } else {
                // Fall back to global memory for target (should be rare)
                const int tgt_idx = tgt_y_global * W + tgt_x_global;
                tgt_val = (float)tgt_img[tgt_idx];
            }
            
            total_cost += fabsf(ref_val - tgt_val);
            valid_pixels += 1.0f;
        }
    }

    float final_cost = (valid_pixels > 0.0f) ? (total_cost / valid_pixels) : 255.0f;

    const int output_idx = INDEX_3D(depth_idx, gid_y, gid_x, H, W);
    cost_vol[output_idx] = fminf(cost_vol[output_idx], final_cost);
}


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


void sweeping_plane_gpu_device_Naive(
    const uint8_t* ref_Y, const CamParams& ref_params,
    const std::vector<const uint8_t*>& cam_Ys,
    const std::vector<CamParams>& cam_params,
    float* h_cost_vol,
    int W, int H, int window
) {
    clock_t function_start = clock();

    size_t img_size = W * H;

    dim3 threads(16, 16, 1);
    dim3 blocks(
        (W + threads.x - 1) / threads.x,
        (H + threads.y - 1) / threads.y,
        (ZPlanes + threads.z - 1) / threads.z
    );

    uint8_t* d_ref;
    float* d_cost;

    cudaError_t cudaStatus;
    cudaEvent_t start_gpu, stop_gpu; //cudaEvent are used to time the kernel

    std::vector<float> execution_times(cam_Ys.size());

    CHK(cudaEventCreate(&start_gpu));
    CHK(cudaEventCreate(&stop_gpu));

    CHK(cudaMalloc(&d_ref, img_size));
    CHK(cudaMemcpy(d_ref, ref_Y, img_size, cudaMemcpyHostToDevice));

    CHK(cudaMalloc(&d_cost, ZPlanes * img_size * sizeof(float)));
    CHK(cudaMemcpy(d_cost, h_cost_vol, ZPlanes * img_size * sizeof(float), cudaMemcpyHostToDevice));
    
    uint8_t* d_cam;
    CHK(cudaMalloc(&d_cam, img_size));

    cudaMemcpyToSymbol(ref, &ref_params, sizeof(CamParams));


    for (size_t i = 0; i < cam_Ys.size(); ++i) {

        cudaMemcpyToSymbol(cam, &cam_params[i], sizeof(CamParams));

        CHK(cudaMemcpy(d_cam, cam_Ys[i], img_size, cudaMemcpyHostToDevice));
        clock_t kernel_start = clock();
        std::cout << "time kernel start: " << (double)(kernel_start - function_start)/CLOCKS_PER_SEC << std::endl;
        
        CHK(cudaEventRecord(start_gpu, 0));
        sweep_kernel_float_Naive<<<blocks, threads>>>(
            d_cost, d_ref, d_cam, W, H, window
        );

        CHK(cudaEventRecord(stop_gpu, 0));
        CHK(cudaGetLastError());
        CHK(cudaDeviceSynchronize());
        CHK(cudaEventSynchronize(stop_gpu));
        float milliseconds = 0.0f;
        CHK(cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu));
        execution_times[i] = milliseconds / 1000.0f;

        
    }
    CHK(cudaFree(d_cam));
    // Print execution times
    for (size_t i = 0; i < cam_Ys.size(); ++i) {
        std::cout << "Execution time for camera " << i << " kernel : " << execution_times[i] << " seconds" << std::endl;
    }

    CHK(cudaMemcpy(h_cost_vol, d_cost, ZPlanes * img_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHK(cudaFree(d_ref));
    CHK(cudaFree(d_cost));
    CHK(cudaEventDestroy(start_gpu));
    CHK(cudaEventDestroy(stop_gpu));

    

    return;

Error:
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



void sweeping_plane_gpu_device_Shared_REF(
    const uint8_t* ref_Y, const CamParams& ref_params,
    const std::vector<const uint8_t*>& cam_Ys,
    const std::vector<CamParams>& cam_params,
    float* h_cost_vol,
    int W, int H, int window
) {
    clock_t function_start = clock();

    size_t img_size = W * H;

    dim3 threads(16, 16, 1);
    dim3 blocks(
        (W + threads.x - 1) / threads.x,
        (H + threads.y - 1) / threads.y,
        (ZPlanes + threads.z - 1) / threads.z
    );

    uint8_t* d_ref;
    float* d_cost;

    cudaError_t cudaStatus;
    cudaEvent_t start_gpu, stop_gpu; //cudaEvent are used to time the kernel

    std::vector<float> execution_times(cam_Ys.size());

    size_t shared_mem_size = (threads.x + window - 1) * (threads.y + window - 1) * sizeof(uint8_t); //for ref and tgt

    CHK(cudaEventCreate(&start_gpu));
    CHK(cudaEventCreate(&stop_gpu));

    CHK(cudaMalloc(&d_ref, img_size));
    CHK(cudaMemcpy(d_ref, ref_Y, img_size, cudaMemcpyHostToDevice));

    CHK(cudaMalloc(&d_cost, ZPlanes * img_size * sizeof(float)));
    CHK(cudaMemcpy(d_cost, h_cost_vol, ZPlanes * img_size * sizeof(float), cudaMemcpyHostToDevice));
    
    uint8_t* d_cam;
    CHK(cudaMalloc(&d_cam, img_size));

    cudaMemcpyToSymbol(ref, &ref_params, sizeof(CamParams));


    for (size_t i = 0; i < cam_Ys.size(); ++i) {

		cudaMemcpyToSymbol(cam, &cam_params[i], sizeof(CamParams));

        CHK(cudaMemcpy(d_cam, cam_Ys[i], img_size, cudaMemcpyHostToDevice));
        clock_t kernel_start = clock();
        std::cout << "time kernel start: " << (double)(kernel_start - function_start)/CLOCKS_PER_SEC << std::endl;
        
        CHK(cudaEventRecord(start_gpu, 0));
        sweep_kernel_float_shared_REF<<<blocks, threads, shared_mem_size>>>(
            d_cost, d_ref, d_cam, W, H, window
        );
        CHK(cudaEventRecord(stop_gpu, 0));


        CHK(cudaGetLastError());
        CHK(cudaDeviceSynchronize());


        CHK(cudaEventSynchronize(stop_gpu));
        float milliseconds = 0.0f;
        CHK(cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu));
        execution_times[i] = milliseconds / 1000.0f;

        
    }
    CHK(cudaFree(d_cam));
    // Print execution times
    for (size_t i = 0; i < cam_Ys.size(); ++i) {
        std::cout << "Execution time for camera " << i << " kernel : " << execution_times[i] << " seconds" << std::endl;
    }

    CHK(cudaMemcpy(h_cost_vol, d_cost, ZPlanes * img_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHK(cudaFree(d_ref));
    CHK(cudaFree(d_cost));
    CHK(cudaEventDestroy(start_gpu));
    CHK(cudaEventDestroy(stop_gpu));

    

    return;

Error:
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





void sweeping_plane_gpu_device_Shared_REF_TGT(
    const uint8_t* ref_Y, const CamParams& ref_params,
    const std::vector<const uint8_t*>& cam_Ys,
    const std::vector<CamParams>& cam_params,
    float* h_cost_vol,
    int W, int H, int window
) {
    clock_t function_start = clock();

    size_t img_size = W * H;

    dim3 threads(16, 16, 1);
    dim3 blocks(
        (W + threads.x - 1) / threads.x,
        (H + threads.y - 1) / threads.y,
        (ZPlanes + threads.z - 1) / threads.z
    );

    uint8_t* d_ref;
    float* d_cost;

    cudaError_t cudaStatus;
    cudaEvent_t start_gpu, stop_gpu;

    std::vector<float> execution_times(cam_Ys.size());

    // Calculate shared memory size for the optimized version
    const int radius = window / 2;
    const int ref_tile_width = threads.x + 2 * radius;
    const int ref_tile_height = threads.y + 2 * radius;
    const int tgt_expand = max(radius * 2, 32);
    const int tgt_tile_width = threads.x + 2 * tgt_expand;
    const int tgt_tile_height = threads.y + 2 * tgt_expand;
    
    size_t shared_mem_size = (ref_tile_width * ref_tile_height + 
                             tgt_tile_width * tgt_tile_height) * sizeof(uint8_t);
    
    // For two-pass version, we need additional space for projection coordinates
    // size_t shared_mem_size_twopass = ref_tile_width * ref_tile_height * sizeof(uint8_t) +
    //                                  threads.x * threads.y * 2 * sizeof(float) +
    //                                  estimated_cache_size * sizeof(uint8_t);

    CHK(cudaEventCreate(&start_gpu));
    CHK(cudaEventCreate(&stop_gpu));

    CHK(cudaMalloc(&d_ref, img_size));
    CHK(cudaMemcpy(d_ref, ref_Y, img_size, cudaMemcpyHostToDevice));

    CHK(cudaMalloc(&d_cost, ZPlanes * img_size * sizeof(float)));
    CHK(cudaMemcpy(d_cost, h_cost_vol, ZPlanes * img_size * sizeof(float), cudaMemcpyHostToDevice));
    
    uint8_t* d_cam;
    CHK(cudaMalloc(&d_cam, img_size));

    cudaMemcpyToSymbol(ref, &ref_params, sizeof(CamParams));

    for (size_t i = 0; i < cam_Ys.size(); ++i) {
        cudaMemcpyToSymbol(cam, &cam_params[i], sizeof(CamParams));

        CHK(cudaMemcpy(d_cam, cam_Ys[i], img_size, cudaMemcpyHostToDevice));
        clock_t kernel_start = clock();
        std::cout << "time kernel start: " << (double)(kernel_start - function_start)/CLOCKS_PER_SEC << std::endl;
        
        CHK(cudaEventRecord(start_gpu, 0));
        
        // Use the optimized version with both ref and target shared memory
        sweep_kernel_float_shared_REF_TGT<<<blocks, threads, shared_mem_size>>>(
            d_cost, d_ref, d_cam, W, H, window
        );
        
        // Alternative: Use two-pass version (comment out the above and uncomment below)
        // sweep_kernel_float_shared_REF_TGT_twopass<<<blocks, threads, shared_mem_size_twopass>>>(
        //     d_cost, d_ref, d_cam, W, H, window
        // );
        
        CHK(cudaEventRecord(stop_gpu, 0));
        CHK(cudaGetLastError());
        CHK(cudaDeviceSynchronize());
        CHK(cudaEventSynchronize(stop_gpu));
        
        float milliseconds = 0.0f;
        CHK(cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu));
        execution_times[i] = milliseconds / 1000.0f;
    }
    
    CHK(cudaFree(d_cam));
    
    // Print execution times
    for (size_t i = 0; i < cam_Ys.size(); ++i) {
        std::cout << "Execution time for camera " << i << " kernel : " << execution_times[i] << " seconds" << std::endl;
    }

    CHK(cudaMemcpy(h_cost_vol, d_cost, ZPlanes * img_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHK(cudaFree(d_ref));
    CHK(cudaFree(d_cost));
    CHK(cudaEventDestroy(start_gpu));
    CHK(cudaEventDestroy(stop_gpu));

    return;

Error:
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




void sweeping_plane_gpu_device_texture(
    const uint8_t* ref_Y, const CamParams& ref_params,
    const std::vector<const uint8_t*>& cam_Ys,
    const std::vector<CamParams>& cam_params,
    float* h_cost_vol,
    int W, int H, int window
) {
    clock_t function_start = clock();

    size_t img_size = W * H;

    dim3 threads(16, 16, 1);
    dim3 blocks(
        (W + threads.x - 1) / threads.x,
        (H + threads.y - 1) / threads.y,
        (ZPlanes + threads.z - 1) / threads.z
    );

    uint8_t* d_ref;
    float* d_cost;

    cudaError_t cudaStatus;
    cudaEvent_t start_gpu, stop_gpu; //cudaEvent are used to time the kernel

    std::vector<float> execution_times(cam_Ys.size());

    cudaArray_t tgt_array;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8_t>();
    CHK(cudaMallocArray(&tgt_array, &channelDesc, W, H));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = tgt_array;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t texTgt = 0;
    CHK(cudaCreateTextureObject(&texTgt, &resDesc, &texDesc, NULL));
    


    CHK(cudaEventCreate(&start_gpu));
    CHK(cudaEventCreate(&stop_gpu));

    CHK(cudaMalloc(&d_ref, img_size));
    CHK(cudaMemcpy(d_ref, ref_Y, img_size, cudaMemcpyHostToDevice));

    CHK(cudaMalloc(&d_cost, ZPlanes * img_size * sizeof(float)));
    CHK(cudaMemcpy(d_cost, h_cost_vol, ZPlanes * img_size * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaMemcpyToSymbol(ref, &ref_params, sizeof(CamParams));


    for (size_t i = 0; i < cam_Ys.size(); ++i) {

		cudaMemcpyToSymbol(cam, &cam_params[i], sizeof(CamParams));

        CHK(cudaMemcpyToArray(tgt_array, 0, 0, cam_Ys[i], img_size * sizeof(uint8_t), cudaMemcpyHostToDevice));

        
        CHK(cudaEventRecord(start_gpu, 0));
        sweep_kernel_float_texture<<<blocks, threads, (threads.x + window - 1) * (threads.y + window - 1) * sizeof(uint8_t)>>>(
            d_cost, d_ref, texTgt, W, H, window
        );
        CHK(cudaEventRecord(stop_gpu, 0));


        CHK(cudaGetLastError());
        CHK(cudaDeviceSynchronize());


        CHK(cudaEventSynchronize(stop_gpu));
        float milliseconds = 0.0f;
        CHK(cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu));
        execution_times[i] = milliseconds / 1000.0f;

        
    }
    // Print execution times
    for (size_t i = 0; i < cam_Ys.size(); ++i) {
        std::cout << "Execution time for camera " << i << " kernel : " << execution_times[i] << " seconds" << std::endl;
    }

    CHK(cudaMemcpy(h_cost_vol, d_cost, ZPlanes * img_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CHK(cudaDestroyTextureObject(texTgt));
    CHK(cudaFree(d_ref));
    CHK(cudaFree(d_cost));
    CHK(cudaEventDestroy(start_gpu));
    CHK(cudaEventDestroy(stop_gpu));

    

    return;

Error:
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






