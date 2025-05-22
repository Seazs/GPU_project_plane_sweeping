#include "main.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>


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


__global__ void sweep_kernel_float_shared(
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
                cost += fabsf((float)ref_tile[idx_ref] - (float)tgt_img[idx_cam]);
                count += 1.0f;
            }
        }
    }

    if (count > 0.0f) cost /= count;

    int idx = z * H * W + y * W + x;
    cost_vol[idx] = fminf(cost_vol[idx], cost);
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



void sweeping_plane_gpu_device_Shared(
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

    size_t shared_mem_size = (threads.x + window - 1) * (threads.y + window - 1) * sizeof(uint8_t); //

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
        sweep_kernel_float_shared<<<blocks, threads, shared_mem_size>>>(
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






