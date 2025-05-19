#include "main.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cuda.h>


#define ZPlanes 256
#define ZNear 0.3f
#define ZFar 1.1f




__global__ void sweep_kernel_double_Naive(
    float* cost_vol,             // [ZPlanes * H * W]
    const uint8_t* ref_img,      // [H * W]
    const uint8_t* tgt_img,
    int W, int H,
    CamParams ref, CamParams cam,
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

    if (x_proj < 0 || x_proj >= W) x_proj = 0;
    if (y_proj < 0 || y_proj >= H) y_proj = 0;

    int x_p = roundf(x_proj);
    int y_p = roundf(y_proj);

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
    //cost_vol[idx] = atomicMinFloat(&cost_vol[idx], cost);  // Atomic min
}


void sweeping_plane_gpu_device(
    const uint8_t* ref_Y, const CamParams& ref_params,
    const std::vector<const uint8_t*>& cam_Ys,
    const std::vector<CamParams>& cam_params,
    float* h_cost_vol,
    int W, int H, int window
) {
    size_t img_size = W * H;

    uint8_t* d_ref;
    float* d_cost;
    cudaMalloc(&d_ref, img_size);
    cudaMemcpy(d_ref, ref_Y, img_size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_cost, ZPlanes * img_size * sizeof(float));
    cudaMemcpy(d_cost, h_cost_vol, ZPlanes * img_size * sizeof(float), cudaMemcpyHostToDevice);

    for (size_t i = 0; i < cam_Ys.size(); ++i) {
        uint8_t* d_cam;
        cudaMalloc(&d_cam, img_size);
        cudaMemcpy(d_cam, cam_Ys[i], img_size, cudaMemcpyHostToDevice);


        dim3 threads(8, 8, 4);
        dim3 blocks(
            (W + threads.x - 1) / threads.x,
            (H + threads.y - 1) / threads.y,
            (ZPlanes + threads.z - 1) / threads.z
        );

        clock_t start = clock();
        sweep_kernel_double_Naive << <blocks, threads >> > (
            d_cost, d_ref, d_cam, W, H, ref_params, cam_params[i], window
            );
        cudaDeviceSynchronize();
        clock_t end = clock();
        double elapsed = double(end - start) / CLOCKS_PER_SEC;
        std::cout << "Kernel execution time: " << elapsed * 1000 << " ms" << std::endl;
        cudaFree(d_cam);
    }

    cudaMemcpy(h_cost_vol, d_cost, ZPlanes * img_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_ref);
    cudaFree(d_cost);
}






