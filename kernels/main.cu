#include "main.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cuda.h>




#define ZPlanes 256
#define ZNear 0.3f
#define ZFar 1.1f


__device__ float fatomicMin(float* addr, float value)

{
    float old = *addr, assumed;
    if (old <= value) return old;

    do
    {
        assumed = old;

        old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));

    } while (old != assumed);

        return old;
}


// Those functions are an example on how to call cuda functions from the main.cpp




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

    if (x_proj < 0 || x_proj >= W || y_proj < 0 || y_proj >= H) return;

    int x_p = round(x_proj);
    int y_p = round(y_proj);

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


        dim3 threads(32, 16, 1);
        dim3 blocks(
            (W + threads.x - 1) / threads.x,
            (H + threads.y - 1) / threads.y,
            (ZPlanes + threads.z - 1) / threads.z
        );

        sweep_kernel_double_Naive << <blocks, threads >> > (
            d_cost, d_ref, d_cam, W, H, ref_params, cam_params[i], window
            );
        cudaDeviceSynchronize();
        cudaFree(d_cam);
    }

    cudaMemcpy(h_cost_vol, d_cost, ZPlanes * img_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_ref);
    cudaFree(d_cost);
}






