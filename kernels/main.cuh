#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>


// minimal GPU-side struct for camera intrinsics/extrinsics:
struct CamParams {
    float K[9], R[9], t[3];
    float K_inv[9], R_inv[9], t_inv[3];
};

__global__ void sweep_kernel_double_Naive(
    float* cost_vol, const uint8_t* ref_img, const uint8_t* tgt_img,
    int W, int H, int window);

__global__ void sweep_kernel_float_Naive(
    float* cost_vol, const uint8_t* ref_img, const uint8_t* tgt_img,
    int W, int H, int window);

__global__ void sweep_kernel_float_shared(
    float* cost_vol, const uint8_t* ref_img, const uint8_t* tgt_img,
    int W, int H, int window);

__global__ void sweep_kernel_float_texture(
    float* cost_vol, const uint8_t* ref_img, cudaTextureObject_t tgt_img,
    int W, int H, int window);

void sweeping_plane_gpu_device_Naive(
    const uint8_t* ref_Y, const CamParams& ref_params,
    const std::vector<const uint8_t*>& cam_Ys,
    const std::vector<CamParams>& cam_params,
    float* h_cost_vol,
    int W, int H, int window
);

void sweeping_plane_gpu_device_Shared(
    const uint8_t* ref_Y, const CamParams& ref_params,
    const std::vector<const uint8_t*>& cam_Ys,
    const std::vector<CamParams>& cam_params,
    float* h_cost_vol,
    int W, int H, int window
);

void sweeping_plane_gpu_device_texture(
    const uint8_t* ref_Y, const CamParams& ref_params,
    const std::vector<const uint8_t*>& cam_Ys,
    const std::vector<CamParams>& cam_params,
    float* h_cost_vol,
    int W, int H, int window
);


