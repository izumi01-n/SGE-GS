#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace FORWARD 
{
// Evaluate view-dependent color from spherical-harmonic coefficients.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
    glm::vec3 pos = means[idx];
    glm::vec3 dir = pos - campos;
    dir = dir / glm::length(dir);

    glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
    glm::vec3 result = SH_C0 * sh[0];

    if (deg > 0)
    {
        float x = dir.x; float y = dir.y; float z = dir.z;
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

        if (deg > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result +
                SH_C2[0] * xy * sh[4] +
                SH_C2[1] * yz * sh[5] +
                SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
                SH_C2[3] * xz * sh[7] +
                SH_C2[4] * (xx - yy) * sh[8];

            if (deg > 2)
            {
                result = result +
                    SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                    SH_C3[1] * xy * z * sh[10] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                    SH_C3[5] * z * (xx - yy) * sh[14] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }
    result += 0.5f;
    clamped[3 * idx + 0] = (result.x < 0);
    clamped[3 * idx + 1] = (result.y < 0);
    clamped[3 * idx + 2] = (result.z < 0);
    return glm::max(result, 0.0f);
}

// Project the unfiltered covariance for geometry-related rasterization.
__device__ float3 computeCov2DBase(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
    float3 t = transformPoint4x3(mean, viewmatrix);
    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    glm::mat3 J = glm::mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0);

    glm::mat3 W = glm::mat3(
        viewmatrix[0], viewmatrix[4], viewmatrix[8],
        viewmatrix[1], viewmatrix[5], viewmatrix[9],
        viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    glm::mat3 T = W * J;
    glm::mat3 Vrk = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]);

    glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;
    
    // Return the symmetric 2D covariance in compact form.
    return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Construct the 3D covariance from Gaussian scale and rotation.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
    glm::mat3 S = glm::mat3(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

    glm::vec4 q = rot;
    float r = q.x; float x = q.y; float y = q.z; float z = q.w;
    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    glm::mat3 M = S * R;
    glm::mat3 Sigma = glm::transpose(M) * M;
    cov3D[0] = Sigma[0][0]; cov3D[1] = Sigma[0][1]; cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1]; cov3D[4] = Sigma[1][2]; cov3D[5] = Sigma[2][2];
}

template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
    const float* orig_points, const glm::vec3* scales, const float scale_modifier, const glm::vec4* rotations,
    const float* opacities, const float* shs, bool* clamped, const float* cov3D_precomp, const float* colors_precomp,
    const float* viewmatrix, const float* projmatrix, const glm::vec3* cam_pos,
    const int W, int H, const float tan_fovx, float tan_fovy, const float focal_x, float focal_y,
    const float kernel_size, int* radii, float2* points_xy_image, float* depths, float* cov3Ds, float* colors, 
    float4* conic_opacity,float4* conic_opacity_raw, const dim3 grid, uint32_t* tiles_touched, bool prefiltered,
    float* normals)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    radii[idx] = 0;
    tiles_touched[idx] = 0;

    float3 p_view;
    if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view)) return;

    float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
    float4 p_hom = transformPoint4x4(p_orig, projmatrix);
    float p_w = 1.0f / (p_hom.w + 0.0000001f);
    float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

    // Estimate a geometric normal from the Gaussian’s shortest principal axis, independent of the Mip filter.
    if (normals != nullptr) {
        glm::vec3 scale = scales[idx];
        glm::vec4 q = rotations[idx];
        float r = q.x, x = q.y, y = q.z, z = q.w;

        glm::mat3 R = glm::mat3(
            1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
            2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
            2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
        );

        int min_idx = 0;
        if (scale.y < scale.x) {
            min_idx = 1;
            if (scale.z < scale.y) min_idx = 2;
        } else {
            if (scale.z < scale.x) min_idx = 2;
        }
        glm::vec3 local_normal = glm::vec3(R[0][min_idx], R[1][min_idx], R[2][min_idx]);

        glm::mat3 W_mat = glm::mat3(
            viewmatrix[0], viewmatrix[4], viewmatrix[8],
            viewmatrix[1], viewmatrix[5], viewmatrix[9],
            viewmatrix[2], viewmatrix[6], viewmatrix[10]
        );
        glm::vec3 cam_normal = glm::normalize(W_mat * local_normal);

        normals[idx * 3 + 0] = cam_normal.x;
        normals[idx * 3 + 1] = cam_normal.y;
        normals[idx * 3 + 2] = cam_normal.z;
    }
    // -------------------------->v<------------------------------

    const float* cov3D;
    if (cov3D_precomp != nullptr) {
        cov3D = cov3D_precomp + idx * 6;
    } else {
        computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
        cov3D = cov3Ds + idx * 6;
    }

    // Keep filtered and unfiltered conics separate: appearance uses Mip filtering, while geometry retains the original footprint.
    float3 cov2D_raw = computeCov2DBase(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
    float det_raw = cov2D_raw.x * cov2D_raw.z - cov2D_raw.y * cov2D_raw.y;
    if (det_raw == 0.0f) return;

    float3 cov2D_filt = { cov2D_raw.x + kernel_size, cov2D_raw.y, cov2D_raw.z + kernel_size };
    float det_filt = cov2D_filt.x * cov2D_filt.z - cov2D_filt.y * cov2D_filt.y;

    float coef = sqrt(max(1e-6f, det_raw) / max(1e-6f, det_filt) + 1e-6f);

    float det_inv_raw = 1.f / det_raw;
    float3 conic_r = { cov2D_raw.z * det_inv_raw, -cov2D_raw.y * det_inv_raw, cov2D_raw.x * det_inv_raw };

    float det_inv_filt = 1.f / det_filt;
    float3 conic_f = { cov2D_filt.z * det_inv_filt, -cov2D_filt.y * det_inv_filt, cov2D_filt.x * det_inv_filt };

    float mid = 0.5f * (cov2D_filt.x + cov2D_filt.z);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det_filt));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det_filt));
    float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
    
    float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
    uint2 rect_min, rect_max;
    getRect(point_image, my_radius, rect_min, rect_max, grid);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) return;

    if (colors_precomp == nullptr) {
        glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
        colors[idx * C + 0] = result.x;
        colors[idx * C + 1] = result.y;
        colors[idx * C + 2] = result.z;
    }

    depths[idx] = p_view.z;
    radii[idx] = my_radius;
    points_xy_image[idx] = point_image;

    conic_opacity[idx] = { conic_f.x, conic_f.y, conic_f.z, opacities[idx] * coef };
    if (conic_opacity_raw != nullptr) {
        conic_opacity_raw[idx] = { conic_r.x, conic_r.y, conic_r.z, opacities[idx] };
    }
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float2* __restrict__ subpixel_offset,
    const float2* __restrict__ points_xy_image,
    const float* __restrict__ features,

    const float4* __restrict__ conic_opacity,     // filtered (RGB)
    const float4* __restrict__ conic_opacity_raw, // raw (geometry)

    const float* __restrict__ depths,
    const float* __restrict__ normals,

    float* __restrict__ final_T,
    uint32_t* __restrict__ n_contrib,
    const float* __restrict__ bg_color,

    float* __restrict__ out_color,
    float* __restrict__ out_depth,
    float* __restrict__ out_alpha,
    float* __restrict__ out_normal)
{
    auto block = cg::this_thread_block();

    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = {
        block.group_index().x * BLOCK_X,
        block.group_index().y * BLOCK_Y
    };

    uint2 pix = {
        pix_min.x + block.thread_index().x,
        pix_min.y + block.thread_index().y
    };

    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = { (float)pix.x, (float)pix.y };

    bool inside = pix.x < W && pix.y < H;
    bool done = !inside;

    if (inside) {
        pixf.x += subpixel_offset[pix_id].x;
        pixf.y += subpixel_offset[pix_id].y;
    }

    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_f[BLOCK_SIZE];
    __shared__ float4 collected_conic_r[BLOCK_SIZE];

    float T = 1.0f;  // Shared transmittance controls visibility for both output branches.

    uint32_t last_contributor = 0;

    float C[CHANNELS] = { 0 };
    float N[3] = { 0 };

    float depth_acc = 0.0f;
    float weight_acc_geo = 0.0f;
    float weight_acc_rgb = 0.0f;

    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE) break;

        int progress = i * BLOCK_SIZE + block.thread_rank();

        if (range.x + progress < range.y) {
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];
            collected_conic_f[block.thread_rank()] = conic_opacity[coll_id];
            collected_conic_r[block.thread_rank()] = conic_opacity_raw[coll_id];
        }

        block.sync();

        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
        {
            float2 xy = collected_xy[j];
            float2 d = { xy.x - pixf.x, xy.y - pixf.y };

            float4 con_f = collected_conic_f[j];
            float4 con_r = collected_conic_r[j];

            float power_f = -0.5f * (con_f.x * d.x * d.x + con_f.z * d.y * d.y) - con_f.y * d.x * d.y;
            float power_r = -0.5f * (con_r.x * d.x * d.x + con_r.z * d.y * d.y) - con_r.y * d.x * d.y;

            if (power_f > 0.0f && power_r > 0.0f) continue;

            float alpha_f = (power_f > 0.0f) ? 0.0f : min(0.99f, con_f.w * exp(power_f));
            float alpha_r = (power_r > 0.0f) ? 0.0f : min(0.99f, con_r.w * exp(power_r));

            if (alpha_f < 1.0f / 255.0f && alpha_r < 1.0f / 255.0f) continue;

            // Use filtered opacity for shared visibility, while accumulating depth and normals with the unfiltered footprint.
            float alpha_vis = alpha_f;
            float test_T = T * (1.0f - alpha_vis);

            if (test_T < 0.0001f) {
                done = true;
                continue;
            }

            float weight_rgb = alpha_f * T;
            float weight_geo = alpha_r * T;

            for (int ch = 0; ch < CHANNELS; ch++) {
                C[ch] += features[collected_id[j] * CHANNELS + ch] * weight_rgb;
            }
            weight_acc_rgb += weight_rgb;

            float z = depths[collected_id[j]];
            depth_acc += z * weight_geo;
            weight_acc_geo += weight_geo;

            if (normals != nullptr && out_normal != nullptr) {
                N[0] += normals[collected_id[j] * 3 + 0] * weight_geo;
                N[1] += normals[collected_id[j] * 3 + 1] * weight_geo;
                N[2] += normals[collected_id[j] * 3 + 2] * weight_geo;
            }

            T = test_T;
            last_contributor = i * BLOCK_SIZE + j + 1;
        }
    }

    if (inside) {
        final_T[pix_id] = T;
        n_contrib[pix_id] = last_contributor;

        out_alpha[pix_id] = weight_acc_geo; 

        out_depth[pix_id] = (weight_acc_geo > 0.0f) ? (depth_acc / weight_acc_geo) : 0.0f;

        for (int ch = 0; ch < CHANNELS; ch++)
            out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

        if (out_normal != nullptr && weight_acc_geo > 1e-6f) {
            float norm = sqrtf(N[0]*N[0] + N[1]*N[1] + N[2]*N[2] + 1e-8f);
            out_normal[0 * H * W + pix_id] = N[0] / norm;
            out_normal[1 * H * W + pix_id] = N[1] / norm;
            out_normal[2 * H * W + pix_id] = N[2] / norm;
        }
    }
}

// [Original excerpt placeholder: renderCUDA and preprocessCUDA definitions omitted.]
void render(
    const dim3 grid, dim3 block, 
    const uint2* ranges, 
    const uint32_t* point_list, 
    int W, int H, 
    const float2* subpixel_offset, 
    const float2* means2D,
    const float* colors, 
    const float4* conic_opacity, 
    const float4* conic_opacity_raw, 
    const float* depths, 
    const float* normals, 
    float* final_T, 
    uint32_t* n_contrib, 
    const float* bg_color, 
    float* out_color, 
    float* out_depth, 
    float* out_alpha, 
    float* out_normal)
{
    renderCUDA<NUM_CHANNELS> << <grid, block >> > (
        ranges, point_list, W, H, subpixel_offset, 
        means2D,
        colors, conic_opacity, conic_opacity_raw, depths, normals, 
        final_T, n_contrib, bg_color, out_color, out_depth, out_alpha, out_normal
    );
}

void preprocess(
    int P, int D, int M, 
    const float* means3D,
    const glm::vec3* scales, 
    const float scale_modifier, 
    const glm::vec4* rotations, 
    const float* opacities, 
    const float* shs, 
    bool* clamped, 
    const float* cov3D_precomp, 
    const float* colors_precomp, 
    const float* viewmatrix, 
    const float* projmatrix, 
    const glm::vec3* cam_pos, 
    const int W, int H, 
    const float tan_fovx, float tan_fovy, 
    const float focal_x, float focal_y, 
    const float kernel_size, 
    int* radii, 
    float2* means2D,
    float* depths, 
    float* cov3Ds, 
    float* colors, 
    float4* conic_opacity, 
    float4* conic_opacity_raw, 
    const dim3 grid, 
    uint32_t* tiles_touched, 
    bool prefiltered, 
    float* normals)
{
    preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
        P, D, M, 
        means3D,
        scales, scale_modifier, rotations, opacities, shs, clamped, 
        cov3D_precomp, colors_precomp, viewmatrix, projmatrix, cam_pos, 
        W, H, tan_fovx, tan_fovy, focal_x, focal_y, kernel_size, radii, 
        means2D,
        depths, cov3Ds, colors, conic_opacity, conic_opacity_raw, grid, 
        tiles_touched,
        prefiltered, normals
    );
}

} // namespace FORWARD