// GeoBrain G6 stage-E(3D): native CUDA forward + adjoint for the 3-D elastic
// velocity-stress equation (isotropic/VTI-form coefficients c11,c12,c13,c33,
// c44,c66,buoyancy). No free surface (the eager 3-D equation has none either).
// Same verified recipe: V-then-S order, zero-fill D+/D- stencils, K-M CPML,
// whole loops in C++, adjoint via transpose identities + stored strain
// histories (ex,ey,ez,gxy,gxz,gyz) + velocity histories for gB.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace {

constexpr int MAX_H = 8;

#define IDX3(b, z, y, x) ((((long)(b) * nz + (z)) * ny + (y)) * nx + (x))
#define MIDX3(z, y, x) (((long)(z) * ny + (y)) * nx + (x))

// D+ / D- reads with zero fill, per axis (macro-free helpers via lambdas are
// awkward in device code; write the taps inline per use).

template <typename scalar_t>
__global__ void el3_velocity_kernel(
    scalar_t* __restrict__ vx, scalar_t* __restrict__ vy, scalar_t* __restrict__ vz,
    const scalar_t* __restrict__ sxx, const scalar_t* __restrict__ syy,
    const scalar_t* __restrict__ szz, const scalar_t* __restrict__ sxy,
    const scalar_t* __restrict__ sxz, const scalar_t* __restrict__ syz,
    scalar_t* __restrict__ p_sxx_x, scalar_t* __restrict__ p_sxy_y,
    scalar_t* __restrict__ p_sxz_z, scalar_t* __restrict__ p_sxy_x,
    scalar_t* __restrict__ p_syy_y, scalar_t* __restrict__ p_syz_z,
    scalar_t* __restrict__ p_sxz_x, scalar_t* __restrict__ p_syz_y,
    scalar_t* __restrict__ p_szz_z,
    const scalar_t* __restrict__ buoy_m, const scalar_t* __restrict__ fd,
    const scalar_t* __restrict__ bx, const scalar_t* __restrict__ ax, const scalar_t* __restrict__ kx,
    const scalar_t* __restrict__ by, const scalar_t* __restrict__ ay, const scalar_t* __restrict__ ky,
    const scalar_t* __restrict__ bz, const scalar_t* __restrict__ az, const scalar_t* __restrict__ kz,
    const scalar_t* __restrict__ bxh, const scalar_t* __restrict__ axh, const scalar_t* __restrict__ kxh,
    const scalar_t* __restrict__ byh, const scalar_t* __restrict__ ayh, const scalar_t* __restrict__ kyh,
    const scalar_t* __restrict__ bzh, const scalar_t* __restrict__ azh, const scalar_t* __restrict__ kzh,
    const int H, const int nz, const int ny, const int nx, const int ztiles,
    const scalar_t dt, const scalar_t ix_, const scalar_t iy_, const scalar_t iz_)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int zt = blockIdx.z % ztiles, b = blockIdx.z / ztiles;
    const int z = zt * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const long i = IDX3(b, z, y, x);
    const long m = MIDX3(z, y, x);

    scalar_t a1 = 0, a2 = 0, a3 = 0;   // vx: D+x sxx, D-y sxy, D-z sxz
    scalar_t b1 = 0, b2 = 0, b3 = 0;   // vy: D-x sxy, D+y syy, D-z syz
    scalar_t d1 = 0, d2 = 0, d3 = 0;   // vz: D-x sxz, D-y syz, D+z szz
    for (int k = 1; k <= H; ++k) {
        const scalar_t c = fd[k-1];
        int p, q;
        p = x + k; q = x - k + 1;      // D+x
        a1 += c * (((p < nx) ? sxx[IDX3(b,z,y,p)] : scalar_t(0))
                 - ((q >= 0) ? sxx[IDX3(b,z,y,q)] : scalar_t(0)));
        p = y + k - 1; q = y - k;      // D-y
        a2 += c * (((p < ny) ? sxy[IDX3(b,z,p,x)] : scalar_t(0))
                 - ((q >= 0) ? sxy[IDX3(b,z,q,x)] : scalar_t(0)));
        p = z + k - 1; q = z - k;      // D-z
        a3 += c * (((p < nz) ? sxz[IDX3(b,p,y,x)] : scalar_t(0))
                 - ((q >= 0) ? sxz[IDX3(b,q,y,x)] : scalar_t(0)));
        p = x + k - 1; q = x - k;      // D-x
        b1 += c * (((p < nx) ? sxy[IDX3(b,z,y,p)] : scalar_t(0))
                 - ((q >= 0) ? sxy[IDX3(b,z,y,q)] : scalar_t(0)));
        p = y + k; q = y - k + 1;      // D+y
        b2 += c * (((p < ny) ? syy[IDX3(b,z,p,x)] : scalar_t(0))
                 - ((q >= 0) ? syy[IDX3(b,z,q,x)] : scalar_t(0)));
        p = z + k - 1; q = z - k;      // D-z
        b3 += c * (((p < nz) ? syz[IDX3(b,p,y,x)] : scalar_t(0))
                 - ((q >= 0) ? syz[IDX3(b,q,y,x)] : scalar_t(0)));
        p = x + k - 1; q = x - k;      // D-x
        d1 += c * (((p < nx) ? sxz[IDX3(b,z,y,p)] : scalar_t(0))
                 - ((q >= 0) ? sxz[IDX3(b,z,y,q)] : scalar_t(0)));
        p = y + k - 1; q = y - k;      // D-y
        d2 += c * (((p < ny) ? syz[IDX3(b,z,p,x)] : scalar_t(0))
                 - ((q >= 0) ? syz[IDX3(b,z,q,x)] : scalar_t(0)));
        p = z + k; q = z - k + 1;      // D+z
        d3 += c * (((p < nz) ? szz[IDX3(b,p,y,x)] : scalar_t(0))
                 - ((q >= 0) ? szz[IDX3(b,q,y,x)] : scalar_t(0)));
    }
    a1 *= ix_; a2 *= iy_; a3 *= iz_;
    b1 *= ix_; b2 *= iy_; b3 *= iz_;
    d1 *= ix_; d2 *= iy_; d3 *= iz_;

    scalar_t P;
    scalar_t accx = 0, accy = 0, accz = 0;
    P = bxh[x] * p_sxx_x[i] + axh[x] * a1; p_sxx_x[i] = P; accx = a1 / kxh[x] + P;
    P = by[y] * p_sxy_y[i] + ay[y] * a2;  p_sxy_y[i] = P; accy = a2 / ky[y] + P;
    P = bz[z] * p_sxz_z[i] + az[z] * a3;  p_sxz_z[i] = P; accz = a3 / kz[z] + P;
    vx[i] += dt * buoy_m[m] * (accx + accy + accz);

    P = bx[x] * p_sxy_x[i] + ax[x] * b1;  p_sxy_x[i] = P; accx = b1 / kx[x] + P;
    P = byh[y] * p_syy_y[i] + ayh[y] * b2; p_syy_y[i] = P; accy = b2 / kyh[y] + P;
    P = bz[z] * p_syz_z[i] + az[z] * b3;  p_syz_z[i] = P; accz = b3 / kz[z] + P;
    vy[i] += dt * buoy_m[m] * (accx + accy + accz);

    P = bx[x] * p_sxz_x[i] + ax[x] * d1;  p_sxz_x[i] = P; accx = d1 / kx[x] + P;
    P = by[y] * p_syz_y[i] + ay[y] * d2;  p_syz_y[i] = P; accy = d2 / ky[y] + P;
    P = bzh[z] * p_szz_z[i] + azh[z] * d3; p_szz_z[i] = P; accz = d3 / kzh[z] + P;
    vz[i] += dt * buoy_m[m] * (accx + accy + accz);
}

template <typename scalar_t>
__global__ void el3_stress_kernel(
    const scalar_t* __restrict__ vx, const scalar_t* __restrict__ vy,
    const scalar_t* __restrict__ vz,
    scalar_t* __restrict__ sxx, scalar_t* __restrict__ syy,
    scalar_t* __restrict__ szz, scalar_t* __restrict__ sxy,
    scalar_t* __restrict__ sxz, scalar_t* __restrict__ syz,
    scalar_t* __restrict__ p_vx_x, scalar_t* __restrict__ p_vy_y,
    scalar_t* __restrict__ p_vz_z, scalar_t* __restrict__ p_vx_y,
    scalar_t* __restrict__ p_vy_x, scalar_t* __restrict__ p_vx_z,
    scalar_t* __restrict__ p_vz_x, scalar_t* __restrict__ p_vy_z,
    scalar_t* __restrict__ p_vz_y,
    const scalar_t* __restrict__ c11, const scalar_t* __restrict__ c12,
    const scalar_t* __restrict__ c13, const scalar_t* __restrict__ c33,
    const scalar_t* __restrict__ c44, const scalar_t* __restrict__ c66,
    const scalar_t* __restrict__ fd,
    const scalar_t* __restrict__ bx, const scalar_t* __restrict__ ax, const scalar_t* __restrict__ kx,
    const scalar_t* __restrict__ by, const scalar_t* __restrict__ ay, const scalar_t* __restrict__ ky,
    const scalar_t* __restrict__ bz, const scalar_t* __restrict__ az, const scalar_t* __restrict__ kz,
    const scalar_t* __restrict__ bxh, const scalar_t* __restrict__ axh, const scalar_t* __restrict__ kxh,
    const scalar_t* __restrict__ byh, const scalar_t* __restrict__ ayh, const scalar_t* __restrict__ kyh,
    const scalar_t* __restrict__ bzh, const scalar_t* __restrict__ azh, const scalar_t* __restrict__ kzh,
    scalar_t* __restrict__ ex_h, scalar_t* __restrict__ ey_h,
    scalar_t* __restrict__ ez_h, scalar_t* __restrict__ gxy_h,
    scalar_t* __restrict__ gxz_h, scalar_t* __restrict__ gyz_h,
    const int it, const int B,
    const int H, const int nz, const int ny, const int nx, const int ztiles,
    const scalar_t dt, const scalar_t ix_, const scalar_t iy_, const scalar_t iz_)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int zt = blockIdx.z % ztiles, b = blockIdx.z / ztiles;
    const int z = zt * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const long i = IDX3(b, z, y, x);
    const long m = MIDX3(z, y, x);

    scalar_t dex = 0, dey = 0, dez = 0;      // D-x vx, D-y vy, D-z vz
    scalar_t g1 = 0, g2 = 0;                 // D+y vx, D+x vy
    scalar_t h1 = 0, h2 = 0;                 // D+z vx, D+x vz
    scalar_t i1 = 0, i2 = 0;                 // D+z vy, D+y vz
    for (int k = 1; k <= H; ++k) {
        const scalar_t c = fd[k-1];
        int p, q;
        p = x + k - 1; q = x - k;
        dex += c * (((p < nx) ? vx[IDX3(b,z,y,p)] : scalar_t(0))
                  - ((q >= 0) ? vx[IDX3(b,z,y,q)] : scalar_t(0)));
        p = y + k - 1; q = y - k;
        dey += c * (((p < ny) ? vy[IDX3(b,z,p,x)] : scalar_t(0))
                  - ((q >= 0) ? vy[IDX3(b,z,q,x)] : scalar_t(0)));
        p = z + k - 1; q = z - k;
        dez += c * (((p < nz) ? vz[IDX3(b,p,y,x)] : scalar_t(0))
                  - ((q >= 0) ? vz[IDX3(b,q,y,x)] : scalar_t(0)));
        p = y + k; q = y - k + 1;
        g1 += c * (((p < ny) ? vx[IDX3(b,z,p,x)] : scalar_t(0))
                 - ((q >= 0) ? vx[IDX3(b,z,q,x)] : scalar_t(0)));
        p = x + k; q = x - k + 1;
        g2 += c * (((p < nx) ? vy[IDX3(b,z,y,p)] : scalar_t(0))
                 - ((q >= 0) ? vy[IDX3(b,z,y,q)] : scalar_t(0)));
        p = z + k; q = z - k + 1;
        h1 += c * (((p < nz) ? vx[IDX3(b,p,y,x)] : scalar_t(0))
                 - ((q >= 0) ? vx[IDX3(b,q,y,x)] : scalar_t(0)));
        p = x + k; q = x - k + 1;
        h2 += c * (((p < nx) ? vz[IDX3(b,z,y,p)] : scalar_t(0))
                 - ((q >= 0) ? vz[IDX3(b,z,y,q)] : scalar_t(0)));
        p = z + k; q = z - k + 1;
        i1 += c * (((p < nz) ? vy[IDX3(b,p,y,x)] : scalar_t(0))
                 - ((q >= 0) ? vy[IDX3(b,q,y,x)] : scalar_t(0)));
        p = y + k; q = y - k + 1;
        i2 += c * (((p < ny) ? vz[IDX3(b,z,p,x)] : scalar_t(0))
                 - ((q >= 0) ? vz[IDX3(b,z,q,x)] : scalar_t(0)));
    }
    dex *= ix_; dey *= iy_; dez *= iz_;
    g1 *= iy_; g2 *= ix_; h1 *= iz_; h2 *= ix_; i1 *= iz_; i2 *= iy_;

    scalar_t P;
    P = bx[x] * p_vx_x[i] + ax[x] * dex; p_vx_x[i] = P; const scalar_t ex = dex / kx[x] + P;
    P = by[y] * p_vy_y[i] + ay[y] * dey; p_vy_y[i] = P; const scalar_t ey = dey / ky[y] + P;
    P = bz[z] * p_vz_z[i] + az[z] * dez; p_vz_z[i] = P; const scalar_t ez = dez / kz[z] + P;
    sxx[i] += dt * (c11[m] * ex + c12[m] * ey + c13[m] * ez);
    syy[i] += dt * (c12[m] * ex + c11[m] * ey + c13[m] * ez);
    szz[i] += dt * (c13[m] * ex + c13[m] * ey + c33[m] * ez);

    P = byh[y] * p_vx_y[i] + ayh[y] * g1; p_vx_y[i] = P; const scalar_t G1 = g1 / kyh[y] + P;
    P = bxh[x] * p_vy_x[i] + axh[x] * g2; p_vy_x[i] = P; const scalar_t G2 = g2 / kxh[x] + P;
    sxy[i] += dt * c66[m] * (G1 + G2);

    P = bzh[z] * p_vx_z[i] + azh[z] * h1; p_vx_z[i] = P; const scalar_t H1 = h1 / kzh[z] + P;
    P = bxh[x] * p_vz_x[i] + axh[x] * h2; p_vz_x[i] = P; const scalar_t H2 = h2 / kxh[x] + P;
    sxz[i] += dt * c44[m] * (H1 + H2);

    P = bzh[z] * p_vy_z[i] + azh[z] * i1; p_vy_z[i] = P; const scalar_t I1 = i1 / kzh[z] + P;
    P = byh[y] * p_vz_y[i] + ayh[y] * i2; p_vz_y[i] = P; const scalar_t I2 = i2 / kyh[y] + P;
    syz[i] += dt * c44[m] * (I1 + I2);

    if (ex_h != nullptr) {
        const long hidx = (long)it * B * ((long)nz * ny * nx) + i;
        ex_h[hidx] = ex; ey_h[hidx] = ey; ez_h[hidx] = ez;
        gxy_h[hidx] = G1 + G2; gxz_h[hidx] = H1 + H2; gyz_h[hidx] = I1 + I2;
    }
}

template <typename scalar_t>
__global__ void el3_source_kernel(
    scalar_t* __restrict__ sxx, scalar_t* __restrict__ syy,
    scalar_t* __restrict__ szz,
    const int64_t* __restrict__ sz, const int64_t* __restrict__ sy,
    const int64_t* __restrict__ sx,
    const scalar_t* __restrict__ wav,
    const int it, const int nt, const int nz, const int ny, const int nx,
    const scalar_t dt, const int n_src)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_src) return;
    const long i = IDX3(b, sz[b], sy[b], sx[b]);
    const scalar_t v = dt * wav[(long)b * nt + it];
    sxx[i] += v; syy[i] += v; szz[i] += v;
}

template <typename scalar_t>
__global__ void el3_record_kernel(
    const scalar_t* __restrict__ sxx, const scalar_t* __restrict__ syy,
    const scalar_t* __restrict__ szz,
    const int64_t* __restrict__ rz, const int64_t* __restrict__ ry,
    const int64_t* __restrict__ rx,
    scalar_t* __restrict__ out,
    const int it, const int nt, const int n_rcv,
    const int nz, const int ny, const int nx)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (r >= n_rcv) return;
    const long i = IDX3(b, rz[r], ry[r], rx[r]);
    out[((long)b * nt + it) * n_rcv + r] =
        -(sxx[i] + syy[i] + szz[i]) / scalar_t(3);
}

template <typename scalar_t>
__global__ void el3_store_hist_kernel(
    const scalar_t* __restrict__ vx, const scalar_t* __restrict__ vy,
    const scalar_t* __restrict__ vz,
    scalar_t* __restrict__ vxh, scalar_t* __restrict__ vyh,
    scalar_t* __restrict__ vzh,
    const int it, const long ncell)
{
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= ncell) return;
    const long o = (long)(it + 1) * ncell + i;
    vxh[o] = vx[i]; vyh[o] = vy[i]; vzh[o] = vz[i];
}

// ------------------------------ adjoint ------------------------------------

template <typename scalar_t>
__global__ void el3_adj_record_scatter_kernel(
    scalar_t* __restrict__ sxxbar, scalar_t* __restrict__ syybar,
    scalar_t* __restrict__ szzbar,
    const scalar_t* __restrict__ gout,
    const int64_t* __restrict__ rz, const int64_t* __restrict__ ry,
    const int64_t* __restrict__ rx,
    const int it, const int nt, const int n_rcv,
    const int nz, const int ny, const int nx)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (r >= n_rcv) return;
    const long i = IDX3(b, rz[r], ry[r], rx[r]);
    const scalar_t g = -gout[((long)b * nt + it) * n_rcv + r] / scalar_t(3);
    atomicAdd(&sxxbar[i], g);
    atomicAdd(&syybar[i], g);
    atomicAdd(&szzbar[i], g);
}

template <typename scalar_t>
__global__ void el3_adj_source_kernel(
    const scalar_t* __restrict__ sxxbar, const scalar_t* __restrict__ syybar,
    const scalar_t* __restrict__ szzbar,
    const int64_t* __restrict__ sz, const int64_t* __restrict__ sy,
    const int64_t* __restrict__ sx,
    scalar_t* __restrict__ gwav,
    const int it, const int nt, const int nz, const int ny, const int nx,
    const scalar_t dt, const int n_src)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_src) return;
    const long i = IDX3(b, sz[b], sy[b], sx[b]);
    gwav[(long)b * nt + it] = dt * (sxxbar[i] + syybar[i] + szzbar[i]);
}

template <typename scalar_t>
__global__ void el3_adj_stress_point_kernel(
    const scalar_t* __restrict__ sxxbar, const scalar_t* __restrict__ syybar,
    const scalar_t* __restrict__ szzbar, const scalar_t* __restrict__ sxybar,
    const scalar_t* __restrict__ sxzbar, const scalar_t* __restrict__ syzbar,
    scalar_t* __restrict__ pb_vx_x, scalar_t* __restrict__ pb_vy_y,
    scalar_t* __restrict__ pb_vz_z, scalar_t* __restrict__ pb_vx_y,
    scalar_t* __restrict__ pb_vy_x, scalar_t* __restrict__ pb_vx_z,
    scalar_t* __restrict__ pb_vz_x, scalar_t* __restrict__ pb_vy_z,
    scalar_t* __restrict__ pb_vz_y,
    scalar_t* __restrict__ db_vx_x, scalar_t* __restrict__ db_vy_y,
    scalar_t* __restrict__ db_vz_z, scalar_t* __restrict__ db_vx_y,
    scalar_t* __restrict__ db_vy_x, scalar_t* __restrict__ db_vx_z,
    scalar_t* __restrict__ db_vz_x, scalar_t* __restrict__ db_vy_z,
    scalar_t* __restrict__ db_vz_y,
    scalar_t* __restrict__ gc11, scalar_t* __restrict__ gc12,
    scalar_t* __restrict__ gc13, scalar_t* __restrict__ gc33,
    scalar_t* __restrict__ gc44, scalar_t* __restrict__ gc66,
    const scalar_t* __restrict__ c11, const scalar_t* __restrict__ c12,
    const scalar_t* __restrict__ c13, const scalar_t* __restrict__ c33,
    const scalar_t* __restrict__ c44, const scalar_t* __restrict__ c66,
    const scalar_t* __restrict__ ex_h, const scalar_t* __restrict__ ey_h,
    const scalar_t* __restrict__ ez_h, const scalar_t* __restrict__ gxy_h,
    const scalar_t* __restrict__ gxz_h, const scalar_t* __restrict__ gyz_h,
    const scalar_t* __restrict__ bx, const scalar_t* __restrict__ ax, const scalar_t* __restrict__ kx,
    const scalar_t* __restrict__ by, const scalar_t* __restrict__ ay, const scalar_t* __restrict__ ky,
    const scalar_t* __restrict__ bz, const scalar_t* __restrict__ az, const scalar_t* __restrict__ kz,
    const scalar_t* __restrict__ bxh, const scalar_t* __restrict__ axh, const scalar_t* __restrict__ kxh,
    const scalar_t* __restrict__ byh, const scalar_t* __restrict__ ayh, const scalar_t* __restrict__ kyh,
    const scalar_t* __restrict__ bzh, const scalar_t* __restrict__ azh, const scalar_t* __restrict__ kzh,
    const int it, const int B,
    const int nz, const int ny, const int nx, const int ztiles, const scalar_t dt)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int zt = blockIdx.z % ztiles, b = blockIdx.z / ztiles;
    const int z = zt * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const long i = IDX3(b, z, y, x);
    const long m = MIDX3(z, y, x);
    const long h = (long)it * B * ((long)nz * ny * nx) + i;

    const scalar_t bxx = sxxbar[i], byy = syybar[i], bzz = szzbar[i];
    const scalar_t bxy = sxybar[i], bxz = sxzbar[i], byz = syzbar[i];

    const scalar_t ex = ex_h[h], ey = ey_h[h], ez = ez_h[h];
    atomicAdd(&gc11[m], dt * (ex * bxx + ey * byy));
    atomicAdd(&gc12[m], dt * (ey * bxx + ex * byy));
    atomicAdd(&gc13[m], dt * (ez * (bxx + byy) + (ex + ey) * bzz));
    atomicAdd(&gc33[m], dt * ez * bzz);
    atomicAdd(&gc66[m], dt * gxy_h[h] * bxy);
    atomicAdd(&gc44[m], dt * (gxz_h[h] * bxz + gyz_h[h] * byz));

    const scalar_t exbar = dt * (c11[m] * bxx + c12[m] * byy + c13[m] * bzz);
    const scalar_t eybar = dt * (c12[m] * bxx + c11[m] * byy + c13[m] * bzz);
    const scalar_t ezbar = dt * (c13[m] * bxx + c13[m] * byy + c33[m] * bzz);
    const scalar_t gxybar = dt * c66[m] * bxy;
    const scalar_t gxzbar = dt * c44[m] * bxz;
    const scalar_t gyzbar = dt * c44[m] * byz;

    scalar_t F;
    F = pb_vx_x[i] + exbar;  db_vx_x[i] = exbar / kx[x] + F * ax[x];  pb_vx_x[i] = F * bx[x];
    F = pb_vy_y[i] + eybar;  db_vy_y[i] = eybar / ky[y] + F * ay[y];  pb_vy_y[i] = F * by[y];
    F = pb_vz_z[i] + ezbar;  db_vz_z[i] = ezbar / kz[z] + F * az[z];  pb_vz_z[i] = F * bz[z];
    F = pb_vx_y[i] + gxybar; db_vx_y[i] = gxybar / kyh[y] + F * ayh[y]; pb_vx_y[i] = F * byh[y];
    F = pb_vy_x[i] + gxybar; db_vy_x[i] = gxybar / kxh[x] + F * axh[x]; pb_vy_x[i] = F * bxh[x];
    F = pb_vx_z[i] + gxzbar; db_vx_z[i] = gxzbar / kzh[z] + F * azh[z]; pb_vx_z[i] = F * bzh[z];
    F = pb_vz_x[i] + gxzbar; db_vz_x[i] = gxzbar / kxh[x] + F * axh[x]; pb_vz_x[i] = F * bxh[x];
    F = pb_vy_z[i] + gyzbar; db_vy_z[i] = gyzbar / kzh[z] + F * azh[z]; pb_vy_z[i] = F * bzh[z];
    F = pb_vz_y[i] + gyzbar; db_vz_y[i] = gyzbar / kyh[y] + F * ayh[y]; pb_vz_y[i] = F * byh[y];
}

// vbar += transposes: vx gets (D-x)^T db_vx_x = -D+x, (D+y)^T db_vx_y = -D-y,
// (D+z)^T db_vx_z = -D-z ; similarly vy, vz.
template <typename scalar_t>
__global__ void el3_adj_velocity_stencil_kernel(
    scalar_t* __restrict__ vxbar, scalar_t* __restrict__ vybar,
    scalar_t* __restrict__ vzbar,
    const scalar_t* __restrict__ db_vx_x, const scalar_t* __restrict__ db_vy_y,
    const scalar_t* __restrict__ db_vz_z, const scalar_t* __restrict__ db_vx_y,
    const scalar_t* __restrict__ db_vy_x, const scalar_t* __restrict__ db_vx_z,
    const scalar_t* __restrict__ db_vz_x, const scalar_t* __restrict__ db_vy_z,
    const scalar_t* __restrict__ db_vz_y,
    const scalar_t* __restrict__ fd,
    const int H, const int nz, const int ny, const int nx, const int ztiles,
    const scalar_t ix_, const scalar_t iy_, const scalar_t iz_)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int zt = blockIdx.z % ztiles, b = blockIdx.z / ztiles;
    const int z = zt * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const long i = IDX3(b, z, y, x);

    scalar_t vxa = 0, vya = 0, vza = 0;
    for (int k = 1; k <= H; ++k) {
        const scalar_t c = fd[k-1];
        int p, q;
        // vx: -D+x db_vx_x -D-y db_vx_y -D-z db_vx_z
        p = x + k; q = x - k + 1;
        vxa -= c * ix_ * (((p < nx) ? db_vx_x[IDX3(b,z,y,p)] : scalar_t(0))
                        - ((q >= 0) ? db_vx_x[IDX3(b,z,y,q)] : scalar_t(0)));
        p = y + k - 1; q = y - k;
        vxa -= c * iy_ * (((p < ny) ? db_vx_y[IDX3(b,z,p,x)] : scalar_t(0))
                        - ((q >= 0) ? db_vx_y[IDX3(b,z,q,x)] : scalar_t(0)));
        p = z + k - 1; q = z - k;
        vxa -= c * iz_ * (((p < nz) ? db_vx_z[IDX3(b,p,y,x)] : scalar_t(0))
                        - ((q >= 0) ? db_vx_z[IDX3(b,q,y,x)] : scalar_t(0)));
        // vy: -D-x db_vy_x -D+y db_vy_y -D-z db_vy_z
        p = x + k - 1; q = x - k;
        vya -= c * ix_ * (((p < nx) ? db_vy_x[IDX3(b,z,y,p)] : scalar_t(0))
                        - ((q >= 0) ? db_vy_x[IDX3(b,z,y,q)] : scalar_t(0)));
        p = y + k; q = y - k + 1;
        vya -= c * iy_ * (((p < ny) ? db_vy_y[IDX3(b,z,p,x)] : scalar_t(0))
                        - ((q >= 0) ? db_vy_y[IDX3(b,z,q,x)] : scalar_t(0)));
        p = z + k - 1; q = z - k;
        vya -= c * iz_ * (((p < nz) ? db_vy_z[IDX3(b,p,y,x)] : scalar_t(0))
                        - ((q >= 0) ? db_vy_z[IDX3(b,q,y,x)] : scalar_t(0)));
        // vz: -D-x db_vz_x -D-y db_vz_y -D+z db_vz_z
        p = x + k - 1; q = x - k;
        vza -= c * ix_ * (((p < nx) ? db_vz_x[IDX3(b,z,y,p)] : scalar_t(0))
                        - ((q >= 0) ? db_vz_x[IDX3(b,z,y,q)] : scalar_t(0)));
        p = y + k - 1; q = y - k;
        vza -= c * iy_ * (((p < ny) ? db_vz_y[IDX3(b,z,p,x)] : scalar_t(0))
                        - ((q >= 0) ? db_vz_y[IDX3(b,z,q,x)] : scalar_t(0)));
        p = z + k; q = z - k + 1;
        vza -= c * iz_ * (((p < nz) ? db_vz_z[IDX3(b,p,y,x)] : scalar_t(0))
                        - ((q >= 0) ? db_vz_z[IDX3(b,q,y,x)] : scalar_t(0)));
    }
    vxbar[i] += vxa;
    vybar[i] += vya;
    vzbar[i] += vza;
}

template <typename scalar_t>
__global__ void el3_adj_velocity_point_kernel(
    const scalar_t* __restrict__ vxbar, const scalar_t* __restrict__ vybar,
    const scalar_t* __restrict__ vzbar,
    scalar_t* __restrict__ pb_sxx_x, scalar_t* __restrict__ pb_sxy_y,
    scalar_t* __restrict__ pb_sxz_z, scalar_t* __restrict__ pb_sxy_x,
    scalar_t* __restrict__ pb_syy_y, scalar_t* __restrict__ pb_syz_z,
    scalar_t* __restrict__ pb_sxz_x, scalar_t* __restrict__ pb_syz_y,
    scalar_t* __restrict__ pb_szz_z,
    scalar_t* __restrict__ db_sxx_x, scalar_t* __restrict__ db_sxy_y,
    scalar_t* __restrict__ db_sxz_z, scalar_t* __restrict__ db_sxy_x,
    scalar_t* __restrict__ db_syy_y, scalar_t* __restrict__ db_syz_z,
    scalar_t* __restrict__ db_sxz_x, scalar_t* __restrict__ db_syz_y,
    scalar_t* __restrict__ db_szz_z,
    scalar_t* __restrict__ gB,
    const scalar_t* __restrict__ buoy_m,
    const scalar_t* __restrict__ vxh, const scalar_t* __restrict__ vyh,
    const scalar_t* __restrict__ vzh,
    const scalar_t* __restrict__ bx, const scalar_t* __restrict__ ax, const scalar_t* __restrict__ kx,
    const scalar_t* __restrict__ by, const scalar_t* __restrict__ ay, const scalar_t* __restrict__ ky,
    const scalar_t* __restrict__ bz, const scalar_t* __restrict__ az, const scalar_t* __restrict__ kz,
    const scalar_t* __restrict__ bxh, const scalar_t* __restrict__ axh, const scalar_t* __restrict__ kxh,
    const scalar_t* __restrict__ byh, const scalar_t* __restrict__ ayh, const scalar_t* __restrict__ kyh,
    const scalar_t* __restrict__ bzh, const scalar_t* __restrict__ azh, const scalar_t* __restrict__ kzh,
    const int it, const long ncell,
    const int nz, const int ny, const int nx, const int ztiles, const scalar_t dt)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int zt = blockIdx.z % ztiles, b = blockIdx.z / ztiles;
    const int z = zt * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const long i = IDX3(b, z, y, x);
    const long m = MIDX3(z, y, x);
    const long h1 = (long)(it + 1) * ncell + i;
    const long h0 = (long)it * ncell + i;

    const scalar_t vbx = vxbar[i], vby = vybar[i], vbz = vzbar[i];
    atomicAdd(&gB[m], (vbx * (vxh[h1] - vxh[h0]) + vby * (vyh[h1] - vyh[h0])
                       + vbz * (vzh[h1] - vzh[h0])) / buoy_m[m]);
    const scalar_t sx = dt * buoy_m[m] * vbx;
    const scalar_t sy = dt * buoy_m[m] * vby;
    const scalar_t sz = dt * buoy_m[m] * vbz;

    scalar_t F;
    F = pb_sxx_x[i] + sx; db_sxx_x[i] = sx / kxh[x] + F * axh[x]; pb_sxx_x[i] = F * bxh[x];
    F = pb_sxy_y[i] + sx; db_sxy_y[i] = sx / ky[y] + F * ay[y];  pb_sxy_y[i] = F * by[y];
    F = pb_sxz_z[i] + sx; db_sxz_z[i] = sx / kz[z] + F * az[z];  pb_sxz_z[i] = F * bz[z];
    F = pb_sxy_x[i] + sy; db_sxy_x[i] = sy / kx[x] + F * ax[x];  pb_sxy_x[i] = F * bx[x];
    F = pb_syy_y[i] + sy; db_syy_y[i] = sy / kyh[y] + F * ayh[y]; pb_syy_y[i] = F * byh[y];
    F = pb_syz_z[i] + sy; db_syz_z[i] = sy / kz[z] + F * az[z];  pb_syz_z[i] = F * bz[z];
    F = pb_sxz_x[i] + sz; db_sxz_x[i] = sz / kx[x] + F * ax[x];  pb_sxz_x[i] = F * bx[x];
    F = pb_syz_y[i] + sz; db_syz_y[i] = sz / ky[y] + F * ay[y];  pb_syz_y[i] = F * by[y];
    F = pb_szz_z[i] + sz; db_szz_z[i] = sz / kzh[z] + F * azh[z]; pb_szz_z[i] = F * bzh[z];
}

// sigmabar += transposes:
//   sxx: (D+x)^T db_sxx_x = -D-x        syy: -D-y db_syy_y     szz: -D-z db_szz_z
//   sxy: (D-y)^T db_sxy_y = -D+y, (D-x)^T db_sxy_x = -D+x
//   sxz: -D+z db_sxz_z, -D+x db_sxz_x    syz: -D+z db_syz_z, -D+y db_syz_y
template <typename scalar_t>
__global__ void el3_adj_stress_stencil_kernel(
    scalar_t* __restrict__ sxxbar, scalar_t* __restrict__ syybar,
    scalar_t* __restrict__ szzbar, scalar_t* __restrict__ sxybar,
    scalar_t* __restrict__ sxzbar, scalar_t* __restrict__ syzbar,
    const scalar_t* __restrict__ db_sxx_x, const scalar_t* __restrict__ db_sxy_y,
    const scalar_t* __restrict__ db_sxz_z, const scalar_t* __restrict__ db_sxy_x,
    const scalar_t* __restrict__ db_syy_y, const scalar_t* __restrict__ db_syz_z,
    const scalar_t* __restrict__ db_sxz_x, const scalar_t* __restrict__ db_syz_y,
    const scalar_t* __restrict__ db_szz_z,
    const scalar_t* __restrict__ fd,
    const int H, const int nz, const int ny, const int nx, const int ztiles,
    const scalar_t ix_, const scalar_t iy_, const scalar_t iz_)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int zt = blockIdx.z % ztiles, b = blockIdx.z / ztiles;
    const int z = zt * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const long i = IDX3(b, z, y, x);

    scalar_t axx = 0, ayy = 0, azz = 0, axy = 0, axz = 0, ayz = 0;
    for (int k = 1; k <= H; ++k) {
        const scalar_t c = fd[k-1];
        int p, q;
        p = x + k - 1; q = x - k;   // -D-x db_sxx_x
        axx -= c * ix_ * (((p < nx) ? db_sxx_x[IDX3(b,z,y,p)] : scalar_t(0))
                        - ((q >= 0) ? db_sxx_x[IDX3(b,z,y,q)] : scalar_t(0)));
        p = y + k - 1; q = y - k;
        ayy -= c * iy_ * (((p < ny) ? db_syy_y[IDX3(b,z,p,x)] : scalar_t(0))
                        - ((q >= 0) ? db_syy_y[IDX3(b,z,q,x)] : scalar_t(0)));
        p = z + k - 1; q = z - k;
        azz -= c * iz_ * (((p < nz) ? db_szz_z[IDX3(b,p,y,x)] : scalar_t(0))
                        - ((q >= 0) ? db_szz_z[IDX3(b,q,y,x)] : scalar_t(0)));
        p = y + k; q = y - k + 1;   // -D+y db_sxy_y
        axy -= c * iy_ * (((p < ny) ? db_sxy_y[IDX3(b,z,p,x)] : scalar_t(0))
                        - ((q >= 0) ? db_sxy_y[IDX3(b,z,q,x)] : scalar_t(0)));
        p = x + k; q = x - k + 1;   // -D+x db_sxy_x
        axy -= c * ix_ * (((p < nx) ? db_sxy_x[IDX3(b,z,y,p)] : scalar_t(0))
                        - ((q >= 0) ? db_sxy_x[IDX3(b,z,y,q)] : scalar_t(0)));
        p = z + k; q = z - k + 1;
        axz -= c * iz_ * (((p < nz) ? db_sxz_z[IDX3(b,p,y,x)] : scalar_t(0))
                        - ((q >= 0) ? db_sxz_z[IDX3(b,q,y,x)] : scalar_t(0)));
        p = x + k; q = x - k + 1;
        axz -= c * ix_ * (((p < nx) ? db_sxz_x[IDX3(b,z,y,p)] : scalar_t(0))
                        - ((q >= 0) ? db_sxz_x[IDX3(b,z,y,q)] : scalar_t(0)));
        p = z + k; q = z - k + 1;
        ayz -= c * iz_ * (((p < nz) ? db_syz_z[IDX3(b,p,y,x)] : scalar_t(0))
                        - ((q >= 0) ? db_syz_z[IDX3(b,q,y,x)] : scalar_t(0)));
        p = y + k; q = y - k + 1;
        ayz -= c * iy_ * (((p < ny) ? db_syz_y[IDX3(b,z,p,x)] : scalar_t(0))
                        - ((q >= 0) ? db_syz_y[IDX3(b,z,q,x)] : scalar_t(0)));
    }
    sxxbar[i] += axx; syybar[i] += ayy; szzbar[i] += azz;
    sxybar[i] += axy; sxzbar[i] += axz; syzbar[i] += ayz;
}

}  // namespace

// Long but flat drivers: forward runs V,S,src,rec[,hist] per step; backward
// runs scatter,src-grad,stress-point,velocity-stencil,velocity-point,
// stress-stencil per reverse step. All tensors contiguous CUDA.
void elastic3d_forward(
    std::vector<torch::Tensor> fields,      // 27 state fields, in eager order
    std::vector<torch::Tensor> coeffs,      // c11,c12,c13,c33,c44,c66,buoyancy
    torch::Tensor fd,
    std::vector<torch::Tensor> profiles,    // 18: (b,a,k) x (x,y,z) x (int,half)
    torch::Tensor src_z, torch::Tensor src_y, torch::Tensor src_x,
    torch::Tensor wav,
    torch::Tensor rcv_z, torch::Tensor rcv_y, torch::Tensor rcv_x,
    torch::Tensor out,
    double dt, double dx, double dy, double dz,
    std::vector<torch::Tensor> hists)       // empty OR [vxh,vyh,vzh,exh,eyh,ezh,gxyh,gxzh,gyzh]
{
    auto& vx = fields[0];
    TORCH_CHECK(vx.is_cuda() && vx.is_contiguous(), "state must be contiguous CUDA");
    const int B = vx.size(0), nz = vx.size(2), ny = vx.size(3), nx = vx.size(4);
    const int nt = wav.size(1);
    const int n_src = src_z.size(0), n_rcv = rcv_z.size(0);
    const int H = fd.size(0);
    TORCH_CHECK(H >= 1 && H <= MAX_H, "fd half-order out of range");
    const bool with_hist = !hists.empty();
    const long ncell = (long)B * nz * ny * nx;

    const dim3 threads(32, 4, 2);
    const int ztiles = (nz + 1) / 2;
    const dim3 grid((nx + 31) / 32, (ny + 3) / 4, (unsigned)(ztiles * B));
    const int sblocks = (n_src + 127) / 128;
    const dim3 rgrid((n_rcv + 127) / 128, B, 1);
    const long hblocks = (ncell + 255) / 256;
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(vx.scalar_type(), "elastic3d_forward", [&] {
        const scalar_t dt_ = (scalar_t)dt;
        const scalar_t ix_ = (scalar_t)(1.0/dx), iy_ = (scalar_t)(1.0/dy),
                       iz_ = (scalar_t)(1.0/dz);
        auto F = [&](int j) { return fields[j].data_ptr<scalar_t>(); };
        auto Cc = [&](int j) { return coeffs[j].data_ptr<scalar_t>(); };
        auto Pr = [&](int j) { return profiles[j].data_ptr<scalar_t>(); };
        for (int it = 0; it < nt; ++it) {
            el3_velocity_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                F(0), F(1), F(2), F(3), F(4), F(5), F(6), F(7), F(8),
                F(9), F(10), F(11), F(12), F(13), F(14), F(15), F(16), F(17),
                Cc(6), fd.data_ptr<scalar_t>(),
                Pr(0), Pr(1), Pr(2), Pr(3), Pr(4), Pr(5), Pr(6), Pr(7), Pr(8),
                Pr(9), Pr(10), Pr(11), Pr(12), Pr(13), Pr(14), Pr(15), Pr(16), Pr(17),
                H, nz, ny, nx, ztiles, dt_, ix_, iy_, iz_);
            el3_stress_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                F(0), F(1), F(2), F(3), F(4), F(5), F(6), F(7), F(8),
                F(18), F(19), F(20), F(21), F(22), F(23), F(24), F(25), F(26),
                Cc(0), Cc(1), Cc(2), Cc(3), Cc(4), Cc(5),
                fd.data_ptr<scalar_t>(),
                Pr(0), Pr(1), Pr(2), Pr(3), Pr(4), Pr(5), Pr(6), Pr(7), Pr(8),
                Pr(9), Pr(10), Pr(11), Pr(12), Pr(13), Pr(14), Pr(15), Pr(16), Pr(17),
                with_hist ? hists[3].data_ptr<scalar_t>() : nullptr,
                with_hist ? hists[4].data_ptr<scalar_t>() : nullptr,
                with_hist ? hists[5].data_ptr<scalar_t>() : nullptr,
                with_hist ? hists[6].data_ptr<scalar_t>() : nullptr,
                with_hist ? hists[7].data_ptr<scalar_t>() : nullptr,
                with_hist ? hists[8].data_ptr<scalar_t>() : nullptr,
                it, B, H, nz, ny, nx, ztiles, dt_, ix_, iy_, iz_);
            el3_source_kernel<scalar_t><<<sblocks, 128, 0, stream>>>(
                F(3), F(4), F(5),
                src_z.data_ptr<int64_t>(), src_y.data_ptr<int64_t>(),
                src_x.data_ptr<int64_t>(),
                wav.data_ptr<scalar_t>(), it, nt, nz, ny, nx, dt_, n_src);
            el3_record_kernel<scalar_t><<<rgrid, 128, 0, stream>>>(
                F(3), F(4), F(5),
                rcv_z.data_ptr<int64_t>(), rcv_y.data_ptr<int64_t>(),
                rcv_x.data_ptr<int64_t>(),
                out.data_ptr<scalar_t>(), it, nt, n_rcv, nz, ny, nx);
            if (with_hist) {
                el3_store_hist_kernel<scalar_t><<<hblocks, 256, 0, stream>>>(
                    F(0), F(1), F(2),
                    hists[0].data_ptr<scalar_t>(), hists[1].data_ptr<scalar_t>(),
                    hists[2].data_ptr<scalar_t>(), it, ncell);
            }
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void elastic3d_backward(
    torch::Tensor grad_out,
    std::vector<torch::Tensor> hists,       // [vxh,vyh,vzh,exh,eyh,ezh,gxyh,gxzh,gyzh]
    std::vector<torch::Tensor> coeffs,      // c11,c12,c13,c33,c44,c66,buoyancy
    torch::Tensor fd,
    std::vector<torch::Tensor> profiles,
    torch::Tensor src_z, torch::Tensor src_y, torch::Tensor src_x,
    torch::Tensor wav,
    torch::Tensor rcv_z, torch::Tensor rcv_y, torch::Tensor rcv_x,
    std::vector<torch::Tensor> adj,         // 27 zero-init adjoint fields (eager order)
    std::vector<torch::Tensor> scratch,     // 9 scratch fields
    std::vector<torch::Tensor> grads,       // gc11,gc12,gc13,gc33,gc44,gc66,gB
    torch::Tensor gwav,
    double dt, double dx, double dy, double dz)
{
    auto& vxbar = adj[0];
    const int B = vxbar.size(0), nz = vxbar.size(2), ny = vxbar.size(3),
              nx = vxbar.size(4);
    const int nt = wav.size(1);
    const int n_src = src_z.size(0), n_rcv = rcv_z.size(0);
    const int H = fd.size(0);
    const long ncell = (long)B * nz * ny * nx;

    const dim3 threads(32, 4, 2);
    const int ztiles = (nz + 1) / 2;
    const dim3 grid((nx + 31) / 32, (ny + 3) / 4, (unsigned)(ztiles * B));
    const int sblocks = (n_src + 127) / 128;
    const dim3 rgrid((n_rcv + 127) / 128, B, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(vxbar.scalar_type(), "elastic3d_backward", [&] {
        const scalar_t dt_ = (scalar_t)dt;
        const scalar_t ix_ = (scalar_t)(1.0/dx), iy_ = (scalar_t)(1.0/dy),
                       iz_ = (scalar_t)(1.0/dz);
        auto A = [&](int j) { return adj[j].data_ptr<scalar_t>(); };
        auto S = [&](int j) { return scratch[j].data_ptr<scalar_t>(); };
        auto Cc = [&](int j) { return coeffs[j].data_ptr<scalar_t>(); };
        auto Pr = [&](int j) { return profiles[j].data_ptr<scalar_t>(); };
        auto Hh = [&](int j) { return hists[j].data_ptr<scalar_t>(); };
        auto G = [&](int j) { return grads[j].data_ptr<scalar_t>(); };
        for (int it = nt - 1; it >= 0; --it) {
            el3_adj_record_scatter_kernel<scalar_t><<<rgrid, 128, 0, stream>>>(
                A(3), A(4), A(5), grad_out.data_ptr<scalar_t>(),
                rcv_z.data_ptr<int64_t>(), rcv_y.data_ptr<int64_t>(),
                rcv_x.data_ptr<int64_t>(), it, nt, n_rcv, nz, ny, nx);
            el3_adj_source_kernel<scalar_t><<<sblocks, 128, 0, stream>>>(
                A(3), A(4), A(5),
                src_z.data_ptr<int64_t>(), src_y.data_ptr<int64_t>(),
                src_x.data_ptr<int64_t>(),
                gwav.data_ptr<scalar_t>(), it, nt, nz, ny, nx, dt_, n_src);
            el3_adj_stress_point_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                A(3), A(4), A(5), A(6), A(7), A(8),
                A(18), A(19), A(20), A(21), A(22), A(23), A(24), A(25), A(26),
                S(0), S(1), S(2), S(3), S(4), S(5), S(6), S(7), S(8),
                G(0), G(1), G(2), G(3), G(4), G(5),
                Cc(0), Cc(1), Cc(2), Cc(3), Cc(4), Cc(5),
                Hh(3), Hh(4), Hh(5), Hh(6), Hh(7), Hh(8),
                Pr(0), Pr(1), Pr(2), Pr(3), Pr(4), Pr(5), Pr(6), Pr(7), Pr(8),
                Pr(9), Pr(10), Pr(11), Pr(12), Pr(13), Pr(14), Pr(15), Pr(16), Pr(17),
                it, B, nz, ny, nx, ztiles, dt_);
            el3_adj_velocity_stencil_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                A(0), A(1), A(2),
                S(0), S(1), S(2), S(3), S(4), S(5), S(6), S(7), S(8),
                fd.data_ptr<scalar_t>(), H, nz, ny, nx, ztiles, ix_, iy_, iz_);
            el3_adj_velocity_point_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                A(0), A(1), A(2),
                A(9), A(10), A(11), A(12), A(13), A(14), A(15), A(16), A(17),
                S(0), S(1), S(2), S(3), S(4), S(5), S(6), S(7), S(8),
                G(6), Cc(6),
                Hh(0), Hh(1), Hh(2),
                Pr(0), Pr(1), Pr(2), Pr(3), Pr(4), Pr(5), Pr(6), Pr(7), Pr(8),
                Pr(9), Pr(10), Pr(11), Pr(12), Pr(13), Pr(14), Pr(15), Pr(16), Pr(17),
                it, ncell, nz, ny, nx, ztiles, dt_);
            el3_adj_stress_stencil_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                A(3), A(4), A(5), A(6), A(7), A(8),
                S(0), S(1), S(2), S(3), S(4), S(5), S(6), S(7), S(8),
                fd.data_ptr<scalar_t>(), H, nz, ny, nx, ztiles, ix_, iy_, iz_);
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elastic3d_forward, "elastic 3D forward (CUDA, C++ loop)");
    m.def("backward", &elastic3d_backward, "elastic 3D adjoint (CUDA, C++ loop)");
}
