// GeoBrain G6 stage-E: native CUDA forward + adjoint for the 2-D elastic (isotropic /
// VTI-form) velocity-stress equation, INCLUDING the Levander/Robertsson
// traction-free surface in both directions.
//
// Forward step (matches equations/elastic.py exactly; note V BEFORE S):
//   V: vx += dt*B*( s~[D+x sxx] + s~[D-z sxz]  )   (fs: sxz z-read images ODD)
//      vz += dt*B*( s~[D-x sxz] + s~[D+z szz]  )   (fs: szz z-read images ODD)
//   S: e1 = s~[D-x vx'], e2 = s~[D-z vz']          (fs: vz z-read images EVEN)
//      sxx += dt*(c11 e1 + c13 e2); szz += dt*(c13 e1 + c33 e2); (fs: szz[0,:]=0)
//      g1 = s~[D+z vx'] (fs: vx EVEN), g2 = s~[D+x vz']; sxz += dt*c44*(g1+g2)
//   source: sxx,szz += dt*wav at src; record: p = -(sxx+szz)/2 at rcv.
//
// Free-surface image reads (surface = integer row 0; H = half-order):
//   backward-type z-read f[z-k] < 0      -> s * f[k-z-1]   (half-node images)
//        v2 vz EVEN s=+1 ; v3 sxz ODD s=-1
//   forward-type  z-read f[z-k+1] < 0    -> s * f[k-1-z]   (integer-node images)
//        v1 szz ODD s=-1 ; v4 vx EVEN s=+1
//
// Adjoint: plain parts via (D+)^T=-D-, (D-)^T=-D+ (zero-fill exact). The image
// terms transpose into EXTRA TAPS on the top rows only (derived from the
// operator matrix): for every variant,
//   extra[m] = sigma_v * sum_{k=m+1..H} c_k * dbar[k-1-m] / dz
// with sigma_v = -s_v (v1:+1 v2:-1 v3:+1 v4:-1); m in [1,H-1] for the
// integer-image variants (v1,v4), m in [0,H-1] for the half-image ones (v2,v3).
// The szz surface-row zeroing is a diagonal projection -> self-adjoint (mask the
// szz cotangent's row 0 before the stress-stage reverse).
//
// Coefficient gradients use STORED STRAIN HISTORIES (exact even at the fs rows,
// and immune to the source since strains are computed pre-injection):
//   gc11 += dt*e1*sxxbar ; gc13 += dt*(e2*sxxbar + e1*szzbar) ; gc33 += dt*e2*szzbar
//   gc44 += dt*gsum*sxzbar ;  gB += (vbar . dv)/B  (vx/vz histories)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace {

constexpr int MAX_H = 8;

template <typename scalar_t>
__global__ void el_velocity_kernel(
    scalar_t* __restrict__ vx, scalar_t* __restrict__ vz,
    const scalar_t* __restrict__ sxx, const scalar_t* __restrict__ szz,
    const scalar_t* __restrict__ sxz,
    scalar_t* __restrict__ psi_sxx_x, scalar_t* __restrict__ psi_sxz_z,
    scalar_t* __restrict__ psi_sxz_x, scalar_t* __restrict__ psi_szz_z,
    const scalar_t* __restrict__ buoy_m, const scalar_t* __restrict__ fd,
    const scalar_t* __restrict__ bx, const scalar_t* __restrict__ ax, const scalar_t* __restrict__ kx,
    const scalar_t* __restrict__ bz, const scalar_t* __restrict__ az, const scalar_t* __restrict__ kz,
    const scalar_t* __restrict__ bxh, const scalar_t* __restrict__ axh, const scalar_t* __restrict__ kxh,
    const scalar_t* __restrict__ bzh, const scalar_t* __restrict__ azh, const scalar_t* __restrict__ kzh,
    const int H, const int nz, const int nx, const int fs,
    const scalar_t dt, const scalar_t inv_dx, const scalar_t inv_dz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (x >= nx || z >= nz) return;
    const long plane = (long)nz * nx, base = (long)b * plane;
    const long i = base + (long)z * nx + x;
    const long m = (long)z * nx + x;

    // d_sxx_x = D+x sxx ; d_sxz_z = D-z sxz (fs: ODD half image, s=-1)
    scalar_t d1 = 0, d2 = 0;
    for (int k = 1; k <= H; ++k) {
        const int xp = x + k, xm = x - k + 1;
        d1 += fd[k-1] * (((xp < nx) ? sxx[base + (long)z*nx + xp] : scalar_t(0))
                       - ((xm >= 0) ? sxx[base + (long)z*nx + xm] : scalar_t(0)));
        const int zp = z + k - 1, zm = z - k;
        scalar_t fm;
        if (zm >= 0) fm = sxz[base + (long)zm*nx + x];
        else if (fs) fm = -sxz[base + (long)(k - z - 1)*nx + x];
        else fm = scalar_t(0);
        d2 += fd[k-1] * (((zp < nz) ? sxz[base + (long)zp*nx + x] : scalar_t(0)) - fm);
    }
    d1 *= inv_dx; d2 *= inv_dz;
    const scalar_t P1 = bxh[x] * psi_sxx_x[i] + axh[x] * d1;
    const scalar_t P2 = bz[z] * psi_sxz_z[i] + az[z] * d2;
    psi_sxx_x[i] = P1; psi_sxz_z[i] = P2;
    vx[i] += dt * buoy_m[m] * (d1 / kxh[x] + P1 + d2 / kz[z] + P2);

    // d_sxz_x = D-x sxz ; d_szz_z = D+z szz (fs: ODD integer image, s=-1)
    scalar_t d3 = 0, d4 = 0;
    for (int k = 1; k <= H; ++k) {
        const int xp = x + k - 1, xm = x - k;
        d3 += fd[k-1] * (((xp < nx) ? sxz[base + (long)z*nx + xp] : scalar_t(0))
                       - ((xm >= 0) ? sxz[base + (long)z*nx + xm] : scalar_t(0)));
        const int zp = z + k, zm = z - k + 1;
        scalar_t fm;
        if (zm >= 0) fm = szz[base + (long)zm*nx + x];
        else if (fs) fm = -szz[base + (long)(k - 1 - z)*nx + x];
        else fm = scalar_t(0);
        d4 += fd[k-1] * (((zp < nz) ? szz[base + (long)zp*nx + x] : scalar_t(0)) - fm);
    }
    d3 *= inv_dx; d4 *= inv_dz;
    const scalar_t P3 = bx[x] * psi_sxz_x[i] + ax[x] * d3;
    const scalar_t P4 = bzh[z] * psi_szz_z[i] + azh[z] * d4;
    psi_sxz_x[i] = P3; psi_szz_z[i] = P4;
    vz[i] += dt * buoy_m[m] * (d3 / kx[x] + P3 + d4 / kzh[z] + P4);
}

template <typename scalar_t>
__global__ void el_stress_kernel(
    const scalar_t* __restrict__ vx, const scalar_t* __restrict__ vz,
    scalar_t* __restrict__ sxx, scalar_t* __restrict__ szz,
    scalar_t* __restrict__ sxz,
    scalar_t* __restrict__ psi_vx_x, scalar_t* __restrict__ psi_vz_z,
    scalar_t* __restrict__ psi_vx_z, scalar_t* __restrict__ psi_vz_x,
    const scalar_t* __restrict__ c11, const scalar_t* __restrict__ c13,
    const scalar_t* __restrict__ c33, const scalar_t* __restrict__ c44,
    const scalar_t* __restrict__ fd,
    const scalar_t* __restrict__ bx, const scalar_t* __restrict__ ax, const scalar_t* __restrict__ kx,
    const scalar_t* __restrict__ bz, const scalar_t* __restrict__ az, const scalar_t* __restrict__ kz,
    const scalar_t* __restrict__ bxh, const scalar_t* __restrict__ axh, const scalar_t* __restrict__ kxh,
    const scalar_t* __restrict__ bzh, const scalar_t* __restrict__ azh, const scalar_t* __restrict__ kzh,
    scalar_t* __restrict__ e1_h, scalar_t* __restrict__ e2_h,
    scalar_t* __restrict__ gs_h,          // strain histories, slot it (or null)
    const int it, const int H, const int nz, const int nx, const int fs,
    const scalar_t dt, const scalar_t inv_dx, const scalar_t inv_dz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (x >= nx || z >= nz) return;
    const long plane = (long)nz * nx, base = (long)b * plane;
    const long i = base + (long)z * nx + x;
    const long m = (long)z * nx + x;

    // e1 = D-x vx ; e2 = D-z vz (fs: EVEN half image, s=+1)
    scalar_t d1 = 0, d2 = 0;
    for (int k = 1; k <= H; ++k) {
        const int xp = x + k - 1, xm = x - k;
        d1 += fd[k-1] * (((xp < nx) ? vx[base + (long)z*nx + xp] : scalar_t(0))
                       - ((xm >= 0) ? vx[base + (long)z*nx + xm] : scalar_t(0)));
        const int zp = z + k - 1, zm = z - k;
        scalar_t fm;
        if (zm >= 0) fm = vz[base + (long)zm*nx + x];
        else if (fs) fm = vz[base + (long)(k - z - 1)*nx + x];
        else fm = scalar_t(0);
        d2 += fd[k-1] * (((zp < nz) ? vz[base + (long)zp*nx + x] : scalar_t(0)) - fm);
    }
    d1 *= inv_dx; d2 *= inv_dz;
    const scalar_t Q1 = bx[x] * psi_vx_x[i] + ax[x] * d1;
    const scalar_t Q2 = bz[z] * psi_vz_z[i] + az[z] * d2;
    psi_vx_x[i] = Q1; psi_vz_z[i] = Q2;
    const scalar_t e1 = d1 / kx[x] + Q1;
    const scalar_t e2 = d2 / kz[z] + Q2;
    sxx[i] += dt * (c11[m] * e1 + c13[m] * e2);
    if (fs && z == 0) szz[i] = scalar_t(0);
    else szz[i] += dt * (c13[m] * e1 + c33[m] * e2);

    // g1 = D+z vx (fs: EVEN integer image, s=+1) ; g2 = D+x vz
    scalar_t d3 = 0, d4 = 0;
    for (int k = 1; k <= H; ++k) {
        const int zp = z + k, zm = z - k + 1;
        scalar_t fm;
        if (zm >= 0) fm = vx[base + (long)zm*nx + x];
        else if (fs) fm = vx[base + (long)(k - 1 - z)*nx + x];
        else fm = scalar_t(0);
        d3 += fd[k-1] * (((zp < nz) ? vx[base + (long)zp*nx + x] : scalar_t(0)) - fm);
        const int xp = x + k, xm = x - k + 1;
        d4 += fd[k-1] * (((xp < nx) ? vz[base + (long)z*nx + xp] : scalar_t(0))
                       - ((xm >= 0) ? vz[base + (long)z*nx + xm] : scalar_t(0)));
    }
    d3 *= inv_dz; d4 *= inv_dx;
    const scalar_t Q3 = bzh[z] * psi_vx_z[i] + azh[z] * d3;
    const scalar_t Q4 = bxh[x] * psi_vz_x[i] + axh[x] * d4;
    psi_vx_z[i] = Q3; psi_vz_x[i] = Q4;
    const scalar_t g1 = d3 / kzh[z] + Q3;
    const scalar_t g2 = d4 / kxh[x] + Q4;
    sxz[i] += dt * c44[m] * (g1 + g2);

    if (e1_h != nullptr) {
        const long hidx = (long)it * gridDim.z * plane + i;
        e1_h[hidx] = e1; e2_h[hidx] = e2; gs_h[hidx] = g1 + g2;
    }
}

template <typename scalar_t>
__global__ void el_source_kernel(
    scalar_t* __restrict__ sxx, scalar_t* __restrict__ szz,
    const int64_t* __restrict__ src_z, const int64_t* __restrict__ src_x,
    const scalar_t* __restrict__ wav,
    const int it, const int nt, const int nz, const int nx,
    const scalar_t dt, const int n_src)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_src) return;
    const long i = (long)b * nz * nx + (long)src_z[b] * nx + src_x[b];
    const scalar_t v = dt * wav[(long)b * nt + it];
    sxx[i] += v; szz[i] += v;
}

template <typename scalar_t>
__global__ void el_record_kernel(
    const scalar_t* __restrict__ sxx, const scalar_t* __restrict__ szz,
    const int64_t* __restrict__ rcv_z, const int64_t* __restrict__ rcv_x,
    scalar_t* __restrict__ out,
    const int it, const int nt, const int n_rcv, const int nz, const int nx,
    const int n_comp)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (r >= n_rcv) return;
    const long i = (long)b * nz * nx + (long)rcv_z[r] * nx + rcv_x[r];
    out[(((long)b * nt + it) * n_rcv + r) * n_comp] =
        scalar_t(-0.5) * (sxx[i] + szz[i]);
}

template <typename scalar_t>
__global__ void el_record_field_kernel(
    const scalar_t* __restrict__ f,
    const int64_t* __restrict__ rcv_z, const int64_t* __restrict__ rcv_x,
    scalar_t* __restrict__ out,
    const int it, const int nt, const int n_rcv, const int nz, const int nx,
    const int n_comp, const int comp)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (r >= n_rcv) return;
    const long i = (long)b * nz * nx + (long)rcv_z[r] * nx + rcv_x[r];
    out[(((long)b * nt + it) * n_rcv + r) * n_comp + comp] = f[i];
}

template <typename scalar_t>
__global__ void el_store_hist_kernel(
    const scalar_t* __restrict__ vx, const scalar_t* __restrict__ vz,
    scalar_t* __restrict__ vxh, scalar_t* __restrict__ vzh,
    const int it, const long ncell)
{
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= ncell) return;
    const long o = (long)(it + 1) * ncell + i;
    vxh[o] = vx[i]; vzh[o] = vz[i];
}

// ------------------------------ adjoint ------------------------------------

template <typename scalar_t>
__global__ void el_adj_record_scatter_kernel(
    scalar_t* __restrict__ sxxbar, scalar_t* __restrict__ szzbar,
    const scalar_t* __restrict__ gout,
    const int64_t* __restrict__ rcv_z, const int64_t* __restrict__ rcv_x,
    const int it, const int nt, const int n_rcv, const int nz, const int nx,
    const int n_comp)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (r >= n_rcv) return;
    const long i = (long)b * nz * nx + (long)rcv_z[r] * nx + rcv_x[r];
    const scalar_t g = scalar_t(-0.5)
        * gout[(((long)b * nt + it) * n_rcv + r) * n_comp];
    atomicAdd(&sxxbar[i], g);
    atomicAdd(&szzbar[i], g);
}

template <typename scalar_t>
__global__ void el_adj_scatter_field_kernel(
    scalar_t* __restrict__ fbar,
    const scalar_t* __restrict__ gout,
    const int64_t* __restrict__ rcv_z, const int64_t* __restrict__ rcv_x,
    const int it, const int nt, const int n_rcv, const int nz, const int nx,
    const int n_comp, const int comp)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (r >= n_rcv) return;
    const long i = (long)b * nz * nx + (long)rcv_z[r] * nx + rcv_x[r];
    atomicAdd(&fbar[i], gout[(((long)b * nt + it) * n_rcv + r) * n_comp + comp]);
}

template <typename scalar_t>
__global__ void el_adj_source_kernel(
    const scalar_t* __restrict__ sxxbar, const scalar_t* __restrict__ szzbar,
    const int64_t* __restrict__ src_z, const int64_t* __restrict__ src_x,
    scalar_t* __restrict__ gwav,
    const int it, const int nt, const int nz, const int nx,
    const scalar_t dt, const int n_src)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_src) return;
    const long i = (long)b * nz * nx + (long)src_z[b] * nx + src_x[b];
    gwav[(long)b * nt + it] = dt * (sxxbar[i] + szzbar[i]);
}

// stress-stage reverse, pointwise: e/g cotangents -> 4 dbar scratches, psi
// threads, c gradients. Also applies the fs szz row-0 projection to szzbar.
template <typename scalar_t>
__global__ void el_adj_stress_point_kernel(
    const scalar_t* __restrict__ sxxbar, scalar_t* __restrict__ szzbar,
    const scalar_t* __restrict__ sxzbar,
    scalar_t* __restrict__ psibar_vx_x, scalar_t* __restrict__ psibar_vz_z,
    scalar_t* __restrict__ psibar_vx_z, scalar_t* __restrict__ psibar_vz_x,
    scalar_t* __restrict__ db_vx_x, scalar_t* __restrict__ db_vz_z,
    scalar_t* __restrict__ db_vx_z, scalar_t* __restrict__ db_vz_x,
    scalar_t* __restrict__ gc11, scalar_t* __restrict__ gc13,
    scalar_t* __restrict__ gc33, scalar_t* __restrict__ gc44,
    const scalar_t* __restrict__ c11, const scalar_t* __restrict__ c13,
    const scalar_t* __restrict__ c33, const scalar_t* __restrict__ c44,
    const scalar_t* __restrict__ e1_h, const scalar_t* __restrict__ e2_h,
    const scalar_t* __restrict__ gs_h,
    const scalar_t* __restrict__ bx, const scalar_t* __restrict__ ax, const scalar_t* __restrict__ kx,
    const scalar_t* __restrict__ bz, const scalar_t* __restrict__ az, const scalar_t* __restrict__ kz,
    const scalar_t* __restrict__ bxh, const scalar_t* __restrict__ axh, const scalar_t* __restrict__ kxh,
    const scalar_t* __restrict__ bzh, const scalar_t* __restrict__ azh, const scalar_t* __restrict__ kzh,
    const int it, const int B, const int H, const int nz, const int nx,
    const int fs, const scalar_t dt)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (x >= nx || z >= nz) return;
    const long plane = (long)nz * nx, base = (long)b * plane;
    const long i = base + (long)z * nx + x;
    const long m = (long)z * nx + x;
    const long h = (long)it * B * plane + i;

    scalar_t sb_zz = szzbar[i];
    if (fs && z == 0) { sb_zz = scalar_t(0); szzbar[i] = scalar_t(0); }
    const scalar_t sb_xx = sxxbar[i];
    const scalar_t sb_xz = sxzbar[i];

    const scalar_t e1 = e1_h[h], e2 = e2_h[h], gs = gs_h[h];
    atomicAdd(&gc11[m], dt * e1 * sb_xx);
    atomicAdd(&gc13[m], dt * (e2 * sb_xx + e1 * sb_zz));
    atomicAdd(&gc33[m], dt * e2 * sb_zz);
    atomicAdd(&gc44[m], dt * gs * sb_xz);

    const scalar_t e1bar = dt * (c11[m] * sb_xx + c13[m] * sb_zz);
    const scalar_t e2bar = dt * (c13[m] * sb_xx + c33[m] * sb_zz);
    const scalar_t gbar = dt * c44[m] * sb_xz;

    const scalar_t F1 = psibar_vx_x[i] + e1bar;   // e1 path: x-int profiles
    const scalar_t F2 = psibar_vz_z[i] + e2bar;   // e2 path: z-int
    const scalar_t F3 = psibar_vx_z[i] + gbar;    // g1 path: z-half
    const scalar_t F4 = psibar_vz_x[i] + gbar;    // g2 path: x-half
    db_vx_x[i] = e1bar / kx[x] + F1 * ax[x];
    db_vz_z[i] = e2bar / kz[z] + F2 * az[z];
    db_vx_z[i] = gbar / kzh[z] + F3 * azh[z];
    db_vz_x[i] = gbar / kxh[x] + F4 * axh[x];
    psibar_vx_x[i] = F1 * bx[x];
    psibar_vz_z[i] = F2 * bz[z];
    psibar_vx_z[i] = F3 * bzh[z];
    psibar_vz_x[i] = F4 * bxh[x];
}

// vbar += transposes of the strain operators. Plain parts:
//   vxbar += -D+x db_vx_x  - D-z db_vx_z      (e1: (D-x)^T; g1: (D+z)^T)
//   vzbar += -D+z db_vz_z  - D-x db_vz_x      (e2: (D-z)^T; g2: (D+x)^T)
// fs extra taps (top rows): g1 is v4 (integer EVEN, sigma=-1) on db_vx_z;
//                           e2 is v2 (half EVEN,   sigma=-1) on db_vz_z.
template <typename scalar_t>
__global__ void el_adj_velocity_stencil_kernel(
    scalar_t* __restrict__ vxbar, scalar_t* __restrict__ vzbar,
    const scalar_t* __restrict__ db_vx_x, const scalar_t* __restrict__ db_vz_z,
    const scalar_t* __restrict__ db_vx_z, const scalar_t* __restrict__ db_vz_x,
    const scalar_t* __restrict__ fd,
    const int H, const int nz, const int nx, const int fs,
    const scalar_t inv_dx, const scalar_t inv_dz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (x >= nx || z >= nz) return;
    const long base = (long)b * nz * nx;
    const long i = base + (long)z * nx + x;

    scalar_t accx = 0;  // D+x db_vx_x
    scalar_t accz = 0;  // D-z db_vx_z
    scalar_t accz2 = 0; // D+z db_vz_z
    scalar_t accx2 = 0; // D-x db_vz_x
    for (int k = 1; k <= H; ++k) {
        const int xpf = x + k, xmf = x - k + 1;
        accx += fd[k-1] * (((xpf < nx) ? db_vx_x[base + (long)z*nx + xpf] : scalar_t(0))
                         - ((xmf >= 0) ? db_vx_x[base + (long)z*nx + xmf] : scalar_t(0)));
        const int zpb = z + k - 1, zmb = z - k;
        accz += fd[k-1] * (((zpb < nz) ? db_vx_z[base + (long)zpb*nx + x] : scalar_t(0))
                         - ((zmb >= 0) ? db_vx_z[base + (long)zmb*nx + x] : scalar_t(0)));
        const int zpf = z + k, zmf = z - k + 1;
        accz2 += fd[k-1] * (((zpf < nz) ? db_vz_z[base + (long)zpf*nx + x] : scalar_t(0))
                          - ((zmf >= 0) ? db_vz_z[base + (long)zmf*nx + x] : scalar_t(0)));
        const int xpb = x + k - 1, xmb = x - k;
        accx2 += fd[k-1] * (((xpb < nx) ? db_vz_x[base + (long)xpb + (long)z*nx] : scalar_t(0))
                          - ((xmb >= 0) ? db_vz_x[base + (long)xmb + (long)z*nx] : scalar_t(0)));
    }
    scalar_t vx_add = -(accx * inv_dx) - (accz * inv_dz);
    scalar_t vz_add = -(accz2 * inv_dz) - (accx2 * inv_dx);
    if (fs && z <= H - 1) {
        scalar_t exz = 0, ezz = 0;
        for (int k = z + 1; k <= H; ++k) {
            exz += fd[k-1] * db_vx_z[base + (long)(k - 1 - z)*nx + x];
            ezz += fd[k-1] * db_vz_z[base + (long)(k - 1 - z)*nx + x];
        }
        if (z >= 1) vx_add += scalar_t(-1) * exz * inv_dz;  // v4: integer EVEN
        vz_add += scalar_t(-1) * ezz * inv_dz;              // v2: half EVEN
    }
    vxbar[i] += vx_add;
    vzbar[i] += vz_add;
}

// velocity-stage reverse, pointwise: vbar -> 4 dbar scratches, psi threads, gB.
template <typename scalar_t>
__global__ void el_adj_velocity_point_kernel(
    const scalar_t* __restrict__ vxbar, const scalar_t* __restrict__ vzbar,
    scalar_t* __restrict__ psibar_sxx_x, scalar_t* __restrict__ psibar_sxz_z,
    scalar_t* __restrict__ psibar_sxz_x, scalar_t* __restrict__ psibar_szz_z,
    scalar_t* __restrict__ db_sxx_x, scalar_t* __restrict__ db_sxz_z,
    scalar_t* __restrict__ db_sxz_x, scalar_t* __restrict__ db_szz_z,
    scalar_t* __restrict__ gB,
    const scalar_t* __restrict__ buoy_m,
    const scalar_t* __restrict__ vxh, const scalar_t* __restrict__ vzh,
    const scalar_t* __restrict__ bx, const scalar_t* __restrict__ ax, const scalar_t* __restrict__ kx,
    const scalar_t* __restrict__ bz, const scalar_t* __restrict__ az, const scalar_t* __restrict__ kz,
    const scalar_t* __restrict__ bxh, const scalar_t* __restrict__ axh, const scalar_t* __restrict__ kxh,
    const scalar_t* __restrict__ bzh, const scalar_t* __restrict__ azh, const scalar_t* __restrict__ kzh,
    const int it, const long ncell, const int nz, const int nx,
    const scalar_t dt)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (x >= nx || z >= nz) return;
    const long plane = (long)nz * nx;
    const long i = (long)b * plane + (long)z * nx + x;
    const long m = (long)z * nx + x;
    const long h1 = (long)(it + 1) * ncell + i;
    const long h0 = (long)it * ncell + i;

    const scalar_t vbx = vxbar[i], vbz = vzbar[i];
    atomicAdd(&gB[m], (vbx * (vxh[h1] - vxh[h0]) + vbz * (vzh[h1] - vzh[h0]))
                      / buoy_m[m]);
    const scalar_t sx = dt * buoy_m[m] * vbx;   // POSITIVE: v += dt*B*(...)
    const scalar_t sz = dt * buoy_m[m] * vbz;
    const scalar_t G1 = psibar_sxx_x[i] + sx;   // d_sxx_x path: x-half
    const scalar_t G2 = psibar_sxz_z[i] + sx;   // d_sxz_z path: z-int
    const scalar_t G3 = psibar_sxz_x[i] + sz;   // d_sxz_x path: x-int
    const scalar_t G4 = psibar_szz_z[i] + sz;   // d_szz_z path: z-half
    db_sxx_x[i] = sx / kxh[x] + G1 * axh[x];
    db_sxz_z[i] = sx / kz[z] + G2 * az[z];
    db_sxz_x[i] = sz / kx[x] + G3 * ax[x];
    db_szz_z[i] = sz / kzh[z] + G4 * azh[z];
    psibar_sxx_x[i] = G1 * bxh[x];
    psibar_sxz_z[i] = G2 * bz[z];
    psibar_sxz_x[i] = G3 * bx[x];
    psibar_szz_z[i] = G4 * bzh[z];
}

// sigmabar += transposes of the velocity-stage operators. Plain parts:
//   sxxbar += -D-x db_sxx_x                     ((D+x)^T)
//   szzbar += -D-z db_szz_z                     ((D+z)^T)  + fs extra v1 (sigma=+1)
//   sxzbar += -D+z db_sxz_z - D+x db_sxz_x      ((D-z)^T, (D-x)^T) + fs extra v3 (sigma=+1)
template <typename scalar_t>
__global__ void el_adj_stress_stencil_kernel(
    scalar_t* __restrict__ sxxbar, scalar_t* __restrict__ szzbar,
    scalar_t* __restrict__ sxzbar,
    const scalar_t* __restrict__ db_sxx_x, const scalar_t* __restrict__ db_sxz_z,
    const scalar_t* __restrict__ db_sxz_x, const scalar_t* __restrict__ db_szz_z,
    const scalar_t* __restrict__ fd,
    const int H, const int nz, const int nx, const int fs,
    const scalar_t inv_dx, const scalar_t inv_dz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (x >= nx || z >= nz) return;
    const long base = (long)b * nz * nx;
    const long i = base + (long)z * nx + x;

    scalar_t a_xx = 0;   // D-x db_sxx_x
    scalar_t a_zz = 0;   // D-z db_szz_z
    scalar_t a_xz1 = 0;  // D+z db_sxz_z
    scalar_t a_xz2 = 0;  // D+x db_sxz_x
    for (int k = 1; k <= H; ++k) {
        const int xpb = x + k - 1, xmb = x - k;
        a_xx += fd[k-1] * (((xpb < nx) ? db_sxx_x[base + (long)z*nx + xpb] : scalar_t(0))
                         - ((xmb >= 0) ? db_sxx_x[base + (long)z*nx + xmb] : scalar_t(0)));
        const int zpb = z + k - 1, zmb = z - k;
        a_zz += fd[k-1] * (((zpb < nz) ? db_szz_z[base + (long)zpb*nx + x] : scalar_t(0))
                         - ((zmb >= 0) ? db_szz_z[base + (long)zmb*nx + x] : scalar_t(0)));
        const int zpf = z + k, zmf = z - k + 1;
        a_xz1 += fd[k-1] * (((zpf < nz) ? db_sxz_z[base + (long)zpf*nx + x] : scalar_t(0))
                          - ((zmf >= 0) ? db_sxz_z[base + (long)zmf*nx + x] : scalar_t(0)));
        const int xpf = x + k, xmf = x - k + 1;
        a_xz2 += fd[k-1] * (((xpf < nx) ? db_sxz_x[base + (long)z*nx + xpf] : scalar_t(0))
                          - ((xmf >= 0) ? db_sxz_x[base + (long)z*nx + xmf] : scalar_t(0)));
    }
    sxxbar[i] -= a_xx * inv_dx;
    scalar_t zz_add = -(a_zz * inv_dz);
    scalar_t xz_add = -(a_xz1 * inv_dz) - (a_xz2 * inv_dx);
    if (fs && z <= H - 1) {
        scalar_t ezz = 0, exz = 0;
        for (int k = z + 1; k <= H; ++k) {
            ezz += fd[k-1] * db_szz_z[base + (long)(k - 1 - z)*nx + x];
            exz += fd[k-1] * db_sxz_z[base + (long)(k - 1 - z)*nx + x];
        }
        if (z >= 1) zz_add += ezz * inv_dz;   // v1: integer ODD, sigma=+1
        xz_add += exz * inv_dz;               // v3: half ODD,    sigma=+1
    }
    szzbar[i] += zz_add;
    sxzbar[i] += xz_add;
}

}  // namespace

void elastic2d_forward(
    torch::Tensor vx, torch::Tensor vz, torch::Tensor sxx, torch::Tensor szz,
    torch::Tensor sxz,
    torch::Tensor psi_sxx_x, torch::Tensor psi_sxz_z, torch::Tensor psi_sxz_x,
    torch::Tensor psi_szz_z, torch::Tensor psi_vx_x, torch::Tensor psi_vz_z,
    torch::Tensor psi_vx_z, torch::Tensor psi_vz_x,
    torch::Tensor c11, torch::Tensor c13, torch::Tensor c33, torch::Tensor c44,
    torch::Tensor buoyancy, torch::Tensor fd,
    torch::Tensor bx_int, torch::Tensor ax_int, torch::Tensor kx_int,
    torch::Tensor bz_int, torch::Tensor az_int, torch::Tensor kz_int,
    torch::Tensor bx_half, torch::Tensor ax_half, torch::Tensor kx_half,
    torch::Tensor bz_half, torch::Tensor az_half, torch::Tensor kz_half,
    torch::Tensor src_z, torch::Tensor src_x, torch::Tensor wav,
    torch::Tensor rcv_z, torch::Tensor rcv_x, torch::Tensor out,
    double dt, double dx, double dz, bool free_surface,
    torch::Tensor vx_hist, torch::Tensor vz_hist,
    torch::Tensor e1_hist, torch::Tensor e2_hist, torch::Tensor gs_hist,
    int64_t n_comp)
{
    TORCH_CHECK(vx.is_cuda() && vx.is_contiguous(), "vx must be contiguous CUDA");
    const int B = vx.size(0), nz = vx.size(2), nx = vx.size(3);
    const int nt = wav.size(1);
    const int n_src = src_z.size(0), n_rcv = rcv_z.size(0);
    const int H = fd.size(0);
    TORCH_CHECK(H >= 1 && H <= MAX_H, "fd half-order out of range");
    TORCH_CHECK(n_comp == 1 || n_comp == 3, "n_comp must be 1 (p) or 3 (p,vx,vz)");
    const int C = (int)n_comp;
    const bool with_hist = vx_hist.numel() > 0;
    const long ncell = (long)B * nz * nx;
    const int fs = free_surface ? 1 : 0;

    const dim3 threads(32, 8, 1);
    const dim3 grid((nx + 31) / 32, (nz + 7) / 8, B);
    const int sblocks = (n_src + 127) / 128;
    const dim3 rgrid((n_rcv + 127) / 128, B, 1);
    const long hblocks = (ncell + 255) / 256;
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(vx.scalar_type(), "elastic2d_forward", [&] {
        const scalar_t dt_ = (scalar_t)dt;
        const scalar_t ix_ = (scalar_t)(1.0 / dx), iz_ = (scalar_t)(1.0 / dz);
        for (int it = 0; it < nt; ++it) {
            el_velocity_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                vx.data_ptr<scalar_t>(), vz.data_ptr<scalar_t>(),
                sxx.data_ptr<scalar_t>(), szz.data_ptr<scalar_t>(),
                sxz.data_ptr<scalar_t>(),
                psi_sxx_x.data_ptr<scalar_t>(), psi_sxz_z.data_ptr<scalar_t>(),
                psi_sxz_x.data_ptr<scalar_t>(), psi_szz_z.data_ptr<scalar_t>(),
                buoyancy.data_ptr<scalar_t>(), fd.data_ptr<scalar_t>(),
                bx_int.data_ptr<scalar_t>(), ax_int.data_ptr<scalar_t>(), kx_int.data_ptr<scalar_t>(),
                bz_int.data_ptr<scalar_t>(), az_int.data_ptr<scalar_t>(), kz_int.data_ptr<scalar_t>(),
                bx_half.data_ptr<scalar_t>(), ax_half.data_ptr<scalar_t>(), kx_half.data_ptr<scalar_t>(),
                bz_half.data_ptr<scalar_t>(), az_half.data_ptr<scalar_t>(), kz_half.data_ptr<scalar_t>(),
                H, nz, nx, fs, dt_, ix_, iz_);
            el_stress_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                vx.data_ptr<scalar_t>(), vz.data_ptr<scalar_t>(),
                sxx.data_ptr<scalar_t>(), szz.data_ptr<scalar_t>(),
                sxz.data_ptr<scalar_t>(),
                psi_vx_x.data_ptr<scalar_t>(), psi_vz_z.data_ptr<scalar_t>(),
                psi_vx_z.data_ptr<scalar_t>(), psi_vz_x.data_ptr<scalar_t>(),
                c11.data_ptr<scalar_t>(), c13.data_ptr<scalar_t>(),
                c33.data_ptr<scalar_t>(), c44.data_ptr<scalar_t>(),
                fd.data_ptr<scalar_t>(),
                bx_int.data_ptr<scalar_t>(), ax_int.data_ptr<scalar_t>(), kx_int.data_ptr<scalar_t>(),
                bz_int.data_ptr<scalar_t>(), az_int.data_ptr<scalar_t>(), kz_int.data_ptr<scalar_t>(),
                bx_half.data_ptr<scalar_t>(), ax_half.data_ptr<scalar_t>(), kx_half.data_ptr<scalar_t>(),
                bz_half.data_ptr<scalar_t>(), az_half.data_ptr<scalar_t>(), kz_half.data_ptr<scalar_t>(),
                with_hist ? e1_hist.data_ptr<scalar_t>() : nullptr,
                with_hist ? e2_hist.data_ptr<scalar_t>() : nullptr,
                with_hist ? gs_hist.data_ptr<scalar_t>() : nullptr,
                it, H, nz, nx, fs, dt_, ix_, iz_);
            el_source_kernel<scalar_t><<<sblocks, 128, 0, stream>>>(
                sxx.data_ptr<scalar_t>(), szz.data_ptr<scalar_t>(),
                src_z.data_ptr<int64_t>(), src_x.data_ptr<int64_t>(),
                wav.data_ptr<scalar_t>(), it, nt, nz, nx, dt_, n_src);
            el_record_kernel<scalar_t><<<rgrid, 128, 0, stream>>>(
                sxx.data_ptr<scalar_t>(), szz.data_ptr<scalar_t>(),
                rcv_z.data_ptr<int64_t>(), rcv_x.data_ptr<int64_t>(),
                out.data_ptr<scalar_t>(), it, nt, n_rcv, nz, nx, C);
            if (C == 3) {
                el_record_field_kernel<scalar_t><<<rgrid, 128, 0, stream>>>(
                    vx.data_ptr<scalar_t>(),
                    rcv_z.data_ptr<int64_t>(), rcv_x.data_ptr<int64_t>(),
                    out.data_ptr<scalar_t>(), it, nt, n_rcv, nz, nx, C, 1);
                el_record_field_kernel<scalar_t><<<rgrid, 128, 0, stream>>>(
                    vz.data_ptr<scalar_t>(),
                    rcv_z.data_ptr<int64_t>(), rcv_x.data_ptr<int64_t>(),
                    out.data_ptr<scalar_t>(), it, nt, n_rcv, nz, nx, C, 2);
            }
            if (with_hist) {
                el_store_hist_kernel<scalar_t><<<hblocks, 256, 0, stream>>>(
                    vx.data_ptr<scalar_t>(), vz.data_ptr<scalar_t>(),
                    vx_hist.data_ptr<scalar_t>(), vz_hist.data_ptr<scalar_t>(),
                    it, ncell);
            }
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void elastic2d_backward(
    torch::Tensor grad_out,
    torch::Tensor vx_hist, torch::Tensor vz_hist,
    torch::Tensor e1_hist, torch::Tensor e2_hist, torch::Tensor gs_hist,
    torch::Tensor c11, torch::Tensor c13, torch::Tensor c33, torch::Tensor c44,
    torch::Tensor buoyancy, torch::Tensor fd,
    torch::Tensor bx_int, torch::Tensor ax_int, torch::Tensor kx_int,
    torch::Tensor bz_int, torch::Tensor az_int, torch::Tensor kz_int,
    torch::Tensor bx_half, torch::Tensor ax_half, torch::Tensor kx_half,
    torch::Tensor bz_half, torch::Tensor az_half, torch::Tensor kz_half,
    torch::Tensor src_z, torch::Tensor src_x, torch::Tensor wav,
    torch::Tensor rcv_z, torch::Tensor rcv_x,
    torch::Tensor vxbar, torch::Tensor vzbar, torch::Tensor sxxbar,
    torch::Tensor szzbar, torch::Tensor sxzbar,
    torch::Tensor psibar_sxx_x, torch::Tensor psibar_sxz_z,
    torch::Tensor psibar_sxz_x, torch::Tensor psibar_szz_z,
    torch::Tensor psibar_vx_x, torch::Tensor psibar_vz_z,
    torch::Tensor psibar_vx_z, torch::Tensor psibar_vz_x,
    torch::Tensor db1, torch::Tensor db2, torch::Tensor db3, torch::Tensor db4,
    torch::Tensor gc11, torch::Tensor gc13, torch::Tensor gc33,
    torch::Tensor gc44, torch::Tensor gB, torch::Tensor gwav,
    double dt, double dx, double dz, bool free_surface, int64_t n_comp)
{
    const int C = (int)n_comp;
    const int B = vxbar.size(0), nz = vxbar.size(2), nx = vxbar.size(3);
    const int nt = wav.size(1);
    const int n_src = src_z.size(0), n_rcv = rcv_z.size(0);
    const int H = fd.size(0);
    const long ncell = (long)B * nz * nx;
    const int fs = free_surface ? 1 : 0;

    const dim3 threads(32, 8, 1);
    const dim3 grid((nx + 31) / 32, (nz + 7) / 8, B);
    const int sblocks = (n_src + 127) / 128;
    const dim3 rgrid((n_rcv + 127) / 128, B, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(vxbar.scalar_type(), "elastic2d_backward", [&] {
        const scalar_t dt_ = (scalar_t)dt;
        const scalar_t ix_ = (scalar_t)(1.0 / dx), iz_ = (scalar_t)(1.0 / dz);
        for (int it = nt - 1; it >= 0; --it) {
            el_adj_record_scatter_kernel<scalar_t><<<rgrid, 128, 0, stream>>>(
                sxxbar.data_ptr<scalar_t>(), szzbar.data_ptr<scalar_t>(),
                grad_out.data_ptr<scalar_t>(),
                rcv_z.data_ptr<int64_t>(), rcv_x.data_ptr<int64_t>(),
                it, nt, n_rcv, nz, nx, C);
            if (C == 3) {
                el_adj_scatter_field_kernel<scalar_t><<<rgrid, 128, 0, stream>>>(
                    vxbar.data_ptr<scalar_t>(), grad_out.data_ptr<scalar_t>(),
                    rcv_z.data_ptr<int64_t>(), rcv_x.data_ptr<int64_t>(),
                    it, nt, n_rcv, nz, nx, C, 1);
                el_adj_scatter_field_kernel<scalar_t><<<rgrid, 128, 0, stream>>>(
                    vzbar.data_ptr<scalar_t>(), grad_out.data_ptr<scalar_t>(),
                    rcv_z.data_ptr<int64_t>(), rcv_x.data_ptr<int64_t>(),
                    it, nt, n_rcv, nz, nx, C, 2);
            }
            el_adj_source_kernel<scalar_t><<<sblocks, 128, 0, stream>>>(
                sxxbar.data_ptr<scalar_t>(), szzbar.data_ptr<scalar_t>(),
                src_z.data_ptr<int64_t>(), src_x.data_ptr<int64_t>(),
                gwav.data_ptr<scalar_t>(), it, nt, nz, nx, dt_, n_src);
            el_adj_stress_point_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                sxxbar.data_ptr<scalar_t>(), szzbar.data_ptr<scalar_t>(),
                sxzbar.data_ptr<scalar_t>(),
                psibar_vx_x.data_ptr<scalar_t>(), psibar_vz_z.data_ptr<scalar_t>(),
                psibar_vx_z.data_ptr<scalar_t>(), psibar_vz_x.data_ptr<scalar_t>(),
                db1.data_ptr<scalar_t>(), db2.data_ptr<scalar_t>(),
                db3.data_ptr<scalar_t>(), db4.data_ptr<scalar_t>(),
                gc11.data_ptr<scalar_t>(), gc13.data_ptr<scalar_t>(),
                gc33.data_ptr<scalar_t>(), gc44.data_ptr<scalar_t>(),
                c11.data_ptr<scalar_t>(), c13.data_ptr<scalar_t>(),
                c33.data_ptr<scalar_t>(), c44.data_ptr<scalar_t>(),
                e1_hist.data_ptr<scalar_t>(), e2_hist.data_ptr<scalar_t>(),
                gs_hist.data_ptr<scalar_t>(),
                bx_int.data_ptr<scalar_t>(), ax_int.data_ptr<scalar_t>(), kx_int.data_ptr<scalar_t>(),
                bz_int.data_ptr<scalar_t>(), az_int.data_ptr<scalar_t>(), kz_int.data_ptr<scalar_t>(),
                bx_half.data_ptr<scalar_t>(), ax_half.data_ptr<scalar_t>(), kx_half.data_ptr<scalar_t>(),
                bz_half.data_ptr<scalar_t>(), az_half.data_ptr<scalar_t>(), kz_half.data_ptr<scalar_t>(),
                it, B, H, nz, nx, fs, dt_);
            el_adj_velocity_stencil_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                vxbar.data_ptr<scalar_t>(), vzbar.data_ptr<scalar_t>(),
                db1.data_ptr<scalar_t>(), db2.data_ptr<scalar_t>(),
                db3.data_ptr<scalar_t>(), db4.data_ptr<scalar_t>(),
                fd.data_ptr<scalar_t>(), H, nz, nx, fs, ix_, iz_);
            el_adj_velocity_point_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                vxbar.data_ptr<scalar_t>(), vzbar.data_ptr<scalar_t>(),
                psibar_sxx_x.data_ptr<scalar_t>(), psibar_sxz_z.data_ptr<scalar_t>(),
                psibar_sxz_x.data_ptr<scalar_t>(), psibar_szz_z.data_ptr<scalar_t>(),
                db1.data_ptr<scalar_t>(), db2.data_ptr<scalar_t>(),
                db3.data_ptr<scalar_t>(), db4.data_ptr<scalar_t>(),
                gB.data_ptr<scalar_t>(), buoyancy.data_ptr<scalar_t>(),
                vx_hist.data_ptr<scalar_t>(), vz_hist.data_ptr<scalar_t>(),
                bx_int.data_ptr<scalar_t>(), ax_int.data_ptr<scalar_t>(), kx_int.data_ptr<scalar_t>(),
                bz_int.data_ptr<scalar_t>(), az_int.data_ptr<scalar_t>(), kz_int.data_ptr<scalar_t>(),
                bx_half.data_ptr<scalar_t>(), ax_half.data_ptr<scalar_t>(), kx_half.data_ptr<scalar_t>(),
                bz_half.data_ptr<scalar_t>(), az_half.data_ptr<scalar_t>(), kz_half.data_ptr<scalar_t>(),
                it, ncell, nz, nx, dt_);
            el_adj_stress_stencil_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                sxxbar.data_ptr<scalar_t>(), szzbar.data_ptr<scalar_t>(),
                sxzbar.data_ptr<scalar_t>(),
                db1.data_ptr<scalar_t>(), db2.data_ptr<scalar_t>(),
                db3.data_ptr<scalar_t>(), db4.data_ptr<scalar_t>(),
                fd.data_ptr<scalar_t>(), H, nz, nx, fs, ix_, iz_);
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elastic2d_forward, "elastic 2D forward (CUDA, C++ loop)");
    m.def("backward", &elastic2d_backward, "elastic 2D adjoint (CUDA, C++ loop)");
}
