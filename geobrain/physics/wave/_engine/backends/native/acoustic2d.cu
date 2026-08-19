// GeoBrain G6 stage-B: native CUDA forward for the 2-D acoustic velocity-stress equation.
//
// Numerically replicates equations/acoustic.py step-for-step:
//   per time step (matching eager _step_once + record order exactly):
//     1) pressure update   p -= dt*kappa*( s~x + s~z )         [D- stencils, CPML int]
//     2) velocity update   v -= dt*buoyancy * t~                [D+ stencils, CPML half]
//     3) source injection  p[b, src_z[b], src_x[b]] += dt*wav[b, it]
//     4) receiver record   out[b, it, r] = p[b, rcv_z[r], rcv_x[r]]
//   with D- f[i] = (1/h) sum_k c_k ( f[i+k-1] - f[i-k] )   (zero fill outside)
//        D+ f[i] = (1/h) sum_k c_k ( f[i+k]   - f[i-k+1] ) (zero fill outside)
//   CPML: psi <- b*psi + a*d ;  d~ = d/kappa + psi   (profiles per axis, int/half offset)
//
// The whole nt loop runs in C++ (zero Python dispatch per step); the eager engine
// remains the correctness reference and the autograd path (this is forward-only).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

constexpr int MAX_H = 8;  // supports fd_order up to 16

template <typename scalar_t>
__global__ void pressure_kernel(
    scalar_t* __restrict__ p,
    const scalar_t* __restrict__ vx,
    const scalar_t* __restrict__ vz,
    scalar_t* __restrict__ psi_vxx,
    scalar_t* __restrict__ psi_vzz,
    const scalar_t* __restrict__ kappa_m,   // (nz*nx) bulk modulus
    const scalar_t* __restrict__ fd,        // (H,) stencil taps
    const scalar_t* __restrict__ bx, const scalar_t* __restrict__ ax,
    const scalar_t* __restrict__ kx,        // (nx,) int-offset x profiles
    const scalar_t* __restrict__ bz, const scalar_t* __restrict__ az,
    const scalar_t* __restrict__ kz,        // (nz,) int-offset z profiles
    scalar_t* __restrict__ ph,              // hist slot it+1 (nullptr = off);
                                            // pre-source value, source_kernel
                                            // overwrites at src cells
    const int it, const int B,
    const int H, const int nz, const int nx,
    const scalar_t dt, const scalar_t inv_dx, const scalar_t inv_dz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (x >= nx || z >= nz) return;
    const long plane = (long)nz * nx;
    const long base = (long)b * plane;
    const long i = base + (long)z * nx + x;

    // D- along x on vx: sum_k c_k ( vx[x+k-1] - vx[x-k] )
    scalar_t dvx = 0;
    for (int k = 1; k <= H; ++k) {
        const int xp = x + k - 1, xm = x - k;
        const scalar_t fp = (xp < nx) ? vx[base + (long)z * nx + xp] : scalar_t(0);
        const scalar_t fm = (xm >= 0) ? vx[base + (long)z * nx + xm] : scalar_t(0);
        dvx += fd[k - 1] * (fp - fm);
    }
    dvx *= inv_dx;
    // D- along z on vz
    scalar_t dvz = 0;
    for (int k = 1; k <= H; ++k) {
        const int zp = z + k - 1, zm = z - k;
        const scalar_t fp = (zp < nz) ? vz[base + (long)zp * nx + x] : scalar_t(0);
        const scalar_t fm = (zm >= 0) ? vz[base + (long)zm * nx + x] : scalar_t(0);
        dvz += fd[k - 1] * (fp - fm);
    }
    dvz *= inv_dz;

    // CPML stretch (int profiles); psi updated in place
    const scalar_t px_new = bx[x] * psi_vxx[i] + ax[x] * dvx;
    const scalar_t pz_new = bz[z] * psi_vzz[i] + az[z] * dvz;
    psi_vxx[i] = px_new;
    psi_vzz[i] = pz_new;
    const scalar_t sx = dvx / kx[x] + px_new;
    const scalar_t sz = dvz / kz[z] + pz_new;

    const scalar_t p_new = p[i] - dt * kappa_m[(long)z * nx + x] * (sx + sz);
    p[i] = p_new;
    if (ph != nullptr) ph[(long)(it + 1) * B * plane + i] = p_new;
}

template <typename scalar_t>
__global__ void velocity_kernel(
    const scalar_t* __restrict__ p,
    scalar_t* __restrict__ vx,
    scalar_t* __restrict__ vz,
    scalar_t* __restrict__ psi_px,
    scalar_t* __restrict__ psi_pz,
    const scalar_t* __restrict__ buoy_m,    // (nz*nx)
    const scalar_t* __restrict__ fd,
    const scalar_t* __restrict__ bxh, const scalar_t* __restrict__ axh,
    const scalar_t* __restrict__ kxh,       // (nx,) half-offset x profiles
    const scalar_t* __restrict__ bzh, const scalar_t* __restrict__ azh,
    const scalar_t* __restrict__ kzh,       // (nz,) half-offset z profiles
    scalar_t* __restrict__ vxh, scalar_t* __restrict__ vzh,  // hist slot it+1
    const int it, const int B,
    const int H, const int nz, const int nx,
    const scalar_t dt, const scalar_t inv_dx, const scalar_t inv_dz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (x >= nx || z >= nz) return;
    const long plane = (long)nz * nx;
    const long base = (long)b * plane;
    const long i = base + (long)z * nx + x;

    // D+ along x on p: sum_k c_k ( p[x+k] - p[x-k+1] )
    scalar_t dpx = 0;
    for (int k = 1; k <= H; ++k) {
        const int xp = x + k, xm = x - k + 1;
        const scalar_t fp = (xp < nx) ? p[base + (long)z * nx + xp] : scalar_t(0);
        const scalar_t fm = (xm >= 0) ? p[base + (long)z * nx + xm] : scalar_t(0);
        dpx += fd[k - 1] * (fp - fm);
    }
    dpx *= inv_dx;
    // D+ along z on p
    scalar_t dpz = 0;
    for (int k = 1; k <= H; ++k) {
        const int zp = z + k, zm = z - k + 1;
        const scalar_t fp = (zp < nz) ? p[base + (long)zp * nx + x] : scalar_t(0);
        const scalar_t fm = (zm >= 0) ? p[base + (long)zm * nx + x] : scalar_t(0);
        dpz += fd[k - 1] * (fp - fm);
    }
    dpz *= inv_dz;

    // CPML stretch (half profiles)
    const scalar_t qx_new = bxh[x] * psi_px[i] + axh[x] * dpx;
    const scalar_t qz_new = bzh[z] * psi_pz[i] + azh[z] * dpz;
    psi_px[i] = qx_new;
    psi_pz[i] = qz_new;
    const scalar_t tx = dpx / kxh[x] + qx_new;
    const scalar_t tz = dpz / kzh[z] + qz_new;

    const scalar_t bmul = dt * buoy_m[(long)z * nx + x];
    const scalar_t vx_new = vx[i] - bmul * tx;
    const scalar_t vz_new = vz[i] - bmul * tz;
    vx[i] = vx_new;
    vz[i] = vz_new;
    if (vxh != nullptr) {
        const long o = (long)(it + 1) * B * plane + i;
        vxh[o] = vx_new;
        vzh[o] = vz_new;
    }
}

template <typename scalar_t>
__global__ void source_kernel(
    scalar_t* __restrict__ p,
    const int64_t* __restrict__ src_z, const int64_t* __restrict__ src_x,
    const scalar_t* __restrict__ wav,   // (B, nt)
    scalar_t* __restrict__ ph,          // hist slot it+1: overwrite src cells
    const int it, const int nt, const int B, const int nz, const int nx,
    const scalar_t dt, const int n_src)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_src) return;
    const long plane = (long)nz * nx;
    const long i = (long)b * plane + (long)src_z[b] * nx + src_x[b];
    const scalar_t p_new = p[i] + dt * wav[(long)b * nt + it];
    p[i] = p_new;
    if (ph != nullptr) ph[(long)(it + 1) * B * plane + i] = p_new;
}

template <typename scalar_t>
__global__ void record_kernel(
    const scalar_t* __restrict__ p,
    const int64_t* __restrict__ rcv_z, const int64_t* __restrict__ rcv_x,
    scalar_t* __restrict__ out,         // (B, nt, n_rcv[, C]) — slot (comp of n_comp)
    const int it, const int nt, const int n_rcv,
    const int nz, const int nx, const int n_comp, const int comp)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (r >= n_rcv) return;
    const long i = (long)b * nz * nx + (long)rcv_z[r] * nx + rcv_x[r];
    out[(((long)b * nt + it) * n_rcv + r) * n_comp + comp] = p[i];
}

// (history stores are fused into pressure/velocity/source kernels above)

// ------------------------- adjoint (stage C) -------------------------------
// Reverse-time recurrence. Transposes of the zero-filled staggered operators:
//   (D+)^T = -D-   and   (D-)^T = -D+   (elementwise exact, boundaries included).
// The CPML adjoint needs only the (b, a, kappa) PROFILES (the recurrence is
// linear); coefficient gradients use the value tricks
//   gK += pbar' * (p' - p_it) / K       (= -dt * pbar' * (sx+sz))
//   gB += Vbar . (v_{it+1} - v_it) / B  (= -dt * Vbar . t)
// so only the p/vx/vz histories are stored — never the psi history.

template <typename scalar_t>
__global__ void adj_record_scatter_kernel(
    scalar_t* __restrict__ pbar,
    const scalar_t* __restrict__ gout,  // (B, nt, n_rcv[, C]) — slot (comp of n_comp)
    const int64_t* __restrict__ rcv_z, const int64_t* __restrict__ rcv_x,
    const int it, const int nt, const int n_rcv, const int nz, const int nx,
    const int n_comp, const int comp)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (r >= n_rcv) return;
    const long i = (long)b * nz * nx + (long)rcv_z[r] * nx + rcv_x[r];
    atomicAdd(&pbar[i], gout[(((long)b * nt + it) * n_rcv + r) * n_comp + comp]);
}

template <typename scalar_t>
__global__ void adj_source_kernel(
    const scalar_t* __restrict__ pbar,
    const int64_t* __restrict__ src_z, const int64_t* __restrict__ src_x,
    scalar_t* __restrict__ gwav,        // (B, nt)
    const int it, const int nt, const int nz, const int nx,
    const scalar_t dt, const int n_src)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_src) return;
    const long i = (long)b * nz * nx + (long)src_z[b] * nx + src_x[b];
    gwav[(long)b * nt + it] = dt * pbar[i];
}

// k3a: velocity-stage reverse (pointwise): dbar_p* scratch, psi-threads, gB
template <typename scalar_t>
__global__ void adj_velocity_point_kernel(
    const scalar_t* __restrict__ vxbar, const scalar_t* __restrict__ vzbar,
    scalar_t* __restrict__ psibar_px, scalar_t* __restrict__ psibar_pz,
    scalar_t* __restrict__ dbar_px, scalar_t* __restrict__ dbar_pz,
    scalar_t* __restrict__ gB,
    const scalar_t* __restrict__ buoy_m,
    const scalar_t* __restrict__ vxh, const scalar_t* __restrict__ vzh,
    const scalar_t* __restrict__ bxh, const scalar_t* __restrict__ axh,
    const scalar_t* __restrict__ kxh,
    const scalar_t* __restrict__ bzh, const scalar_t* __restrict__ azh,
    const scalar_t* __restrict__ kzh,
    const int it, const int B, const int nz, const int nx, const scalar_t dt)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (x >= nx || z >= nz) return;
    const long plane = (long)nz * nx;
    const long i = (long)b * plane + (long)z * nx + x;
    const long m = (long)z * nx + x;
    const long h1 = (long)(it + 1) * B * plane + i;
    const long h0 = (long)it * B * plane + i;

    const scalar_t vbx = vxbar[i], vbz = vzbar[i];
    const scalar_t txbar = -dt * buoy_m[m] * vbx;
    const scalar_t tzbar = -dt * buoy_m[m] * vbz;
    // gB via the value trick: -dt*(tx*vbx + tz*vbz) = (dvx*vbx + dvz*vbz)/B
    const scalar_t gb = (vbx * (vxh[h1] - vxh[h0]) + vbz * (vzh[h1] - vzh[h0]))
                        / buoy_m[m];
    atomicAdd(&gB[m], gb);

    const scalar_t Px = psibar_px[i] + txbar;   // total cotangent of psi_px^{it+1}
    const scalar_t Pz = psibar_pz[i] + tzbar;
    dbar_px[i] = txbar / kxh[x] + Px * axh[x];
    dbar_pz[i] = tzbar / kzh[z] + Pz * azh[z];
    psibar_px[i] = Px * bxh[x];                 // thread to step it-1
    psibar_pz[i] = Pz * bzh[z];
}

// k3b+k4a fused: pbar' += (D+)^T dbar = -D- on the velocity-stage scratch,
// then the pressure-stage pointwise reverse (gK, psi-threads, dbar_v scratch)
// on the freshly updated pbar[i] — the point stage only touches its OWN cell,
// so the two stages chain in-register. The velocity-stage scratch (dbar_p*)
// is read and the pressure-stage scratch (dbar_v*) written to DIFFERENT
// buffers, so there is no write-after-read hazard across blocks.
template <typename scalar_t>
__global__ void adj_pressure_fused_kernel(
    scalar_t* __restrict__ pbar,
    const scalar_t* __restrict__ dbar_px, const scalar_t* __restrict__ dbar_pz,
    scalar_t* __restrict__ psibar_vxx, scalar_t* __restrict__ psibar_vzz,
    scalar_t* __restrict__ dbar_vx, scalar_t* __restrict__ dbar_vz,
    scalar_t* __restrict__ gK,
    const scalar_t* __restrict__ kappa_m,
    const scalar_t* __restrict__ ph,
    const scalar_t* __restrict__ fd,
    const scalar_t* __restrict__ bx, const scalar_t* __restrict__ ax,
    const scalar_t* __restrict__ kx,
    const scalar_t* __restrict__ bz, const scalar_t* __restrict__ az,
    const scalar_t* __restrict__ kz,
    const int it, const int B, const int H, const int nz, const int nx,
    const scalar_t dt, const scalar_t inv_dx, const scalar_t inv_dz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (x >= nx || z >= nz) return;
    const long plane = (long)nz * nx;
    const long base = (long)b * plane;
    const long i = base + (long)z * nx + x;
    const long m = (long)z * nx + x;
    const long h1 = (long)(it + 1) * B * plane + i;
    const long h0 = (long)it * B * plane + i;

    scalar_t accx = 0;   // D-_x dbar_px
    for (int k = 1; k <= H; ++k) {
        const int xp = x + k - 1, xm = x - k;
        const scalar_t fp = (xp < nx) ? dbar_px[base + (long)z * nx + xp] : scalar_t(0);
        const scalar_t fm = (xm >= 0) ? dbar_px[base + (long)z * nx + xm] : scalar_t(0);
        accx += fd[k - 1] * (fp - fm);
    }
    scalar_t accz = 0;   // D-_z dbar_pz
    for (int k = 1; k <= H; ++k) {
        const int zp = z + k - 1, zm = z - k;
        const scalar_t fp = (zp < nz) ? dbar_pz[base + (long)zp * nx + x] : scalar_t(0);
        const scalar_t fm = (zm >= 0) ? dbar_pz[base + (long)zm * nx + x] : scalar_t(0);
        accz += fd[k - 1] * (fp - fm);
    }
    const scalar_t pb = pbar[i] - (accx * inv_dx + accz * inv_dz);
    pbar[i] = pb;

    // gK via the value trick with the UNCORRECTED p diff (p_hist[it+1] includes
    // the source; a tiny follow-up kernel subtracts the source term at src cells)
    atomicAdd(&gK[m], pb * (ph[h1] - ph[h0]) / kappa_m[m]);

    const scalar_t sbar = -dt * kappa_m[m] * pb;   // s̄x == s̄z
    const scalar_t Fx = psibar_vxx[i] + sbar;
    const scalar_t Fz = psibar_vzz[i] + sbar;
    dbar_vx[i] = sbar / kx[x] + Fx * ax[x];
    dbar_vz[i] = sbar / kz[z] + Fz * az[z];
    psibar_vxx[i] = Fx * bx[x];
    psibar_vzz[i] = Fz * bz[z];
}

// k4b: vbar -= (D-)^T dbar = +D+ applied to the dbar scratch fields
//      (vbar_new = vbar - D+ dbar, since (D-)^T = -D+)
template <typename scalar_t>
__global__ void adj_velocity_stencil_kernel(
    scalar_t* __restrict__ vxbar, scalar_t* __restrict__ vzbar,
    const scalar_t* __restrict__ dbar_vx, const scalar_t* __restrict__ dbar_vz,
    const scalar_t* __restrict__ fd,
    const int H, const int nz, const int nx,
    const scalar_t inv_dx, const scalar_t inv_dz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (x >= nx || z >= nz) return;
    const long base = (long)b * nz * nx;
    const long i = base + (long)z * nx + x;

    scalar_t accx = 0;   // D+_x dbar_vx
    for (int k = 1; k <= H; ++k) {
        const int xp = x + k, xm = x - k + 1;
        const scalar_t fp = (xp < nx) ? dbar_vx[base + (long)z * nx + xp] : scalar_t(0);
        const scalar_t fm = (xm >= 0) ? dbar_vx[base + (long)z * nx + xm] : scalar_t(0);
        accx += fd[k - 1] * (fp - fm);
    }
    scalar_t accz = 0;   // D+_z dbar_vz
    for (int k = 1; k <= H; ++k) {
        const int zp = z + k, zm = z - k + 1;
        const scalar_t fp = (zp < nz) ? dbar_vz[base + (long)zp * nx + x] : scalar_t(0);
        const scalar_t fm = (zm >= 0) ? dbar_vz[base + (long)zm * nx + x] : scalar_t(0);
        accz += fd[k - 1] * (fp - fm);
    }
    vxbar[i] -= accx * inv_dx;
    vzbar[i] -= accz * inv_dz;
}

// k4c: subtract the source term from gK at the source cells:
//   p' = p_hist[it+1] - dt*wav  =>  gK -= pbar * dt * wav / K   (at src cells)
template <typename scalar_t>
__global__ void adj_ksrc_correct_kernel(
    const scalar_t* __restrict__ pbar,
    scalar_t* __restrict__ gK,
    const scalar_t* __restrict__ kappa_m,
    const int64_t* __restrict__ src_z, const int64_t* __restrict__ src_x,
    const scalar_t* __restrict__ wav,
    const int it, const int nt, const int nz, const int nx,
    const scalar_t dt, const int n_src)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_src) return;
    const long m = (long)src_z[b] * nx + src_x[b];
    const long i = (long)b * nz * nx + m;
    atomicAdd(&gK[m], -pbar[i] * dt * wav[(long)b * nt + it] / kappa_m[m]);
}

}  // namespace

void acoustic2d_forward(
    torch::Tensor p, torch::Tensor vx, torch::Tensor vz,
    torch::Tensor psi_vxx, torch::Tensor psi_vzz,
    torch::Tensor psi_px, torch::Tensor psi_pz,
    torch::Tensor kappa, torch::Tensor buoyancy,
    torch::Tensor fd,
    torch::Tensor bx_int, torch::Tensor ax_int, torch::Tensor kx_int,
    torch::Tensor bz_int, torch::Tensor az_int, torch::Tensor kz_int,
    torch::Tensor bx_half, torch::Tensor ax_half, torch::Tensor kx_half,
    torch::Tensor bz_half, torch::Tensor az_half, torch::Tensor kz_half,
    torch::Tensor src_z, torch::Tensor src_x, torch::Tensor wav,
    torch::Tensor rcv_z, torch::Tensor rcv_x, torch::Tensor out,
    double dt, double dx, double dz,
    torch::Tensor p_hist, torch::Tensor vx_hist, torch::Tensor vz_hist,
    int64_t n_comp)
{
    TORCH_CHECK(p.is_cuda() && p.is_contiguous(), "p must be contiguous CUDA");
    const int B = p.size(0);
    const int nz = p.size(2);
    const int nx = p.size(3);
    const int nt = wav.size(1);
    const int n_src = src_z.size(0);
    const int n_rcv = rcv_z.size(0);
    const int H = fd.size(0);
    TORCH_CHECK(H >= 1 && H <= MAX_H, "fd half-order out of range");
    TORCH_CHECK(n_comp == 1 || n_comp == 3, "n_comp must be 1 (p) or 3 (p,vx,vz)");
    const int C = (int)n_comp;
    const bool with_hist = p_hist.numel() > 0;

    const dim3 threads(32, 8, 1);
    const dim3 grid((nx + 31) / 32, (nz + 7) / 8, B);
    const int src_threads = 128;
    const int src_blocks = (n_src + src_threads - 1) / src_threads;
    const dim3 rec_threads(128, 1, 1);
    const dim3 rec_grid((n_rcv + 127) / 128, B, 1);

    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "acoustic2d_forward", [&] {
        scalar_t* p_ = p.data_ptr<scalar_t>();
        scalar_t* vx_ = vx.data_ptr<scalar_t>();
        scalar_t* vz_ = vz.data_ptr<scalar_t>();
        scalar_t* pvxx = psi_vxx.data_ptr<scalar_t>();
        scalar_t* pvzz = psi_vzz.data_ptr<scalar_t>();
        scalar_t* ppx = psi_px.data_ptr<scalar_t>();
        scalar_t* ppz = psi_pz.data_ptr<scalar_t>();
        const scalar_t* kap = kappa.data_ptr<scalar_t>();
        const scalar_t* buo = buoyancy.data_ptr<scalar_t>();
        const scalar_t* fd_ = fd.data_ptr<scalar_t>();
        scalar_t* ph_ = with_hist ? p_hist.data_ptr<scalar_t>() : nullptr;
        scalar_t* vxh_ = with_hist ? vx_hist.data_ptr<scalar_t>() : nullptr;
        scalar_t* vzh_ = with_hist ? vz_hist.data_ptr<scalar_t>() : nullptr;
        const scalar_t dt_ = static_cast<scalar_t>(dt);
        const scalar_t idx_ = static_cast<scalar_t>(1.0 / dx);
        const scalar_t idz_ = static_cast<scalar_t>(1.0 / dz);
        for (int it = 0; it < nt; ++it) {
            pressure_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                p_, vx_, vz_, pvxx, pvzz, kap, fd_,
                bx_int.data_ptr<scalar_t>(), ax_int.data_ptr<scalar_t>(),
                kx_int.data_ptr<scalar_t>(),
                bz_int.data_ptr<scalar_t>(), az_int.data_ptr<scalar_t>(),
                kz_int.data_ptr<scalar_t>(),
                ph_, it, B, H, nz, nx, dt_, idx_, idz_);
            velocity_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                p_, vx_, vz_, ppx, ppz, buo, fd_,
                bx_half.data_ptr<scalar_t>(), ax_half.data_ptr<scalar_t>(),
                kx_half.data_ptr<scalar_t>(),
                bz_half.data_ptr<scalar_t>(), az_half.data_ptr<scalar_t>(),
                kz_half.data_ptr<scalar_t>(),
                vxh_, vzh_, it, B, H, nz, nx, dt_, idx_, idz_);
            source_kernel<scalar_t><<<src_blocks, src_threads, 0, stream>>>(
                p_, src_z.data_ptr<int64_t>(), src_x.data_ptr<int64_t>(),
                wav.data_ptr<scalar_t>(), ph_, it, nt, B, nz, nx, dt_, n_src);
            record_kernel<scalar_t><<<rec_grid, rec_threads, 0, stream>>>(
                p_, rcv_z.data_ptr<int64_t>(), rcv_x.data_ptr<int64_t>(),
                out.data_ptr<scalar_t>(), it, nt, n_rcv, nz, nx, C, 0);
            if (C == 3) {
                record_kernel<scalar_t><<<rec_grid, rec_threads, 0, stream>>>(
                    vx_, rcv_z.data_ptr<int64_t>(), rcv_x.data_ptr<int64_t>(),
                    out.data_ptr<scalar_t>(), it, nt, n_rcv, nz, nx, C, 1);
                record_kernel<scalar_t><<<rec_grid, rec_threads, 0, stream>>>(
                    vz_, rcv_z.data_ptr<int64_t>(), rcv_x.data_ptr<int64_t>(),
                    out.data_ptr<scalar_t>(), it, nt, n_rcv, nz, nx, C, 2);
            }
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void acoustic2d_backward(
    torch::Tensor grad_out,                       // (B, nt, n_rcv)
    torch::Tensor p_hist, torch::Tensor vx_hist,  // (nt+1, B, nz, nx)
    torch::Tensor vz_hist,
    torch::Tensor kappa, torch::Tensor buoyancy,
    torch::Tensor fd,
    torch::Tensor bx_int, torch::Tensor ax_int, torch::Tensor kx_int,
    torch::Tensor bz_int, torch::Tensor az_int, torch::Tensor kz_int,
    torch::Tensor bx_half, torch::Tensor ax_half, torch::Tensor kx_half,
    torch::Tensor bz_half, torch::Tensor az_half, torch::Tensor kz_half,
    torch::Tensor src_z, torch::Tensor src_x, torch::Tensor wav,
    torch::Tensor rcv_z, torch::Tensor rcv_x,
    // adjoint state (zero-initialised, in place):
    torch::Tensor pbar, torch::Tensor vxbar, torch::Tensor vzbar,
    torch::Tensor psibar_vxx, torch::Tensor psibar_vzz,
    torch::Tensor psibar_px, torch::Tensor psibar_pz,
    // scratch (B,1,nz,nx) x4: a/b = velocity-stage dbar_p*, c/d = pressure-
    // stage dbar_v* (separate buffers so the fused pressure kernel can read
    // a/b while writing c/d without a cross-block hazard)
    torch::Tensor dbar_a, torch::Tensor dbar_b,
    torch::Tensor dbar_c, torch::Tensor dbar_d,
    // outputs (zero-initialised):
    torch::Tensor gK, torch::Tensor gB, torch::Tensor gwav,
    double dt, double dx, double dz, int64_t n_comp)
{
    const int C = (int)n_comp;
    const int B = pbar.size(0);
    const int nz = pbar.size(2);
    const int nx = pbar.size(3);
    const int nt = wav.size(1);
    const int n_src = src_z.size(0);
    const int n_rcv = rcv_z.size(0);
    const int H = fd.size(0);

    const dim3 threads(32, 8, 1);
    const dim3 grid((nx + 31) / 32, (nz + 7) / 8, B);
    const int src_threads = 128;
    const int src_blocks = (n_src + src_threads - 1) / src_threads;
    const dim3 rec_threads(128, 1, 1);
    const dim3 rec_grid((n_rcv + 127) / 128, B, 1);

    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(pbar.scalar_type(), "acoustic2d_backward", [&] {
        const scalar_t dt_ = static_cast<scalar_t>(dt);
        const scalar_t idx_ = static_cast<scalar_t>(1.0 / dx);
        const scalar_t idz_ = static_cast<scalar_t>(1.0 / dz);
        for (int it = nt - 1; it >= 0; --it) {
            adj_record_scatter_kernel<scalar_t><<<rec_grid, rec_threads, 0, stream>>>(
                pbar.data_ptr<scalar_t>(), grad_out.data_ptr<scalar_t>(),
                rcv_z.data_ptr<int64_t>(), rcv_x.data_ptr<int64_t>(),
                it, nt, n_rcv, nz, nx, C, 0);
            if (C == 3) {
                adj_record_scatter_kernel<scalar_t><<<rec_grid, rec_threads, 0, stream>>>(
                    vxbar.data_ptr<scalar_t>(), grad_out.data_ptr<scalar_t>(),
                    rcv_z.data_ptr<int64_t>(), rcv_x.data_ptr<int64_t>(),
                    it, nt, n_rcv, nz, nx, C, 1);
                adj_record_scatter_kernel<scalar_t><<<rec_grid, rec_threads, 0, stream>>>(
                    vzbar.data_ptr<scalar_t>(), grad_out.data_ptr<scalar_t>(),
                    rcv_z.data_ptr<int64_t>(), rcv_x.data_ptr<int64_t>(),
                    it, nt, n_rcv, nz, nx, C, 2);
            }
            adj_source_kernel<scalar_t><<<src_blocks, src_threads, 0, stream>>>(
                pbar.data_ptr<scalar_t>(),
                src_z.data_ptr<int64_t>(), src_x.data_ptr<int64_t>(),
                gwav.data_ptr<scalar_t>(), it, nt, nz, nx, dt_, n_src);
            adj_velocity_point_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                vxbar.data_ptr<scalar_t>(), vzbar.data_ptr<scalar_t>(),
                psibar_px.data_ptr<scalar_t>(), psibar_pz.data_ptr<scalar_t>(),
                dbar_a.data_ptr<scalar_t>(), dbar_b.data_ptr<scalar_t>(),
                gB.data_ptr<scalar_t>(), buoyancy.data_ptr<scalar_t>(),
                vx_hist.data_ptr<scalar_t>(), vz_hist.data_ptr<scalar_t>(),
                bx_half.data_ptr<scalar_t>(), ax_half.data_ptr<scalar_t>(),
                kx_half.data_ptr<scalar_t>(),
                bz_half.data_ptr<scalar_t>(), az_half.data_ptr<scalar_t>(),
                kz_half.data_ptr<scalar_t>(),
                it, B, nz, nx, dt_);
            adj_pressure_fused_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                pbar.data_ptr<scalar_t>(),
                dbar_a.data_ptr<scalar_t>(), dbar_b.data_ptr<scalar_t>(),
                psibar_vxx.data_ptr<scalar_t>(), psibar_vzz.data_ptr<scalar_t>(),
                dbar_c.data_ptr<scalar_t>(), dbar_d.data_ptr<scalar_t>(),
                gK.data_ptr<scalar_t>(), kappa.data_ptr<scalar_t>(),
                p_hist.data_ptr<scalar_t>(), fd.data_ptr<scalar_t>(),
                bx_int.data_ptr<scalar_t>(), ax_int.data_ptr<scalar_t>(),
                kx_int.data_ptr<scalar_t>(),
                bz_int.data_ptr<scalar_t>(), az_int.data_ptr<scalar_t>(),
                kz_int.data_ptr<scalar_t>(),
                it, B, H, nz, nx, dt_, idx_, idz_);
            adj_ksrc_correct_kernel<scalar_t><<<src_blocks, src_threads, 0, stream>>>(
                pbar.data_ptr<scalar_t>(), gK.data_ptr<scalar_t>(),
                kappa.data_ptr<scalar_t>(),
                src_z.data_ptr<int64_t>(), src_x.data_ptr<int64_t>(),
                wav.data_ptr<scalar_t>(), it, nt, nz, nx, dt_, n_src);
            adj_velocity_stencil_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                vxbar.data_ptr<scalar_t>(), vzbar.data_ptr<scalar_t>(),
                dbar_c.data_ptr<scalar_t>(), dbar_d.data_ptr<scalar_t>(),
                fd.data_ptr<scalar_t>(), H, nz, nx, idx_, idz_);
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &acoustic2d_forward,
          "acoustic 2D velocity-stress forward, full time loop in C++ (CUDA)");
    m.def("backward", &acoustic2d_backward,
          "acoustic 2D adjoint-state backward, full reverse loop in C++ (CUDA)");
}
