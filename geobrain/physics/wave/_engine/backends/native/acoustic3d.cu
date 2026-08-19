// GeoBrain G6 stage-D: native CUDA forward + adjoint for the 3-D acoustic equation.
// Systematic axis-extension of acoustic2d.cu (verified stage B/C): same
// P->V->source->record order, same zero-fill D+/D- stencils per axis, same CPML
// recursion, same adjoint identities ((D+)^T=-D-, (D-)^T=-D+; psi adjoint needs
// only the (b,a,kappa) profiles; gK/gB via value tricks on p/vx/vy/vz histories).
// Layout: fields (B,1,nz,ny,nx) contiguous; i = ((b*nz+z)*ny+y)*nx + x.
// Grid mapping: blockIdx.z spans B*nz_tiles (b = blockIdx.z / ztiles).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace {

constexpr int MAX_H = 8;

#define IDX3(b, z, y, x) ((((long)(b) * nz + (z)) * ny + (y)) * nx + (x))
#define MIDX3(z, y, x) (((long)(z) * ny + (y)) * nx + (x))

template <typename scalar_t>
__global__ void pressure3d_kernel(
    scalar_t* __restrict__ p,
    const scalar_t* __restrict__ vx, const scalar_t* __restrict__ vy,
    const scalar_t* __restrict__ vz,
    scalar_t* __restrict__ psi_vxx, scalar_t* __restrict__ psi_vyy,
    scalar_t* __restrict__ psi_vzz,
    const scalar_t* __restrict__ kappa_m, const scalar_t* __restrict__ fd,
    const scalar_t* __restrict__ bx, const scalar_t* __restrict__ ax, const scalar_t* __restrict__ kx,
    const scalar_t* __restrict__ by, const scalar_t* __restrict__ ay, const scalar_t* __restrict__ ky,
    const scalar_t* __restrict__ bz, const scalar_t* __restrict__ az, const scalar_t* __restrict__ kz,
    scalar_t* __restrict__ ph,           // hist slot it+1 (nullptr = off);
                                         // pre-source, source3d overwrites src
    const int it, const long ncell,
    const int H, const int nz, const int ny, const int nx, const int ztiles,
    const scalar_t dt, const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int zt = blockIdx.z % ztiles, b = blockIdx.z / ztiles;
    const int z = zt * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const long i = IDX3(b, z, y, x);

    scalar_t dvx = 0, dvy = 0, dvz = 0;  // D- per axis
    for (int k = 1; k <= H; ++k) {
        const int xp = x + k - 1, xm = x - k;
        dvx += fd[k-1] * (((xp < nx) ? vx[IDX3(b,z,y,xp)] : scalar_t(0))
                        - ((xm >= 0) ? vx[IDX3(b,z,y,xm)] : scalar_t(0)));
        const int yp = y + k - 1, ym = y - k;
        dvy += fd[k-1] * (((yp < ny) ? vy[IDX3(b,z,yp,x)] : scalar_t(0))
                        - ((ym >= 0) ? vy[IDX3(b,z,ym,x)] : scalar_t(0)));
        const int zp = z + k - 1, zm = z - k;
        dvz += fd[k-1] * (((zp < nz) ? vz[IDX3(b,zp,y,x)] : scalar_t(0))
                        - ((zm >= 0) ? vz[IDX3(b,zm,y,x)] : scalar_t(0)));
    }
    dvx *= inv_dx; dvy *= inv_dy; dvz *= inv_dz;

    const scalar_t px = bx[x] * psi_vxx[i] + ax[x] * dvx;
    const scalar_t py = by[y] * psi_vyy[i] + ay[y] * dvy;
    const scalar_t pz = bz[z] * psi_vzz[i] + az[z] * dvz;
    psi_vxx[i] = px; psi_vyy[i] = py; psi_vzz[i] = pz;
    const scalar_t p_new = p[i] - dt * kappa_m[MIDX3(z,y,x)]
          * (dvx / kx[x] + px + dvy / ky[y] + py + dvz / kz[z] + pz);
    p[i] = p_new;
    if (ph != nullptr) ph[(long)(it + 1) * ncell + i] = p_new;
}

template <typename scalar_t>
__global__ void velocity3d_kernel(
    const scalar_t* __restrict__ p,
    scalar_t* __restrict__ vx, scalar_t* __restrict__ vy, scalar_t* __restrict__ vz,
    scalar_t* __restrict__ psi_px, scalar_t* __restrict__ psi_py,
    scalar_t* __restrict__ psi_pz,
    const scalar_t* __restrict__ buoy_m, const scalar_t* __restrict__ fd,
    const scalar_t* __restrict__ bxh, const scalar_t* __restrict__ axh, const scalar_t* __restrict__ kxh,
    const scalar_t* __restrict__ byh, const scalar_t* __restrict__ ayh, const scalar_t* __restrict__ kyh,
    const scalar_t* __restrict__ bzh, const scalar_t* __restrict__ azh, const scalar_t* __restrict__ kzh,
    scalar_t* __restrict__ vxh, scalar_t* __restrict__ vyh,
    scalar_t* __restrict__ vzh,          // hist slot it+1 (nullptr = off)
    const int it, const long ncell,
    const int H, const int nz, const int ny, const int nx, const int ztiles,
    const scalar_t dt, const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int zt = blockIdx.z % ztiles, b = blockIdx.z / ztiles;
    const int z = zt * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const long i = IDX3(b, z, y, x);

    scalar_t dpx = 0, dpy = 0, dpz = 0;  // D+ per axis
    for (int k = 1; k <= H; ++k) {
        const int xp = x + k, xm = x - k + 1;
        dpx += fd[k-1] * (((xp < nx) ? p[IDX3(b,z,y,xp)] : scalar_t(0))
                        - ((xm >= 0) ? p[IDX3(b,z,y,xm)] : scalar_t(0)));
        const int yp = y + k, ym = y - k + 1;
        dpy += fd[k-1] * (((yp < ny) ? p[IDX3(b,z,yp,x)] : scalar_t(0))
                        - ((ym >= 0) ? p[IDX3(b,z,ym,x)] : scalar_t(0)));
        const int zp = z + k, zm = z - k + 1;
        dpz += fd[k-1] * (((zp < nz) ? p[IDX3(b,zp,y,x)] : scalar_t(0))
                        - ((zm >= 0) ? p[IDX3(b,zm,y,x)] : scalar_t(0)));
    }
    dpx *= inv_dx; dpy *= inv_dy; dpz *= inv_dz;

    const scalar_t qx = bxh[x] * psi_px[i] + axh[x] * dpx;
    const scalar_t qy = byh[y] * psi_py[i] + ayh[y] * dpy;
    const scalar_t qz = bzh[z] * psi_pz[i] + azh[z] * dpz;
    psi_px[i] = qx; psi_py[i] = qy; psi_pz[i] = qz;
    const scalar_t bmul = dt * buoy_m[MIDX3(z,y,x)];
    const scalar_t vx_new = vx[i] - bmul * (dpx / kxh[x] + qx);
    const scalar_t vy_new = vy[i] - bmul * (dpy / kyh[y] + qy);
    const scalar_t vz_new = vz[i] - bmul * (dpz / kzh[z] + qz);
    vx[i] = vx_new; vy[i] = vy_new; vz[i] = vz_new;
    if (vxh != nullptr) {
        const long o = (long)(it + 1) * ncell + i;
        vxh[o] = vx_new; vyh[o] = vy_new; vzh[o] = vz_new;
    }
}

template <typename scalar_t>
__global__ void source3d_kernel(
    scalar_t* __restrict__ p,
    const int64_t* __restrict__ sz, const int64_t* __restrict__ sy,
    const int64_t* __restrict__ sx,
    const scalar_t* __restrict__ wav,
    scalar_t* __restrict__ ph,          // hist slot it+1: overwrite src cells
    const int it, const int nt, const long ncell,
    const int nz, const int ny, const int nx,
    const scalar_t dt, const int n_src)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_src) return;
    const long i = IDX3(b, sz[b], sy[b], sx[b]);
    const scalar_t p_new = p[i] + dt * wav[(long)b * nt + it];
    p[i] = p_new;
    if (ph != nullptr) ph[(long)(it + 1) * ncell + i] = p_new;
}

template <typename scalar_t>
__global__ void record3d_kernel(
    const scalar_t* __restrict__ p,
    const int64_t* __restrict__ rz, const int64_t* __restrict__ ry,
    const int64_t* __restrict__ rx,
    scalar_t* __restrict__ out,
    const int it, const int nt, const int n_rcv,
    const int nz, const int ny, const int nx)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (r >= n_rcv) return;
    out[((long)b * nt + it) * n_rcv + r] = p[IDX3(b, rz[r], ry[r], rx[r])];
}

// (history stores are fused into pressure/velocity/source kernels above)

// ------------------------------ adjoint ------------------------------------

template <typename scalar_t>
__global__ void adj3d_record_scatter_kernel(
    scalar_t* __restrict__ pbar, const scalar_t* __restrict__ gout,
    const int64_t* __restrict__ rz, const int64_t* __restrict__ ry,
    const int64_t* __restrict__ rx,
    const int it, const int nt, const int n_rcv,
    const int nz, const int ny, const int nx)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (r >= n_rcv) return;
    atomicAdd(&pbar[IDX3(b, rz[r], ry[r], rx[r])],
              gout[((long)b * nt + it) * n_rcv + r]);
}

template <typename scalar_t>
__global__ void adj3d_source_kernel(
    const scalar_t* __restrict__ pbar,
    const int64_t* __restrict__ sz, const int64_t* __restrict__ sy,
    const int64_t* __restrict__ sx,
    scalar_t* __restrict__ gwav,
    const int it, const int nt, const int nz, const int ny, const int nx,
    const scalar_t dt, const int n_src)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_src) return;
    gwav[(long)b * nt + it] = dt * pbar[IDX3(b, sz[b], sy[b], sx[b])];
}

template <typename scalar_t>
__global__ void adj3d_velocity_point_kernel(
    const scalar_t* __restrict__ vxbar, const scalar_t* __restrict__ vybar,
    const scalar_t* __restrict__ vzbar,
    scalar_t* __restrict__ psibar_px, scalar_t* __restrict__ psibar_py,
    scalar_t* __restrict__ psibar_pz,
    scalar_t* __restrict__ dbx, scalar_t* __restrict__ dby, scalar_t* __restrict__ dbz,
    scalar_t* __restrict__ gB,
    const scalar_t* __restrict__ buoy_m,
    const scalar_t* __restrict__ vxh, const scalar_t* __restrict__ vyh,
    const scalar_t* __restrict__ vzh,
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
    const scalar_t txb = -dt * buoy_m[m] * vbx;
    const scalar_t tyb = -dt * buoy_m[m] * vby;
    const scalar_t tzb = -dt * buoy_m[m] * vbz;
    const scalar_t Px = psibar_px[i] + txb;
    const scalar_t Py = psibar_py[i] + tyb;
    const scalar_t Pz = psibar_pz[i] + tzb;
    dbx[i] = txb / kxh[x] + Px * axh[x];
    dby[i] = tyb / kyh[y] + Py * ayh[y];
    dbz[i] = tzb / kzh[z] + Pz * azh[z];
    psibar_px[i] = Px * bxh[x];
    psibar_py[i] = Py * byh[y];
    psibar_pz[i] = Pz * bzh[z];
}

// stencil+point fused: pbar update reads the velocity-stage scratch (dbx_in..)
// and the pointwise stage writes the pressure-stage scratch (dbx_out..) —
// separate buffers, so no cross-block write-after-read hazard; the point
// stage only needs its OWN updated pbar[i], which stays in-register.
template <typename scalar_t>
__global__ void adj3d_pressure_fused_kernel(
    scalar_t* __restrict__ pbar,
    const scalar_t* __restrict__ dbx_in, const scalar_t* __restrict__ dby_in,
    const scalar_t* __restrict__ dbz_in,
    scalar_t* __restrict__ psibar_vxx, scalar_t* __restrict__ psibar_vyy,
    scalar_t* __restrict__ psibar_vzz,
    scalar_t* __restrict__ dbx_out, scalar_t* __restrict__ dby_out,
    scalar_t* __restrict__ dbz_out,
    scalar_t* __restrict__ gK,
    const scalar_t* __restrict__ kappa_m, const scalar_t* __restrict__ ph,
    const scalar_t* __restrict__ fd,
    const scalar_t* __restrict__ bx, const scalar_t* __restrict__ ax_, const scalar_t* __restrict__ kx,
    const scalar_t* __restrict__ by, const scalar_t* __restrict__ ay_, const scalar_t* __restrict__ ky,
    const scalar_t* __restrict__ bz, const scalar_t* __restrict__ az_, const scalar_t* __restrict__ kz,
    const int it, const long ncell,
    const int H, const int nz, const int ny, const int nx, const int ztiles,
    const scalar_t dt,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz)
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

    scalar_t ax = 0, ay = 0, az = 0;   // D- per axis on the dbar_in fields
    for (int k = 1; k <= H; ++k) {
        const int xp = x + k - 1, xm = x - k;
        ax += fd[k-1] * (((xp < nx) ? dbx_in[IDX3(b,z,y,xp)] : scalar_t(0))
                       - ((xm >= 0) ? dbx_in[IDX3(b,z,y,xm)] : scalar_t(0)));
        const int yp = y + k - 1, ym = y - k;
        ay += fd[k-1] * (((yp < ny) ? dby_in[IDX3(b,z,yp,x)] : scalar_t(0))
                       - ((ym >= 0) ? dby_in[IDX3(b,z,ym,x)] : scalar_t(0)));
        const int zp = z + k - 1, zm = z - k;
        az += fd[k-1] * (((zp < nz) ? dbz_in[IDX3(b,zp,y,x)] : scalar_t(0))
                       - ((zm >= 0) ? dbz_in[IDX3(b,zm,y,x)] : scalar_t(0)));
    }
    const scalar_t pb = pbar[i] - (ax * inv_dx + ay * inv_dy + az * inv_dz);
    pbar[i] = pb;

    atomicAdd(&gK[m], pb * (ph[h1] - ph[h0]) / kappa_m[m]);
    const scalar_t sbar = -dt * kappa_m[m] * pb;
    const scalar_t Fx = psibar_vxx[i] + sbar;
    const scalar_t Fy = psibar_vyy[i] + sbar;
    const scalar_t Fz = psibar_vzz[i] + sbar;
    dbx_out[i] = sbar / kx[x] + Fx * ax_[x];
    dby_out[i] = sbar / ky[y] + Fy * ay_[y];
    dbz_out[i] = sbar / kz[z] + Fz * az_[z];
    psibar_vxx[i] = Fx * bx[x];
    psibar_vyy[i] = Fy * by[y];
    psibar_vzz[i] = Fz * bz[z];
}

template <typename scalar_t>
__global__ void adj3d_ksrc_correct_kernel(
    const scalar_t* __restrict__ pbar, scalar_t* __restrict__ gK,
    const scalar_t* __restrict__ kappa_m,
    const int64_t* __restrict__ sz, const int64_t* __restrict__ sy,
    const int64_t* __restrict__ sx,
    const scalar_t* __restrict__ wav,
    const int it, const int nt, const int nz, const int ny, const int nx,
    const scalar_t dt, const int n_src)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_src) return;
    const long m = MIDX3(sz[b], sy[b], sx[b]);
    atomicAdd(&gK[m], -pbar[IDX3(b, sz[b], sy[b], sx[b])] * dt
                      * wav[(long)b * nt + it] / kappa_m[m]);
}

template <typename scalar_t>
__global__ void adj3d_velocity_stencil_kernel(
    scalar_t* __restrict__ vxbar, scalar_t* __restrict__ vybar,
    scalar_t* __restrict__ vzbar,
    const scalar_t* __restrict__ dbx, const scalar_t* __restrict__ dby,
    const scalar_t* __restrict__ dbz,
    const scalar_t* __restrict__ fd,
    const int H, const int nz, const int ny, const int nx, const int ztiles,
    const scalar_t inv_dx, const scalar_t inv_dy, const scalar_t inv_dz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int zt = blockIdx.z % ztiles, b = blockIdx.z / ztiles;
    const int z = zt * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const long i = IDX3(b, z, y, x);

    scalar_t ax = 0, ay = 0, az = 0;   // D+ per axis on the dbar fields
    for (int k = 1; k <= H; ++k) {
        const int xp = x + k, xm = x - k + 1;
        ax += fd[k-1] * (((xp < nx) ? dbx[IDX3(b,z,y,xp)] : scalar_t(0))
                       - ((xm >= 0) ? dbx[IDX3(b,z,y,xm)] : scalar_t(0)));
        const int yp = y + k, ym = y - k + 1;
        ay += fd[k-1] * (((yp < ny) ? dby[IDX3(b,z,yp,x)] : scalar_t(0))
                       - ((ym >= 0) ? dby[IDX3(b,z,ym,x)] : scalar_t(0)));
        const int zp = z + k, zm = z - k + 1;
        az += fd[k-1] * (((zp < nz) ? dbz[IDX3(b,zp,y,x)] : scalar_t(0))
                       - ((zm >= 0) ? dbz[IDX3(b,zm,y,x)] : scalar_t(0)));
    }
    vxbar[i] -= ax * inv_dx;
    vybar[i] -= ay * inv_dy;
    vzbar[i] -= az * inv_dz;
}

}  // namespace

void acoustic3d_forward(
    torch::Tensor p, torch::Tensor vx, torch::Tensor vy, torch::Tensor vz,
    torch::Tensor psi_vxx, torch::Tensor psi_vyy, torch::Tensor psi_vzz,
    torch::Tensor psi_px, torch::Tensor psi_py, torch::Tensor psi_pz,
    torch::Tensor kappa, torch::Tensor buoyancy, torch::Tensor fd,
    torch::Tensor bx_int, torch::Tensor ax_int, torch::Tensor kx_int,
    torch::Tensor by_int, torch::Tensor ay_int, torch::Tensor ky_int,
    torch::Tensor bz_int, torch::Tensor az_int, torch::Tensor kz_int,
    torch::Tensor bx_half, torch::Tensor ax_half, torch::Tensor kx_half,
    torch::Tensor by_half, torch::Tensor ay_half, torch::Tensor ky_half,
    torch::Tensor bz_half, torch::Tensor az_half, torch::Tensor kz_half,
    torch::Tensor src_z, torch::Tensor src_y, torch::Tensor src_x,
    torch::Tensor wav,
    torch::Tensor rcv_z, torch::Tensor rcv_y, torch::Tensor rcv_x,
    torch::Tensor out,
    double dt, double dx, double dy, double dz,
    torch::Tensor p_hist, torch::Tensor vx_hist, torch::Tensor vy_hist,
    torch::Tensor vz_hist)
{
    TORCH_CHECK(p.is_cuda() && p.is_contiguous(), "p must be contiguous CUDA");
    const int B = p.size(0), nz = p.size(2), ny = p.size(3), nx = p.size(4);
    const int nt = wav.size(1);
    const int n_src = src_z.size(0), n_rcv = rcv_z.size(0);
    const int H = fd.size(0);
    TORCH_CHECK(H >= 1 && H <= MAX_H, "fd half-order out of range");
    const bool with_hist = p_hist.numel() > 0;
    const long ncell = (long)B * nz * ny * nx;

    const dim3 threads(32, 4, 2);
    const int ztiles = (nz + 1) / 2;
    const dim3 grid((nx + 31) / 32, (ny + 3) / 4, (unsigned)(ztiles * B));
    const int sthreads = 128;
    const int sblocks = (n_src + sthreads - 1) / sthreads;
    const dim3 rgrid((n_rcv + 127) / 128, B, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "acoustic3d_forward", [&] {
        const scalar_t dt_ = (scalar_t)dt;
        const scalar_t ix_ = (scalar_t)(1.0 / dx), iy_ = (scalar_t)(1.0 / dy),
                       iz_ = (scalar_t)(1.0 / dz);
        scalar_t* ph_ = with_hist ? p_hist.data_ptr<scalar_t>() : nullptr;
        scalar_t* vxh_ = with_hist ? vx_hist.data_ptr<scalar_t>() : nullptr;
        scalar_t* vyh_ = with_hist ? vy_hist.data_ptr<scalar_t>() : nullptr;
        scalar_t* vzh_ = with_hist ? vz_hist.data_ptr<scalar_t>() : nullptr;
        for (int it = 0; it < nt; ++it) {
            pressure3d_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                p.data_ptr<scalar_t>(), vx.data_ptr<scalar_t>(),
                vy.data_ptr<scalar_t>(), vz.data_ptr<scalar_t>(),
                psi_vxx.data_ptr<scalar_t>(), psi_vyy.data_ptr<scalar_t>(),
                psi_vzz.data_ptr<scalar_t>(),
                kappa.data_ptr<scalar_t>(), fd.data_ptr<scalar_t>(),
                bx_int.data_ptr<scalar_t>(), ax_int.data_ptr<scalar_t>(), kx_int.data_ptr<scalar_t>(),
                by_int.data_ptr<scalar_t>(), ay_int.data_ptr<scalar_t>(), ky_int.data_ptr<scalar_t>(),
                bz_int.data_ptr<scalar_t>(), az_int.data_ptr<scalar_t>(), kz_int.data_ptr<scalar_t>(),
                ph_, it, ncell, H, nz, ny, nx, ztiles, dt_, ix_, iy_, iz_);
            velocity3d_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                p.data_ptr<scalar_t>(), vx.data_ptr<scalar_t>(),
                vy.data_ptr<scalar_t>(), vz.data_ptr<scalar_t>(),
                psi_px.data_ptr<scalar_t>(), psi_py.data_ptr<scalar_t>(),
                psi_pz.data_ptr<scalar_t>(),
                buoyancy.data_ptr<scalar_t>(), fd.data_ptr<scalar_t>(),
                bx_half.data_ptr<scalar_t>(), ax_half.data_ptr<scalar_t>(), kx_half.data_ptr<scalar_t>(),
                by_half.data_ptr<scalar_t>(), ay_half.data_ptr<scalar_t>(), ky_half.data_ptr<scalar_t>(),
                bz_half.data_ptr<scalar_t>(), az_half.data_ptr<scalar_t>(), kz_half.data_ptr<scalar_t>(),
                vxh_, vyh_, vzh_, it, ncell,
                H, nz, ny, nx, ztiles, dt_, ix_, iy_, iz_);
            source3d_kernel<scalar_t><<<sblocks, sthreads, 0, stream>>>(
                p.data_ptr<scalar_t>(),
                src_z.data_ptr<int64_t>(), src_y.data_ptr<int64_t>(),
                src_x.data_ptr<int64_t>(),
                wav.data_ptr<scalar_t>(), ph_, it, nt, ncell,
                nz, ny, nx, dt_, n_src);
            record3d_kernel<scalar_t><<<rgrid, 128, 0, stream>>>(
                p.data_ptr<scalar_t>(),
                rcv_z.data_ptr<int64_t>(), rcv_y.data_ptr<int64_t>(),
                rcv_x.data_ptr<int64_t>(),
                out.data_ptr<scalar_t>(), it, nt, n_rcv, nz, ny, nx);
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void acoustic3d_backward(
    torch::Tensor grad_out,
    torch::Tensor p_hist, torch::Tensor vx_hist, torch::Tensor vy_hist,
    torch::Tensor vz_hist,
    torch::Tensor kappa, torch::Tensor buoyancy, torch::Tensor fd,
    torch::Tensor bx_int, torch::Tensor ax_int, torch::Tensor kx_int,
    torch::Tensor by_int, torch::Tensor ay_int, torch::Tensor ky_int,
    torch::Tensor bz_int, torch::Tensor az_int, torch::Tensor kz_int,
    torch::Tensor bx_half, torch::Tensor ax_half, torch::Tensor kx_half,
    torch::Tensor by_half, torch::Tensor ay_half, torch::Tensor ky_half,
    torch::Tensor bz_half, torch::Tensor az_half, torch::Tensor kz_half,
    torch::Tensor src_z, torch::Tensor src_y, torch::Tensor src_x,
    torch::Tensor wav,
    torch::Tensor rcv_z, torch::Tensor rcv_y, torch::Tensor rcv_x,
    torch::Tensor pbar, torch::Tensor vxbar, torch::Tensor vybar,
    torch::Tensor vzbar,
    torch::Tensor psibar_vxx, torch::Tensor psibar_vyy, torch::Tensor psibar_vzz,
    torch::Tensor psibar_px, torch::Tensor psibar_py, torch::Tensor psibar_pz,
    // scratch x6: dbx/dby/dbz = velocity-stage, dcx/dcy/dcz = pressure-stage
    // (separate buffers for the fused pressure kernel; see 2D note)
    torch::Tensor dbx, torch::Tensor dby, torch::Tensor dbz,
    torch::Tensor dcx, torch::Tensor dcy, torch::Tensor dcz,
    torch::Tensor gK, torch::Tensor gB, torch::Tensor gwav,
    double dt, double dx, double dy, double dz)
{
    const int B = pbar.size(0), nz = pbar.size(2), ny = pbar.size(3),
              nx = pbar.size(4);
    const int nt = wav.size(1);
    const int n_src = src_z.size(0), n_rcv = rcv_z.size(0);
    const int H = fd.size(0);
    const long ncell = (long)B * nz * ny * nx;

    const dim3 threads(32, 4, 2);
    const int ztiles = (nz + 1) / 2;
    const dim3 grid((nx + 31) / 32, (ny + 3) / 4, (unsigned)(ztiles * B));
    const int sthreads = 128;
    const int sblocks = (n_src + sthreads - 1) / sthreads;
    const dim3 rgrid((n_rcv + 127) / 128, B, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(pbar.scalar_type(), "acoustic3d_backward", [&] {
        const scalar_t dt_ = (scalar_t)dt;
        const scalar_t ix_ = (scalar_t)(1.0 / dx), iy_ = (scalar_t)(1.0 / dy),
                       iz_ = (scalar_t)(1.0 / dz);
        for (int it = nt - 1; it >= 0; --it) {
            adj3d_record_scatter_kernel<scalar_t><<<rgrid, 128, 0, stream>>>(
                pbar.data_ptr<scalar_t>(), grad_out.data_ptr<scalar_t>(),
                rcv_z.data_ptr<int64_t>(), rcv_y.data_ptr<int64_t>(),
                rcv_x.data_ptr<int64_t>(), it, nt, n_rcv, nz, ny, nx);
            adj3d_source_kernel<scalar_t><<<sblocks, sthreads, 0, stream>>>(
                pbar.data_ptr<scalar_t>(),
                src_z.data_ptr<int64_t>(), src_y.data_ptr<int64_t>(),
                src_x.data_ptr<int64_t>(),
                gwav.data_ptr<scalar_t>(), it, nt, nz, ny, nx, dt_, n_src);
            adj3d_velocity_point_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                vxbar.data_ptr<scalar_t>(), vybar.data_ptr<scalar_t>(),
                vzbar.data_ptr<scalar_t>(),
                psibar_px.data_ptr<scalar_t>(), psibar_py.data_ptr<scalar_t>(),
                psibar_pz.data_ptr<scalar_t>(),
                dbx.data_ptr<scalar_t>(), dby.data_ptr<scalar_t>(),
                dbz.data_ptr<scalar_t>(),
                gB.data_ptr<scalar_t>(), buoyancy.data_ptr<scalar_t>(),
                vx_hist.data_ptr<scalar_t>(), vy_hist.data_ptr<scalar_t>(),
                vz_hist.data_ptr<scalar_t>(),
                bx_half.data_ptr<scalar_t>(), ax_half.data_ptr<scalar_t>(), kx_half.data_ptr<scalar_t>(),
                by_half.data_ptr<scalar_t>(), ay_half.data_ptr<scalar_t>(), ky_half.data_ptr<scalar_t>(),
                bz_half.data_ptr<scalar_t>(), az_half.data_ptr<scalar_t>(), kz_half.data_ptr<scalar_t>(),
                it, ncell, nz, ny, nx, ztiles, dt_);
            adj3d_pressure_fused_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                pbar.data_ptr<scalar_t>(),
                dbx.data_ptr<scalar_t>(), dby.data_ptr<scalar_t>(),
                dbz.data_ptr<scalar_t>(),
                psibar_vxx.data_ptr<scalar_t>(), psibar_vyy.data_ptr<scalar_t>(),
                psibar_vzz.data_ptr<scalar_t>(),
                dcx.data_ptr<scalar_t>(), dcy.data_ptr<scalar_t>(),
                dcz.data_ptr<scalar_t>(),
                gK.data_ptr<scalar_t>(), kappa.data_ptr<scalar_t>(),
                p_hist.data_ptr<scalar_t>(), fd.data_ptr<scalar_t>(),
                bx_int.data_ptr<scalar_t>(), ax_int.data_ptr<scalar_t>(), kx_int.data_ptr<scalar_t>(),
                by_int.data_ptr<scalar_t>(), ay_int.data_ptr<scalar_t>(), ky_int.data_ptr<scalar_t>(),
                bz_int.data_ptr<scalar_t>(), az_int.data_ptr<scalar_t>(), kz_int.data_ptr<scalar_t>(),
                it, ncell, H, nz, ny, nx, ztiles, dt_, ix_, iy_, iz_);
            adj3d_ksrc_correct_kernel<scalar_t><<<sblocks, sthreads, 0, stream>>>(
                pbar.data_ptr<scalar_t>(), gK.data_ptr<scalar_t>(),
                kappa.data_ptr<scalar_t>(),
                src_z.data_ptr<int64_t>(), src_y.data_ptr<int64_t>(),
                src_x.data_ptr<int64_t>(),
                wav.data_ptr<scalar_t>(), it, nt, nz, ny, nx, dt_, n_src);
            adj3d_velocity_stencil_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                vxbar.data_ptr<scalar_t>(), vybar.data_ptr<scalar_t>(),
                vzbar.data_ptr<scalar_t>(),
                dcx.data_ptr<scalar_t>(), dcy.data_ptr<scalar_t>(),
                dcz.data_ptr<scalar_t>(),
                fd.data_ptr<scalar_t>(), H, nz, ny, nx, ztiles, ix_, iy_, iz_);
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &acoustic3d_forward, "acoustic 3D forward (CUDA, C++ loop)");
    m.def("backward", &acoustic3d_backward, "acoustic 3D adjoint (CUDA, C++ loop)");
}
