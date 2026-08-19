// GeoBrain G6 stage-A build probe: a trivial fused op with HAND-WRITTEN matched forward and
// adjoint (backward) CUDA kernels, to de-risk the cpp_extension toolchain on Sherlock
// (nvcc <-> torch ABI, gcc) before writing the real FDTD kernels. y = sin(x) * scale;
// dL/dx = cos(x) * scale * dL/dy.
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void probe_forward_kernel(const scalar_t* __restrict__ x,
                                     scalar_t* __restrict__ y,
                                     double scale, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) y[i] = sin(x[i]) * (scalar_t)scale;
}

template <typename scalar_t>
__global__ void probe_backward_kernel(const scalar_t* __restrict__ x,
                                      const scalar_t* __restrict__ gy,
                                      scalar_t* __restrict__ gx,
                                      double scale, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) gx[i] = cos(x[i]) * (scalar_t)scale * gy[i];
}

torch::Tensor probe_forward(torch::Tensor x, double scale) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    auto xc = x.contiguous();
    auto y = torch::empty_like(xc);
    int64_t n = xc.numel();
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "probe_forward", [&] {
        probe_forward_kernel<scalar_t><<<blocks, threads>>>(
            xc.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), scale, n);
    });
    return y;
}

torch::Tensor probe_backward(torch::Tensor x, torch::Tensor gy, double scale) {
    auto xc = x.contiguous();
    auto gyc = gy.contiguous();
    auto gx = torch::empty_like(xc);
    int64_t n = xc.numel();
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "probe_backward", [&] {
        probe_backward_kernel<scalar_t><<<blocks, threads>>>(
            xc.data_ptr<scalar_t>(), gyc.data_ptr<scalar_t>(),
            gx.data_ptr<scalar_t>(), scale, n);
    });
    return gx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &probe_forward, "probe fused forward (CUDA)");
    m.def("backward", &probe_backward, "probe fused adjoint (CUDA)");
}
