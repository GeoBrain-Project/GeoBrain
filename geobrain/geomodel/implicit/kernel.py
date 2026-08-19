"""
Covariance kernels and their analytical derivatives for Cokriging interpolation.

Implements the kernel functions needed for Universal Cokriging with gradient
constraints. Each kernel provides C(r), C'(r), C''(r) and the derived
covariance blocks: C_pp, C_up, C_uu.

Mathematical reference:
    Lajaunie, C., Courrioux, G., & Manuel, L. (1997). Foliation fields and
    3D cartography in geology. Mathematical Geology, 29(4), 571-584.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class CovarianceKernel(ABC, nn.Module):
    """
    Abstract covariance kernel with first and second order derivatives.

    All kernels are parameterized by:
        a:   range (correlation length)
        c_o, sill (variance at lag 0)

    Subclasses must implement C(r), C_prime(r), C_double_prime(r) where
    r = ||h|| / a is the normalized distance.
    """

    def __init__(self, a: float, c_o: float, aniso: torch.Tensor | None = None):
        super().__init__()
        self.a = a
        self.c_o = c_o
        # Optional anisotropy: a (D, D) transform applied to displacement
        # vectors (and, consistently, to orientation gradients) so the kernel
        # operates in a rotated/rescaled metric: letting the prior encode the
        # directional correlation of real geology (layering, dip) instead of an
        # isotropic blob. ``None`` ⇒ isotropic (identity). Applying the *same*
        # transform to displacements and to the gradient constraints keeps the
        # gradient-constrained cokriging exact; a degree-≤1 polynomial drift is
        # invariant under a linear coordinate transform, so the interpolator's
        # drift basis needs no change.
        self.aniso = aniso

    # ------------------------------------------------------------------
    # Abstract: radial basis function and its derivatives w.r.t. r
    # ------------------------------------------------------------------

    @abstractmethod
    def C(self, r: torch.Tensor) -> torch.Tensor:
        """Kernel value C(r)."""

    @abstractmethod
    def C_prime(self, r: torch.Tensor) -> torch.Tensor:
        """First derivative dC/dr."""

    @abstractmethod
    def C_double_prime(self, r: torch.Tensor) -> torch.Tensor:
        """Second derivative d²C/dr²."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def compute_r(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Compute displacement vectors and normalized distances.

        Args:
            x1: (N, D) coordinates.
            x2: (M, D) coordinates.

        Returns:
            h: (N, M, D) displacement vectors  h = x1 - x2.
            r: (N, M) normalized distances ||h|| / a.
        """
        # h[i, j, d] = x1[i, d] - x2[j, d]
        h = x1.unsqueeze(1) - x2.unsqueeze(0)  # (N, M, D)
        if self.aniso is not None:
            # metric-transformed displacement h̃ = (T h); the gradient blocks
            # transform their orientation vectors identically (see _aniso_grad).
            h = h @ self.aniso.transpose(-1, -2)
        r = h.norm(dim=-1) / self.a             # (N, M)
        return h, r

    def _aniso_grad(self, g: torch.Tensor) -> torch.Tensor:
        """Apply the metric transform to orientation/gradient vectors so the
        directional-derivative constraints stay consistent with the transformed
        displacements (a no-op when isotropic)."""
        return g if self.aniso is None else g @ self.aniso.transpose(-1, -2)

    # ------------------------------------------------------------------
    # Covariance blocks
    # ------------------------------------------------------------------

    def cov_pp(self, sp1: torch.Tensor, sp2: torch.Tensor) -> torch.Tensor:
        """
        Point-point covariance C_pp[i,j] = C(||sp1_i - sp2_j|| / a).

        Args:
            sp1: (N, D) first set of surface points.
            sp2: (M, D) second set of surface points.

        Returns:
            (N, M) covariance matrix.
        """
        _, r = self.compute_r(sp1, sp2)
        return self.C(r)

    def cov_up(
        self,
        ori: torch.Tensor,
        sp: torch.Tensor,
        gradients: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gradient-point cross-covariance (C_up block).

        C_up[i, j] = sum_d  G_d^(i) * dC/dh_d(ori_i, sp_j)

        where, with r = ||h|| / a,
            dr/dh_d  = h_d / (||h|| * a) = h_d / (r * a²)
            dC/dh_d  = C'(r) * dr/dh_d = C'(r) * h_d / (r * a²).

        Args:
            ori: (M, D) orientation locations.
            sp:  (N, D) surface point locations.
            gradients: (M, D) unit gradient vectors at orientation points.

        Returns:
            (M, N) covariance matrix.
        """
        # h[i,j,d] = ori_i - sp_j
        h, r = self.compute_r(ori, sp)  # h: (M, N, D), r: (M, N)
        gradients = self._aniso_grad(gradients)
        r_safe = r.clamp(min=1e-12)

        Cp = self.C_prime(r)  # (M, N)

        # dC/dh_d = C'(r) * h_d / (r * a²)  →  (M, N, D)
        # (r = ||h||/a, so dr/dh_d = h_d / (||h|| a) = h_d / (r a²): note
        # the a² in the denominator.)
        dC_dh = Cp.unsqueeze(-1) * h / (r_safe.unsqueeze(-1) * self.a * self.a)

        # Contract with gradient: sum over d
        # gradients: (M, D) → (M, 1, D)
        result = (gradients.unsqueeze(1) * dC_dh).sum(dim=-1)  # (M, N)
        return result

    def cov_uu(
        self,
        ori1: torch.Tensor,
        ori2: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gradient-gradient covariance (C_uu block).

        C_uu[i,j] = sum_{d,e} G_d^(i) * G_e^(j) * d²C/(dh_d dh_e)

        where, with r = ||h|| / a:
            d²C/(dh_d dh_e) = [C''(r) - C'(r)/r] * h_d*h_e / (r² * a⁴)
                             + C'(r) / (r * a²) * delta_{de}

        Args:
            ori1: (M1, D) first set of orientation locations.
            ori2: (M2, D) second set of orientation locations.
            g1:   (M1, D) unit gradient vectors at ori1.
            g2:   (M2, D) unit gradient vectors at ori2.

        Returns:
            (M1, M2) covariance matrix.
        """
        h, r = self.compute_r(ori1, ori2)  # h: (M1,M2,D), r: (M1,M2)
        g1 = self._aniso_grad(g1)
        g2 = self._aniso_grad(g2)
        r_safe = r.clamp(min=1e-12)

        Cp = self.C_prime(r)              # (M1, M2)
        Cpp = self.C_double_prime(r)      # (M1, M2)

        a2 = self.a ** 2
        a4 = a2 ** 2
        r2 = r_safe ** 2

        # Term 1 coefficient: [C''(r) - C'(r)/r] / (r² * a⁴)
        # (each dr/dh contributes a 1/a², so the h_d*h_e term carries a⁴.)
        coeff1 = (Cpp - Cp / r_safe) / (r2 * a4)  # (M1, M2)

        # Term 2 coefficient: C'(r) / (r * a²)
        coeff2 = Cp / (r_safe * a2)  # (M1, M2)

        # g1: (M1, D), g2: (M2, D)
        # We need: sum_{d,e} g1_d * g2_e * [coeff1 * h_d * h_e + coeff2 * delta_{de}]

        # Part A: coeff1 * (g1 . h) * (g2 . h)
        # g1_dot_h[i,j] = sum_d g1[i,d] * h[i,j,d]
        g1_dot_h = (g1.unsqueeze(1) * h).sum(dim=-1)  # (M1, M2)
        g2_dot_h = (g2.unsqueeze(0) * h).sum(dim=-1)  # (M1, M2)
        part_a = coeff1 * g1_dot_h * g2_dot_h

        # Part B: coeff2 * (g1 . g2)
        g1_dot_g2 = (g1.unsqueeze(1) * g2.unsqueeze(0)).sum(dim=-1)  # (M1, M2)
        part_b = coeff2 * g1_dot_g2

        # r == 0 limit (coincident orientation points: always hit on the
        # diagonal because build_system calls cov_uu(ori, ori, ...)). There
        # part_a vanishes (h = 0 exactly) and coeff2 = C'(r)/(r a²) is the
        # indeterminate 0/0 form, since C'(0) = 0 for both kernels. The
        # directional-derivative variance limit is fixed by L'Hôpital,
        # C'(r)/r → C''(0), giving
        #     C_uu(0) = C''(0) / a² · (g1 · g2)
        #, the analytic second-derivative variance, not 0. Returning 0 here
        # zeroes the C_uu diagonal → indefinite system → the cokriging silently
        # stops honoring the measured dips. We use the same anisotropy-transformed
        # gradients as the off-diagonal so the limit stays consistent.
        Cpp0 = self.C_double_prime(torch.zeros_like(r))  # C''(0), (M1, M2)
        patch = Cpp0 / a2 * g1_dot_g2  # (M1, M2)

        return torch.where(r == 0, patch, part_a + part_b)


class CubicKernel(CovarianceKernel):
    """
    Cubic covariance kernel (compact support, C² smooth).

    This is the default kernel used in GemPy.

    C(r) = c_o * (1 - 7r² + 35/4 r³ - 7/2 r⁵ + 3/4 r⁷)   for r < 1
         = 0                                                  for r >= 1
    """

    def C(self, r: torch.Tensor) -> torch.Tensor:
        r2 = r * r
        r3 = r2 * r
        r5 = r2 * r3
        r7 = r2 * r5
        val = self.c_o * (1.0 - 7.0 * r2 + 8.75 * r3 - 3.5 * r5 + 0.75 * r7)
        return torch.where(r < 1.0, val, torch.zeros_like(val))

    def C_prime(self, r: torch.Tensor) -> torch.Tensor:
        """dC/dr."""
        r2 = r * r
        r4 = r2 * r2
        r6 = r2 * r4
        val = self.c_o * (-14.0 * r + 26.25 * r2 - 17.5 * r4 + 5.25 * r6)
        return torch.where(r < 1.0, val, torch.zeros_like(val))

    def C_double_prime(self, r: torch.Tensor) -> torch.Tensor:
        """d²C/dr²."""
        r3 = r * r * r
        r5 = r3 * r * r
        val = self.c_o * (-14.0 + 52.5 * r - 70.0 * r3 + 31.5 * r5)
        return torch.where(r < 1.0, val, torch.zeros_like(val))


class GaussianKernel(CovarianceKernel):
    """
    Gaussian (squared-exponential) covariance kernel.

    C(r) = c_o * exp(-r²)

    Infinitely differentiable but not compactly supported.
    """

    def C(self, r: torch.Tensor) -> torch.Tensor:
        return self.c_o * torch.exp(-r * r)

    def C_prime(self, r: torch.Tensor) -> torch.Tensor:
        return self.c_o * (-2.0 * r) * torch.exp(-r * r)

    def C_double_prime(self, r: torch.Tensor) -> torch.Tensor:
        r2 = r * r
        return self.c_o * (4.0 * r2 - 2.0) * torch.exp(-r2)


# Default kernel range: ``ImplicitModelConfig.range`` (GemPy convention,
# sqrt(3)). ``anisotropy_matrix_2d``'s ``base_range`` MUST track this so that
# the default-constructed transform stays calibrated to the default kernel.
_DEFAULT_KERNEL_RANGE = 1.7320508075688772  # sqrt(3)


def anisotropy_matrix_2d(
    major_range: float,
    minor_range: float,
    angle_deg: float = 0.0,
    base_range: float = _DEFAULT_KERNEL_RANGE,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Build a 2D anisotropy transform for the implicit cokriging kernel.

    Encodes a longer correlation ``major_range`` along an axis oriented
    ``angle_deg`` (degrees CCW from +x) and a shorter ``minor_range``
    perpendicular, the layered / dipping correlation structure of real
    geology. Pass the result as ``ImplicitModelConfig(anisotropy=...)``.

    ``base_range`` MUST equal the kernel ``range`` (``config.range``); the kernel
    evaluates ``r = ||T·h|| / base_range``, so ``T = diag(base/major, base/minor)
    @ R(angle)`` makes the correlation reach ~0 at ``major_range`` along the
    major axis. The default is the kernel's own default ``range`` (sqrt(3),
    matching :class:`ImplicitModelConfig`), so a user who passes only
    ``(major, minor, angle)`` stays calibrated against the default kernel;
    a mismatched ``base_range`` would silently rescale every correlation
    length. Returns the identity when ``major == minor == base_range``
    (isotropic).
    """
    import math

    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    # R rotates physical coords so the major axis aligns with +x.
    R = torch.tensor([[c, s], [-s, c]], dtype=dtype)
    S = torch.diag(torch.tensor(
        [base_range / major_range, base_range / minor_range], dtype=dtype))
    return S @ R
