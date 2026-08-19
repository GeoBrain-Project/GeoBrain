"""
Cokriging interpolator for implicit geological modeling.

Implements the Universal Cokriging system with gradient constraints:
assembles the kriging matrix, solves for weights, evaluates the scalar
field, and converts it to lithology blocks (hard or differentiable).

Reference:
    Lajaunie, C., Courrioux, G., & Manuel, L. (1997). Foliation fields and
    3D cartography in geology. Mathematical Geology, 29(4), 571-584.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import torch
import torch.nn as nn

from .data import InterpolationInput
from .kernel import CovarianceKernel


class CokrigingInterpolator(nn.Module):
    """
    Universal Cokriging solver with gradient constraints.

    Builds and solves the universal-cokriging linear system. Orientation
    (dip) data enter as the FULL gradient VECTOR, each orientation o_i
    contributes D observations ∂_d Z(o_i) = g_i[d], so the gradient dual
    block is (M·D, M·D), reproducing arbitrary asymmetric dips exactly:

        [ C_GG + eps_u*I   C_GP            F_G ] [w_G]   [g]
        [ C_GP^T           C_pp + eps_p*I  F_p ] [w_p] = [0]
        [ F_G^T            F_p^T            0  ] [ l ]   [0]

    Then evaluates the scalar field at query points and converts to
    lithology identifiers. See :meth:`build_system` for the block definitions.

    Args:
        kernel: Covariance kernel instance (CubicKernel or GaussianKernel).
        drift_degree: Polynomial drift degree (0=constant, 1=linear).
    """

    def __init__(self, kernel: CovarianceKernel, drift_degree: int = 1):
        super().__init__()
        self.kernel = kernel
        self.drift_degree = drift_degree

    # ------------------------------------------------------------------
    # Drift basis functions
    # ------------------------------------------------------------------

    def _drift_functions(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Evaluate polynomial drift basis at given coordinates.

        degree 0: [1]
        degree 1: [1, x, y(, z)]

        Args:
            coords: (N, D) coordinates.

        Returns:
            (N, n_drift) drift matrix.
        """
        N = coords.shape[0]
        ones = torch.ones(N, 1, dtype=coords.dtype, device=coords.device)
        if self.drift_degree == 0:
            return ones
        # degree 1: [1, x, y, ...]
        return torch.cat([ones, coords], dim=-1)

    def _drift_gradients(self, coords: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        """
        Drift functions differentiated by gradient direction.

        For degree 0: all zeros (constant drift has zero gradient).
        For degree 1: drift = [1, x, y, z], gradient = [0, G_x, G_y, G_z].

        Args:
            coords: (M, D) orientation coordinates.
            gradients: (M, D) unit gradient vectors.

        Returns:
            (M, n_drift) drift-gradient matrix.
        """
        M, D = coords.shape
        if self.drift_degree == 0:
            return torch.zeros(M, 1, dtype=coords.dtype, device=coords.device)
        # degree 1: derivative of [1, x_1, ..., x_D] w.r.t. direction g
        # d(1)/dg = 0, d(x_d)/dg = g_d
        zeros = torch.zeros(M, 1, dtype=coords.dtype, device=coords.device)
        return torch.cat([zeros, gradients], dim=-1)

    @property
    def _n_drift(self) -> int:
        """Number of drift basis functions (set after build_system)."""
        return self._cached_n_drift

    # ------------------------------------------------------------------
    # System assembly
    # ------------------------------------------------------------------

    def _basis_dirs(self, M: int, D: int, dtype, device) -> list[torch.Tensor]:
        """Physical unit basis directions e_d, each broadcast to ``(M, D)``.

        Passing e_d as the "gradient" to the kernel's ``cov_uu`` / ``cov_up``
        yields the per-component physical gradient covariance: the kernel's
        ``_aniso_grad`` transforms e_d identically to the displacement, so
        ``cov_uu(x,y;e_d,e_e) = (Tᵀ Hess T)_{de} = −Cov{∂_d Z(x), ∂_e Z(y)}``
        holds exactly under anisotropy too. The full gradient-vector cokriging
        system is assembled from these.
        """
        eye = torch.eye(D, dtype=dtype, device=device)
        return [eye[d].unsqueeze(0).expand(M, D) for d in range(D)]

    def build_system(self, inp: InterpolationInput):
        """
        Assemble the Cokriging matrix A and right-hand-side b.

        Orientation (dip) data enter as the FULL gradient VECTOR: each
        orientation o_i contributes D scalar observations ∂_d Z(o_i) = g_i[d],
        so the gradient dual block is ``(M·D, M·D)``, one row per
        (orientation, component). This reproduces arbitrary asymmetric dips
        exactly; the earlier single-scalar-row-per-orientation form (RHS=1)
        pinned only the dip PROJECTION g·∇Z and silently failed to honor
        conflicting dip vectors (the operator that imposed the constraint was
        also the negative of the one that reconstructed it).

        Blocks (MG = M·D gradient duals, N point increments, n_drift):
            C_GG[(i,d),(j,e)] = −cov_uu(o_i,o_j; e_d,e_e)  (mixed 2nd derivative;
                the kernel's L'Hôpital-patched r=0 diagonal makes the negated
                self-term the positive directional-derivative variance),
            C_GP[(i,d),j]     = cov_up(o_i,sp_j; e_d),
            F_G               = per-component drift gradient,
            gradient RHS      = the measured dip components g_i.

        Args:
            inp: Packed interpolation input.

        Returns:
            A: (S, S) kriging matrix, S = M·D + N + n_drift.
            b: (S,) right-hand-side vector.
        """
        M = inp.ori_coords.shape[0]
        N = inp.sp_coords.shape[0]
        D = inp.ori_coords.shape[1]
        dtype = inp.ori_coords.dtype
        device = inp.ori_coords.device
        MG = M * D

        E = self._basis_dirs(M, D, dtype, device)

        # Gradient-gradient block (MG, MG), component-blocked:
        # C_GG[d-block, e-block] = −cov_uu(ori, ori, e_d, e_e).
        C_GG = torch.cat(
            [
                torch.cat(
                    [-self.kernel.cov_uu(inp.ori_coords, inp.ori_coords, E[d], E[e])
                     for e in range(D)],
                    dim=1,
                )
                for d in range(D)
            ],
            dim=0,
        )  # (MG, MG)
        # Gradient-point cross block (MG, N): stack cov_up per component.
        C_GP = torch.cat(
            [self.kernel.cov_up(inp.ori_coords, inp.sp_coords, E[d]) for d in range(D)],
            dim=0,
        )  # (MG, N)
        C_pp = self.kernel.cov_pp(inp.sp_coords, inp.sp_coords)  # (N, N)

        # Nugget (one per gradient observation / per point).
        C_GG = C_GG + inp.ori_nugget * torch.eye(MG, dtype=dtype, device=device)
        C_pp = C_pp + inp.sp_nugget * torch.eye(N, dtype=dtype, device=device)

        # Drift: per-component gradient rows for the orientation side.
        F_G = torch.cat(
            [self._drift_gradients(inp.ori_coords, E[d]) for d in range(D)], dim=0
        )  # (MG, n_drift)
        F_p = self._drift_functions(inp.sp_coords)  # (N, n_drift)
        n_drift = F_p.shape[1]
        self._cached_n_drift = n_drift

        S = MG + N + n_drift
        A = torch.zeros(S, S, dtype=dtype, device=device)
        A[:MG, :MG] = C_GG
        A[:MG, MG:MG+N] = C_GP
        A[MG:MG+N, :MG] = C_GP.T
        A[MG:MG+N, MG:MG+N] = C_pp
        A[:MG, MG+N:] = F_G
        A[MG:MG+N, MG+N:] = F_p
        A[MG+N:, :MG] = F_G.T
        A[MG+N:, MG:MG+N] = F_p.T

        # RHS: gradient rows = measured dip components (component-blocked,
        # matching C_GG / w_G), rest zero.
        b_G = torch.cat([inp.ori_gradients[:, d] for d in range(D)], dim=0)  # (MG,)
        b = torch.cat(
            [b_G, torch.zeros(N + n_drift, dtype=dtype, device=device)], dim=0
        )
        return A, b

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self, inp: InterpolationInput) -> torch.Tensor:
        """
        Solve the Cokriging system for weights.

        Uses torch.linalg.solve with lstsq fallback for singular systems.

        Args:
            inp: Packed interpolation input.

        Returns:
            weights: (S,) weight vector [w_G, w_p, lambda] (S = M·D + N + n_drift).
        """
        A, b = self.build_system(inp)
        try:
            weights = torch.linalg.solve(A, b)
        except torch.linalg.LinAlgError:
            # Fallback to least-squares for near-singular systems
            result = torch.linalg.lstsq(A, b.unsqueeze(-1))
            weights = result.solution.squeeze(-1)
        return weights

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        query: torch.Tensor,
        inp: InterpolationInput,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the scalar field at query points.

            Z(x) = Σ_{i,d} w_G[(i,d)] · cov_up(o_i, x; e_d)   (gradient duals)
                 + Σ_j     w_p[j]     · C(x, sp_j)             (point duals)
                 + Σ_k     λ_k        · f_k(x)                 (drift)

        The gradient term sums the per-(orientation, component) duals against
        the matching basis-direction cross-covariance, the consistent adjoint
        of the gradient-vector constraints assembled in :meth:`build_system`,
        so the reconstructed ∇Z honors the measured dip vectors.

        Args:
            query: (Q, D) query coordinates.
            inp: Interpolation input (for data point locations).
            weights: (S,) solution weights [w_G, w_p, λ] from solve().

        Returns:
            Z: (Q,) scalar field values.
        """
        M = inp.ori_coords.shape[0]
        N = inp.sp_coords.shape[0]
        D = inp.ori_coords.shape[1]
        MG = M * D
        dtype = inp.ori_coords.dtype
        device = inp.ori_coords.device

        w_G = weights[:MG]
        w_p = weights[MG:MG+N]
        w_drift = weights[MG+N:]

        E = self._basis_dirs(M, D, dtype, device)

        # Gradient duals: Σ_d Σ_i w_G[(i,d)] · cov_up(o_i, query; e_d).
        term_u = torch.zeros(query.shape[0], dtype=dtype, device=device)
        for d in range(D):
            C_dq = self.kernel.cov_up(inp.ori_coords, query, E[d])  # (M, Q)
            w_d = w_G[d*M:(d+1)*M]                                  # (M,)
            term_u = term_u + (w_d.unsqueeze(-1) * C_dq).sum(dim=0)  # (Q,)

        # Point covariance contributions.
        C_pq = self.kernel.cov_pp(inp.sp_coords, query)  # (N, Q)
        term_p = (w_p.unsqueeze(-1) * C_pq).sum(dim=0)  # (Q,)

        # Drift contributions.
        F_q = self._drift_functions(query)  # (Q, n_drift)
        term_drift = (F_q * w_drift.unsqueeze(0)).sum(dim=-1)  # (Q,)

        return term_u + term_p + term_drift

    def interpolate(
        self,
        query: torch.Tensor,
        inp: InterpolationInput,
    ) -> torch.Tensor:
        """
        Solve and evaluate in one call.

        Args:
            query: (Q, D) query coordinates.
            inp: Interpolation input.

        Returns:
            Z: (Q,) scalar field values at query points.
        """
        weights = self.solve(inp)
        return self.evaluate(query, inp, weights)

    # ------------------------------------------------------------------
    # Scalar field → lithology
    # ------------------------------------------------------------------

    def compute_iso_values(
        self,
        Z_at_sp: torch.Tensor,
        surface_ids: torch.Tensor,
        n_surfaces: int,
    ) -> torch.Tensor:
        """
        Compute iso-surface values as the mean Z per surface.

        Args:
            Z_at_sp: (N,) scalar field values at surface points.
            surface_ids: (N,) integer surface id for each point.
            n_surfaces: Total number of surfaces.

        Returns:
            iso_vals: (n_surfaces,) iso-surface values, sorted ascending.
        """
        iso_vals = torch.zeros(n_surfaces, dtype=Z_at_sp.dtype, device=Z_at_sp.device)
        for k in range(n_surfaces):
            mask = surface_ids == k
            if mask.any():
                iso_vals[k] = Z_at_sp[mask].mean()
        # Sort iso values for consistent ordering
        iso_vals, _ = iso_vals.sort()
        return iso_vals

    def scalar_field_to_block(
        self,
        Z: torch.Tensor,
        iso_vals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hard classification: assign lithology id based on scalar field intervals.

        Each grid point gets the index of the interval it falls into.

        Args:
            Z: (Q,) scalar field values.
            iso_vals: (n_surfaces,) sorted iso-surface values.

        Returns:
            block: (Q,) integer lithology ids (0 to n_surfaces).
        """
        # Count how many iso-values each Z exceeds
        # Z > iso_vals[k] for each k → sum gives the block id
        block = torch.zeros_like(Z, dtype=torch.long)
        for k in range(len(iso_vals)):
            block = block + (Z > iso_vals[k]).long()
        return block

    def scalar_field_to_block_soft(
        self,
        Z: torch.Tensor,
        iso_vals: torch.Tensor,
        temperature: float = 50.0,
    ) -> torch.Tensor:
        """
        Differentiable soft classification using sigmoid.

        block_soft = sum_k sigmoid(T * (Z - iso_k))

        This produces a continuous approximation of the lithology id,
        enabling gradient flow through the classification step.

        Args:
            Z: (Q,) scalar field values.
            iso_vals: (n_surfaces,) sorted iso-surface values.
            temperature: Sigmoid sharpness (higher = closer to hard boundary).

        Returns:
            block_soft: (Q,) continuous lithology ids.
        """
        block_soft = torch.zeros_like(Z)
        for k in range(len(iso_vals)):
            block_soft = block_soft + torch.sigmoid(temperature * (Z - iso_vals[k]))
        return block_soft

    def scalar_field_to_property(
        self,
        Z: torch.Tensor,
        iso_vals: torch.Tensor,
        properties: torch.Tensor,
        temperature: float = 50.0,
    ) -> torch.Tensor:
        """Differentiable map from scalar field to a physical property field.

        This is the bridge that turns an implicit geological model into a
        differentiable property prior for the physics inversion: it assigns
        ``properties[0]`` below the lowest iso-surface and ``properties[k]`` in
        the k-th lithology, blending smoothly across each iso-surface (sigmoid)
        so gradients flow from a geophysical misfit, through the property field,
        back to the geological inputs (surface points / orientations / the
        property values themselves).

        Args:
            Z: ``(Q,)`` scalar field values (e.g. on a grid).
            iso_vals: ``(n_surfaces,)`` sorted iso-surface values.
            properties: ``(n_surfaces + 1,)`` property value per lithology,
                ordered bottom → top (e.g. density per layer).
            temperature: sigmoid sharpness (higher = sharper contacts).

        Returns:
            ``(Q,)`` property field, differentiable in ``Z`` and ``properties``.
        """
        properties = torch.as_tensor(properties, dtype=Z.dtype, device=Z.device)
        prop = properties[0] * torch.ones_like(Z)
        for k in range(len(iso_vals)):
            jump = properties[k + 1] - properties[k]
            prop = prop + jump * torch.sigmoid(temperature * (Z - iso_vals[k]))
        return prop
