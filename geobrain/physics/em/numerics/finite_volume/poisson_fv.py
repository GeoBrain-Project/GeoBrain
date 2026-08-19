# pyright: reportPrivateImportUsage=false
"""Mesh-agnostic cell-centred FV Poisson assembly from FaceRecords.

Generalises :func:`assemble_poisson_octree_3d` (which walks octree-specific
``(i, j, axis, area, dist)`` tuples from ``_find_octree_face_neighbors``) to
consume the mesh-agnostic :class:`FaceRecords` produced by **any**
:class:`ConnectivityMesh` (TensorMesh, OctreeMesh, …). The discrete operator is
the standard TPFA cell-centred FV stencil with the series-harmonic two-point
transmissibility written directly in the per-side HALF-distances
``dist_l``/``dist_r`` (cell centre → shared face)::

    T_f = σ_l · σ_r · area / (σ_l · dist_r + σ_r · dist_l + eps)

This single form reproduces BOTH legacy assemblers to ``atol=1e-12`` (≈1e-15
in practice; the forms differ only by eps placement and float reassociation):

* :func:`assemble_poisson_3d` (TensorMesh) stamps
  ``T = 2·σ_l·σ_r·area / (σ_l·D_r + σ_r·D_l + eps)`` where ``D_l``/``D_r`` are
  the **full** neighbour cell widths. Since the FaceRecords distances are
  halves (``D = 2·dist``), the legacy ``2`` cancels the half→full doubling and
  the expression collapses onto ``T_f`` above to ``atol=1e-12`` (≈1e-15 in
  practice, the forms differ only by eps placement and float reassociation),
  including for non-uniform faces where ``dist_l ≠ dist_r``.
* :func:`assemble_poisson_octree_3d` stamps ``σ_face·area/dist`` with the
  harmonic mean ``σ_face = 2σ_lσ_r/(σ_l+σ_r)`` and the **full** centre-to-centre
  ``dist = dist_l + dist_r``. For equal halves (``dist_l == dist_r``, every
  conforming octree face) this equals ``T_f`` above exactly, the factor of 2 in
  ``σ_face`` cancels the 2 in ``dist`` with NO leftover factor.

Each interior face stamps the symmetric ``±T`` quad. An optional symmetric
Dirichlet pin removes the pure-Neumann nullspace, mirroring both legacy
assemblers (the legacy ``area`` equals ``FaceRecords.area`` and
the record ordering matches index-for-index; see
:meth:`OctreeMesh.face_neighbors` / :meth:`TensorMesh.face_neighbors`). The eps
(``1e-30``), the float64-then-cast dtype handling, and the pin mechanic (drop
any quad entry touching the pin row/col, then append a single ``(pin, pin)=1``)
all match the legacy so the equivalence holds to ``atol=1e-12``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from geobrain.core.errors import GeoBrainError
from geobrain.mesh.capabilities import FaceRecords
from geobrain.mesh.operators import face_transmissibility


__all__ = ["assemble_poisson_fv"]

# The complete boundary-dict vocabulary. Validated on entry so a misspelt key
# (e.g. "robin_alpha") cannot silently drop a boundary condition.
_SUPPORTED_BOUNDARY_KEYS = frozenset(
    {"dirichlet_idx", "robin_cells", "robin_alpha_area"}
)


def assemble_poisson_fv(
    face_records: FaceRecords,
    sigma: torch.Tensor,
    *,
    boundary: dict | None = None,
    extra_diagonal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Assemble ``-∇·(σ∇)`` from face records as a torch sparse COO tensor.

    TPFA cell-centred FV with the series-harmonic two-point transmissibility in
    the per-side HALF-distances ``dist_l``/``dist_r`` (cell centre → face)::

        T_f = σ_l · σ_r · area / (σ_l · dist_r + σ_r · dist_l + eps)

    This is the single form that reproduces both legacy assemblers
    (:func:`assemble_poisson_3d` on TensorMesh and
    :func:`assemble_poisson_octree_3d`) to ``atol=1e-12`` (≈1e-15 in practice;
    the forms differ only by eps placement and float reassociation); see the
    module docstring.

    Each interior face stamps the symmetric ``±T`` quad. Autograd flows through
    ``sigma`` (the returned ``A.values()`` is a torch op chain on ``sigma``).

    Args:
        face_records: :class:`FaceRecords` over the ``nf`` internal faces of any
            :class:`ConnectivityMesh`. ``cell_i``/``cell_j`` index ``sigma``;
            ``area`` and ``dist_l``/``dist_r`` carry the geometry.
        sigma: cell-centred σ tensor of shape ``(n_cells,)``, ``float64`` (DC /
            IP) or ``complex128`` (SIP path with ``σ_eff(ω) = σ(1 − η(ω))``).
        boundary: ``None`` → pure-Neumann (singular, retains the nullspace);
            ``{"dirichlet_idx": k}`` → symmetric Dirichlet pin at cell ``k``
            (row/col of ``k`` zeroed, diagonal set to 1).

            A **mixed / Robin** absorbing-boundary contribution is requested
            with the two mutually-compatible keys
            ``{"robin_cells": LongTensor(n_bf,),
            "robin_alpha_area": Tensor(n_bf,)}``: for each listed boundary
            face the diagonal gains ``sigma[cell] · alpha_area`` (the
            far-field flux ``σ α u A`` with ``α·area`` passed PRE-multiplied so
            the seam stays pure assembly, works for real σ and complex
            ``σ_eff(ω)`` alike). Robin may be combined WITH the pin (the pin
            still anchors if every ``alpha_area`` happens to be 0) or used
            ALONE: a non-trivial Robin term makes ``A`` non-singular, so **no
            Dirichlet pin is required** when Robin is present. When the Robin
            keys are absent the assembled matrix is byte-identical to the
            pure-Neumann / pin-only paths.

            The dict is validated on entry: any key outside
            ``{"dirichlet_idx", "robin_cells", "robin_alpha_area"}`` raises,
            and ``robin_cells``/``robin_alpha_area`` must arrive together
            (either alone raises).
        extra_diagonal: optional ``(n_cells,)`` tensor added cell-by-cell onto
            the diagonal ``A[c, c] += extra_diagonal[c]``, the clean seam for a
            **cell-diagonal mass term** with NO off-diagonal coupling. The
            2.5-D DC wavenumber operator uses it for the Fourier mass term
            ``k²·σ_cell·V_cell`` (``A_k = A_TPFA(σ) + diag(k²·σ·V)``); a positive
            mass term also makes ``A`` non-singular, so the k-solve needs no
            pin. ``extra_diagonal`` must carry σ's autograd graph if the caller
            wants the gradient to flow through the mass term (the 2.5-D operator
            passes ``k²·σ·V`` with σ live). It is applied to EVERY listed cell
            EXCEPT the Dirichlet-pinned one (whose row stays the unit row).
            ``None`` (default) → byte-identical to the paths above.

    Dtype rule: the assembled values follow ``sigma``'s dtype. A real ``sigma``
    combined with a complex ``robin_alpha_area`` or ``extra_diagonal`` raises
    (the cast to ``sigma``'s dtype would silently discard the imaginary part);
    cast ``sigma`` to complex128 or pass the auxiliary input real.

    Returns:
        ``A`` as a coalesced torch sparse COO tensor of shape
        ``(n_cells, n_cells)``.
    """
    if sigma.ndim != 1:
        raise GeoBrainError(
            "assemble_poisson_fv: sigma must be 1-D (n_cells,)",
            object_name="assemble_poisson_fv",
            field="sigma.ndim",
            expected=1,
            actual=sigma.ndim,
        )
    n_cells = int(sigma.shape[0])
    device = sigma.device
    dtype = sigma.dtype
    eps = 1e-30  # matches assemble_poisson_octree_3d's harmonic-mean reg.

    # --- Boundary-dict entry validation -----------------------------------
    # Unknown keys raise (a typo like "robin_alpha" would otherwise silently
    # drop the boundary condition) and the two Robin keys must arrive paired
    # (a lone robin_alpha_area was previously ignored without a trace).
    if boundary is not None:
        unknown = set(boundary) - _SUPPORTED_BOUNDARY_KEYS
        if unknown:
            raise GeoBrainError(
                "assemble_poisson_fv: unknown boundary key(s) "
                f"{sorted(unknown)}",
                object_name="assemble_poisson_fv",
                field="boundary",
                expected=f"keys among {sorted(_SUPPORTED_BOUNDARY_KEYS)}",
                actual=sorted(boundary),
            )
        if ("robin_cells" in boundary) != ("robin_alpha_area" in boundary):
            raise GeoBrainError(
                "assemble_poisson_fv: robin_cells and robin_alpha_area must "
                "be passed together",
                object_name="assemble_poisson_fv",
                field="boundary",
                expected="both robin_cells and robin_alpha_area (or neither)",
                actual=sorted(set(boundary) & {"robin_cells", "robin_alpha_area"}),
            )

    rec_i = face_records.cell_i.to(device=device, dtype=torch.long)
    rec_j = face_records.cell_j.to(device=device, dtype=torch.long)

    # --- Robin / mixed absorbing-boundary diagonal contribution ----------
    # The two keys arrive together; each listed boundary face stamps
    # ``sigma[cell] · alpha_area`` onto the diagonal A[cell, cell]
    # (flux_out = σ α u A, with α·area pre-multiplied so this stays pure
    # assembly). Built once as a COO triple and appended into whichever return
    # path runs below; the additive value carries σ's autograd graph. ``None``
    # when the keys are absent: the matrix is then byte-identical to the
    # pure-Neumann / pin-only paths.
    has_robin = boundary is not None and "robin_cells" in boundary
    robin_cells: torch.Tensor | None = None
    robin_add: torch.Tensor | None = None
    if has_robin:
        robin_cells = boundary["robin_cells"].to(device=device, dtype=torch.long)
        alpha_area = boundary["robin_alpha_area"]
        if alpha_area.is_complex() and not sigma.is_complex():
            raise GeoBrainError(
                "assemble_poisson_fv: complex robin_alpha_area with real sigma "
                ", the assembled dtype follows sigma, so the imaginary part "
                "would be silently discarded. Cast sigma to complex128 or "
                "pass a real robin_alpha_area.",
                object_name="assemble_poisson_fv",
                field="robin_alpha_area.dtype",
                expected=f"real (assembly dtype follows sigma: {sigma.dtype})",
                actual=alpha_area.dtype,
            )
        if robin_cells.ndim != 1 or alpha_area.ndim != 1:
            raise GeoBrainError(
                "assemble_poisson_fv: robin_cells / robin_alpha_area must be 1-D",
                object_name="assemble_poisson_fv",
                field="robin_cells.ndim / robin_alpha_area.ndim",
                expected="both 1-D",
                actual=(robin_cells.ndim, alpha_area.ndim),
            )
        if int(robin_cells.numel()) != int(alpha_area.numel()):
            raise GeoBrainError(
                "assemble_poisson_fv: robin_cells and robin_alpha_area "
                "length mismatch",
                object_name="assemble_poisson_fv",
                field="robin_cells/robin_alpha_area",
                expected=int(robin_cells.numel()),
                actual=int(alpha_area.numel()),
            )
        if int(robin_cells.numel()) and (
            int(robin_cells.min()) < 0 or int(robin_cells.max()) >= n_cells
        ):
            raise GeoBrainError(
                "assemble_poisson_fv: robin_cells index out of range",
                object_name="assemble_poisson_fv",
                field="robin_cells",
                expected=f"[0, {n_cells})",
                actual="out-of-range",
            )
        # ``sigma[cell] · alpha_area`` per listed face (real or complex σ); the
        # multiply by σ keeps the diagonal additions autograd-attached.
        robin_add = sigma[robin_cells] * alpha_area.to(device=device, dtype=dtype)

    # --- Cell-diagonal mass / extra-diagonal term (the 2.5-D Fourier seam) ---
    # ``extra_diagonal`` (when given) is a per-cell additive diagonal with NO
    # off-diagonal coupling: ``A[c, c] += extra_diagonal[c]``. Stamped as one
    # ``(c, c)`` COO triple per cell and summed into the diagonal by
    # ``coalesce``. The value carries whatever autograd graph the caller built
    # (the 2.5-D operator passes ``k²·σ·V`` with σ live, so the σ-gradient flows
    # through the mass term). ``None`` → no entries, byte-identical to before.
    has_extra = extra_diagonal is not None
    extra_cells: torch.Tensor | None = None
    extra_add: torch.Tensor | None = None
    if has_extra:
        if extra_diagonal.ndim != 1 or int(extra_diagonal.shape[0]) != n_cells:
            raise GeoBrainError(
                "assemble_poisson_fv: extra_diagonal must be 1-D (n_cells,)",
                object_name="assemble_poisson_fv",
                field="extra_diagonal.shape",
                expected=(n_cells,),
                actual=tuple(extra_diagonal.shape),
            )
        if extra_diagonal.is_complex() and not sigma.is_complex():
            raise GeoBrainError(
                "assemble_poisson_fv: complex extra_diagonal with real sigma "
                ", the assembled dtype follows sigma, so the imaginary part "
                "would be silently discarded. Cast sigma to complex128 or "
                "pass a real extra_diagonal.",
                object_name="assemble_poisson_fv",
                field="extra_diagonal.dtype",
                expected=f"real (assembly dtype follows sigma: {sigma.dtype})",
                actual=extra_diagonal.dtype,
            )
        extra_cells = torch.arange(n_cells, dtype=torch.long, device=device)
        extra_add = extra_diagonal.to(device=device, dtype=dtype)

    def _extra_coo(
        drop_pin: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(rows, cols, vals)`` of the extra-diagonal entries ``(c, c)`` ↦
        ``extra_diagonal[c]``. ``coalesce`` later sums these into the existing
        diagonal. When ``drop_pin`` is given (pin branch) the pinned cell's
        entry is dropped so the pinned row stays exactly the unit row.
        """
        assert extra_cells is not None and extra_add is not None
        cells, vals = extra_cells, extra_add
        if drop_pin is not None:
            keep = cells != drop_pin
            cells, vals = cells[keep], vals[keep]
        return cells, cells, vals

    def _robin_coo(
        drop_pin: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(rows, cols, vals)`` of the Robin diagonal entries, ``(cell, cell)``
        ↦ ``sigma[cell]·alpha_area``. ``coalesce`` later sums these into the
        existing diagonal. When ``drop_pin`` is given (pin branch) any face on
        the pinned cell is dropped so the pinned row stays exactly the unit row.
        """
        assert robin_cells is not None and robin_add is not None
        cells, vals = robin_cells, robin_add
        if drop_pin is not None:
            keep = cells != drop_pin
            cells, vals = cells[keep], vals[keep]
        return cells, cells, vals

    has_pin = boundary is not None and "dirichlet_idx" in boundary

    def _pin_only() -> torch.Tensor:
        """Degenerate / no-face mesh under a Dirichlet pin (optionally + Robin).

        The pin row is a single ``(pin, pin)=1``; any Robin face diagonals on
        OTHER cells are appended (a Robin face on the pinned cell is dropped so
        the pinned row stays the unit row).
        """
        pin = int(boundary["dirichlet_idx"])
        if not (0 <= pin < n_cells):
            raise GeoBrainError(
                "assemble_poisson_fv: dirichlet_idx out of range",
                object_name="assemble_poisson_fv",
                field="dirichlet_idx",
                expected=f"[0, {n_cells})",
                actual=pin,
            )
        pin_idx = torch.tensor([pin], dtype=torch.long, device=device)
        pin_val = torch.ones(1, dtype=dtype, device=device)
        rows_p, cols_p, vals_p = [pin_idx], [pin_idx], [pin_val]
        if has_robin:
            r_rows, r_cols, r_vals = _robin_coo(drop_pin=pin)
            rows_p.append(r_rows)
            cols_p.append(r_cols)
            vals_p.append(r_vals)
        if has_extra:
            e_rows, e_cols, e_vals = _extra_coo(drop_pin=pin)
            rows_p.append(e_rows)
            cols_p.append(e_cols)
            vals_p.append(e_vals)
        return torch.sparse_coo_tensor(
            torch.stack([torch.cat(rows_p), torch.cat(cols_p)], dim=0),
            torch.cat(vals_p),
            size=(n_cells, n_cells),
        ).coalesce()

    if rec_i.numel() == 0:
        if has_pin:
            return _pin_only()
        if has_robin or has_extra:
            # Robin / mass-term alone on a no-internal-face mesh: only the
            # diagonal additions (Robin far-field flux and/or the extra mass
            # diagonal). Either alone makes A non-singular.
            rows_d: list[torch.Tensor] = []
            cols_d: list[torch.Tensor] = []
            vals_d: list[torch.Tensor] = []
            if has_robin:
                r_rows, r_cols, r_vals = _robin_coo()
                rows_d.append(r_rows)
                cols_d.append(r_cols)
                vals_d.append(r_vals)
            if has_extra:
                e_rows, e_cols, e_vals = _extra_coo()
                rows_d.append(e_rows)
                cols_d.append(e_cols)
                vals_d.append(e_vals)
            return torch.sparse_coo_tensor(
                torch.stack([torch.cat(rows_d), torch.cat(cols_d)], dim=0),
                torch.cat(vals_d),
                size=(n_cells, n_cells),
            ).coalesce()
        return torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=dtype, device=device),
            size=(n_cells, n_cells),
        ).coalesce()

    # Series-harmonic two-point transmissibility expressed directly in the
    # FaceRecords HALF-distances ``dist_l``/``dist_r`` (cell centre → face).
    # Computed by the shared core operator factory with a bit-mirrored op
    # order (``σ_l·σ_r·area / (σ_l·dist_r + σ_r·dist_l + eps)``): see
    # :func:`geobrain.mesh.operators.face_transmissibility` and the
    # byte-equality gate in the operator-factory equivalence suite.
    # This is the SINGLE form that reproduces BOTH legacy assemblers to
    # atol=1e-12 (≈1e-15 in practice; the forms differ only by eps placement
    # and float reassociation: see module docstring):
    #
    #   * assemble_poisson_3d (TensorMesh) uses full cell widths
    #     ``T = 2·σ_l·σ_r·area / (σ_l·D_r + σ_r·D_l)`` with D = 2·dist; the
    #     factor of 2 cancels the 2 from the half→full doubling, leaving the
    #     series form with NO factor of 2.
    #   * assemble_poisson_octree_3d uses ``σ_face·area/dist`` with the
    #     harmonic mean ``σ_face = 2σ_lσ_r/(σ_l+σ_r)`` and full
    #     ``dist = dist_l + dist_r``; for equal halves (dist_l==dist_r) this
    #     equals the series form exactly.
    T = face_transmissibility(face_records, sigma, eps=eps)

    rows_quad = torch.cat([rec_i, rec_j, rec_i, rec_j])
    cols_quad = torch.cat([rec_i, rec_j, rec_j, rec_i])
    vals_quad = torch.cat([+T, +T, -T, -T])

    if not has_pin:
        # Pure-Neumann (boundary None) OR Robin-alone OR mass-term-alone (the
        # 2.5-D Fourier seam). The interior quad is byte-identical to the legacy;
        # Robin and/or extra-diagonal additions (if any) are appended and summed
        # into the diagonal by ``coalesce``.
        rows_parts = [rows_quad]
        cols_parts = [cols_quad]
        vals_parts = [vals_quad]
        if has_robin:
            r_rows, r_cols, r_vals = _robin_coo()
            rows_parts.append(r_rows)
            cols_parts.append(r_cols)
            vals_parts.append(r_vals)
        if has_extra:
            e_rows, e_cols, e_vals = _extra_coo()
            rows_parts.append(e_rows)
            cols_parts.append(e_cols)
            vals_parts.append(e_vals)
        rows_all = torch.cat(rows_parts) if len(rows_parts) > 1 else rows_quad
        cols_all = torch.cat(cols_parts) if len(cols_parts) > 1 else cols_quad
        vals_all = torch.cat(vals_parts) if len(vals_parts) > 1 else vals_quad
        return torch.sparse_coo_tensor(
            torch.stack([rows_all, cols_all], dim=0),
            vals_all,
            size=(n_cells, n_cells),
        ).coalesce()

    pin = int(boundary["dirichlet_idx"])
    if not (0 <= pin < n_cells):
        raise GeoBrainError(
            "assemble_poisson_fv: dirichlet_idx out of range",
            object_name="assemble_poisson_fv",
            field="dirichlet_idx",
            expected=f"[0, {n_cells})",
            actual=pin,
        )
    # Symmetric Dirichlet pin: drop any quad entry whose row OR column touches
    # the pin, then restore a single (pin, pin)=1: identical to the legacy.
    keep = (rows_quad != pin) & (cols_quad != pin)
    pin_idx = torch.tensor([pin], dtype=torch.long, device=device)
    pin_val = torch.ones(1, dtype=dtype, device=device)
    rows_all = torch.cat([rows_quad[keep], pin_idx])
    cols_all = torch.cat([cols_quad[keep], pin_idx])
    vals_all = torch.cat([vals_quad[keep], pin_val])
    if has_robin:
        # Robin combined with a pin: append the Robin diagonals on every cell
        # EXCEPT the pin (whose row must remain the unit row).
        r_rows, r_cols, r_vals = _robin_coo(drop_pin=pin)
        rows_all = torch.cat([rows_all, r_rows])
        cols_all = torch.cat([cols_all, r_cols])
        vals_all = torch.cat([vals_all, r_vals])
    if has_extra:
        # Extra mass diagonal combined with a pin: append on every cell EXCEPT
        # the pin (whose row must remain the unit row).
        e_rows, e_cols, e_vals = _extra_coo(drop_pin=pin)
        rows_all = torch.cat([rows_all, e_rows])
        cols_all = torch.cat([cols_all, e_cols])
        vals_all = torch.cat([vals_all, e_vals])
    return torch.sparse_coo_tensor(
        torch.stack([rows_all, cols_all], dim=0),
        vals_all,
        size=(n_cells, n_cells),
    ).coalesce()
