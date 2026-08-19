"""MCMC chain diagnostics: effective sample size, split-R-hat, summaries, and a
minimal multi-chain runner.

Multi-chain convention (THE library convention)
-----------------------------------------------
Stacked multi-chain sample tensors are shaped ``(n_chains, n_draws,
*field_shape)``, chains first, draws second. A single chain is ``(n_draws,
*field_shape)`` and is treated as one chain wherever a function accepts both.
:func:`run_chains` produces the stacked layout; the per-sampler ``run()``
methods produce the single-chain layout.

Because shape alone cannot distinguish ``(n_draws, *field_shape)`` from
``(n_chains, n_draws, ...)`` (both are >= 2-D), the diagnostics take an
explicit ``chains`` keyword:

- :func:`ess` defaults to ``chains=False`` (input is one chain, matching what
  a single ``Sampler.run`` returns);
- :func:`split_rhat` defaults to ``chains=True`` (input is a chain-major
  stack, matching what :func:`run_chains` returns; a 1-D input is
  auto-promoted to one chain and split in half).

Estimator choices (documented, deliberate)
------------------------------------------
- **ESS** uses FFT autocovariance with Geyer's initial-positive-sequence
  truncation. Multi-chain inputs are **pooled Vehtari/Stan-style** (Vehtari et
  al. 2021, eq. 10; Stan reference manual "Effective Sample Size"): the lag-t
  correlation is ``rho_t = 1 - (W - mean_c acov_c(t)) / var_plus`` where ``W``
  is the mean within-chain variance and ``var_plus = W (N-1)/N + B/N``
  combines within- and between-chain variance; NOT a mean of per-chain ESS
  estimates. For one chain the formula reduces to (essentially) the plain ACF
  estimator the examples used. Only the initial-positive truncation is
  applied (Geyer 1992); Stan's additional initial-monotone smoothing is not.
- **split_rhat** is the CLASSIC split-R-hat (Gelman-Rubin with every chain
  split in half; Gelman et al., BDA3 / Vehtari et al. 2021 eqs. 3-4). The
  rank-normalized / folded variants of Vehtari et al. (2021) are NOT
  implemented.

All statistics are computed internally in float64.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence, cast

import torch

from ..core import ForwardContext
from ..core.errors import GeoBrainError
from ..core.validation import validate_int
from .base import Sampler
from .execution import SamplerStopReason
from .results import InferenceResult

__all__ = ["ess", "split_rhat", "summarize", "run_chains"]


# =========================================================================
# shape handling
# =========================================================================


def _as_chain_major(
    samples: torch.Tensor, chains: bool, owner: str
) -> torch.Tensor:
    """Validate and reshape ``samples`` to float64 ``(M, N, E)``.

    ``chains=False``: input is one chain ``(n_draws, *field_shape)``.
    ``chains=True``: input is stacked ``(n_chains, n_draws, *field_shape)``.
    Returns a view/copy flattened over the field dims (``E = prod(shape)``,
    with ``E = 1`` for scalar chains).
    """
    if not isinstance(samples, torch.Tensor):
        raise GeoBrainError(
            f"{owner} expects a torch.Tensor of samples",
            object_name=owner,
            field="samples",
            expected=torch.Tensor,
            actual=type(samples),
        )
    if chains:
        if samples.ndim < 2:
            raise GeoBrainError(
                f"{owner} with chains=True expects (n_chains, n_draws, "
                f"*field_shape); got a {samples.ndim}-D tensor",
                object_name=owner,
                field="samples",
                expected=">= 2-D (n_chains, n_draws, ...)",
                actual=tuple(samples.shape),
            )
        x = samples
    else:
        if samples.ndim < 1:
            raise GeoBrainError(
                f"{owner} expects (n_draws, *field_shape); got a 0-D tensor",
                object_name=owner,
                field="samples",
                expected=">= 1-D (n_draws, ...)",
                actual=tuple(samples.shape),
            )
        x = samples.unsqueeze(0)  # one chain
    m, n = x.shape[0], x.shape[1]
    return x.reshape(m, n, -1).to(torch.float64)


def _field_shape(samples: torch.Tensor, chains: bool) -> tuple[int, ...]:
    return tuple(samples.shape[2:] if chains else samples.shape[1:])


# =========================================================================
# effective sample size
# =========================================================================


def ess(samples: torch.Tensor, *, chains: bool = False) -> torch.Tensor:
    """Effective sample size via FFT autocorrelation + Geyer truncation.

    The estimator the ``examples/09_bayesian`` scripts previously each
    reimplemented inline, promoted to the library: per-chain autocovariance by
    FFT, Geyer (1992) initial-positive-sequence truncation of the paired
    autocorrelation sums, ``ESS = M N / tau`` with
    ``tau = -1 + 2 sum_k (rho_{2k} + rho_{2k+1})``.

    Args:
        samples: ``(n_draws,)`` or ``(n_draws, *field_shape)`` for a single
            chain (default), or ``(n_chains, n_draws, *field_shape)`` with
            ``chains=True`` (the library's stacked multi-chain convention;
            see the module docstring). Per-element ESS is returned over the
            trailing field dims.
        chains: whether the leading dim indexes chains (``True``) or draws
            (``False``, default).

    Returns:
        float64 tensor shaped like the field (``()`` for scalar chains).
        Multi-chain inputs are pooled Vehtari/Stan-style (within+between
        variance through ``var_plus``; see module docstring) so the result is
        one ESS for all chains combined, not a per-chain mean. Elements with
        zero variance return NaN.

    Raises:
        GeoBrainError: on a non-tensor input, wrong dimensionality, or fewer
            than 4 draws.
    """
    x = _as_chain_major(samples, chains, "ess")
    m, n, _e = x.shape
    if n < 4:
        raise GeoBrainError(
            "ess needs at least 4 draws per chain",
            object_name="ess",
            field="samples",
            expected="n_draws >= 4",
            actual=n,
        )

    # Per-chain autocovariance by FFT (biased, divided by N: the Stan
    # convention). Zero-pad to >= 2N so circular correlation cannot wrap.
    xc = x - x.mean(dim=1, keepdim=True)          # remove per-chain mean
    xc = xc.permute(0, 2, 1)                       # (M, E, N): FFT along draws
    nfft = 1 << (2 * n - 1).bit_length()
    f = torch.fft.rfft(xc, n=nfft, dim=-1)
    acov = torch.fft.irfft(f * f.conj(), n=nfft, dim=-1)[..., :n] / n  # (M,E,N)

    mean_acov = acov.mean(dim=0)                   # (E, N)
    # W: mean within-chain (unbiased) variance; B/N: variance of chain means.
    w = (acov[..., 0] * (n / (n - 1))).mean(dim=0)  # (E,)
    var_plus = w * (n - 1) / n
    if m > 1:
        chain_means = x.mean(dim=1)                # (M, E)
        var_plus = var_plus + chain_means.var(dim=0, unbiased=True)

    # rho_t = 1 - (W - mean_acov_t) / var_plus, with rho_0 := 1 exactly.
    rho = 1.0 - (w.unsqueeze(-1) - mean_acov) / var_plus.unsqueeze(-1)
    rho[..., 0] = 1.0

    # Geyer initial positive sequence over pair sums P_k = rho_{2k}+rho_{2k+1}.
    n_even = n - (n % 2)
    pairs = rho[..., :n_even].reshape(rho.shape[0], -1, 2).sum(dim=-1)  # (E, K)
    keep = (pairs > 0).to(torch.float64).cumprod(dim=-1)
    tau = -1.0 + 2.0 * (pairs * keep).sum(dim=-1)
    # Guard: tau <= 0 can only arise from rounding on an extremely antithetic
    # chain (rho_1 ~ -1); clamp preserves NaN from zero-variance elements.
    tau = tau.clamp(min=torch.finfo(torch.float64).eps)

    out = (m * n) / tau
    return out.reshape(_field_shape(samples, chains))


# =========================================================================
# split-R-hat
# =========================================================================


def split_rhat(samples: torch.Tensor, *, chains: bool = True) -> torch.Tensor:
    """Classic split-R-hat (Gelman-Rubin with each chain split in half).

    Each of the M chains of length N is split into two halves (the last draw
    is dropped when N is odd), giving 2M chains of length ``N // 2``; then the
    standard potential-scale-reduction statistic is computed:
    ``R-hat = sqrt(var_plus / W)`` with ``var_plus = W (n-1)/n + B/n``.
    This is the CLASSIC estimator, the rank-normalized/folded R-hat of
    Vehtari et al. (2021) is not implemented.

    Args:
        samples: ``(n_chains, n_draws, *field_shape)`` (default; the library's
            stacked multi-chain convention). A 1-D ``(n_draws,)`` tensor is
            auto-promoted to one chain, i.e. the single chain is split in two
            halves. Pass ``chains=False`` to treat a >= 2-D input as one chain
            ``(n_draws, *field_shape)``.
        chains: whether the leading dim indexes chains (default ``True``;
            1-D inputs are always a single chain).

    Returns:
        float64 tensor shaped like the field (``()`` for scalar chains).
        Elements with zero within-chain variance return NaN (or +inf when the
        chains are distinct constants).

    Raises:
        GeoBrainError: on a non-tensor input, wrong dimensionality, or fewer
            than 4 draws per chain (each half needs >= 2).
    """
    if isinstance(samples, torch.Tensor) and samples.ndim == 1:
        chains = False  # (n_draws,): one chain, split in half below
    x = _as_chain_major(samples, chains, "split_rhat")
    m, n, e = x.shape
    if n < 4:
        raise GeoBrainError(
            "split_rhat needs at least 4 draws per chain "
            "(each split half needs >= 2)",
            object_name="split_rhat",
            field="samples",
            expected="n_draws >= 4",
            actual=n,
        )

    half = n // 2
    halves = torch.cat([x[:, :half], x[:, half: 2 * half]], dim=0)  # (2M,h,E)
    chain_means = halves.mean(dim=1)                                # (2M, E)
    w = halves.var(dim=1, unbiased=True).mean(dim=0)                # (E,)
    b_over_n = chain_means.var(dim=0, unbiased=True)                # B / n
    var_plus = w * (half - 1) / half + b_over_n
    out = torch.sqrt(var_plus / w)
    return out.reshape(_field_shape(samples, chains))


# =========================================================================
# summaries
# =========================================================================


def _json_safe(t: torch.Tensor) -> float | list[Any]:
    """0-d tensor -> float; n-d tensor -> (nested) list of floats."""
    if t.ndim == 0:
        return float(t.item())
    return cast(list[Any], t.tolist())


def summarize(
    result_or_samples: "InferenceResult | Mapping[str, torch.Tensor]",
    *,
    chains: bool | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-field posterior summary: mean / std / quantiles / ESS / R-hat.

    Args:
        result_or_samples: an :class:`InferenceResult` or a plain
            ``{field: tensor}`` mapping. Tensors follow the library
            convention, ``(n_draws, *field_shape)`` single chain or
            ``(n_chains, n_draws, *field_shape)`` stacked (``chains=True``).
        chains: leading-dim interpretation. ``None`` (default) infers it:
            ``True`` for an :class:`InferenceResult` carrying the
            ``"n_chains"`` metadata key (what :func:`run_chains` emits),
            ``False`` otherwise (a single sampler's output).

    Returns:
        ``{field: stats}`` where ``stats`` holds JSON-safe scalars/lists:
        ``mean``, ``std``, ``q05`` / ``q50`` / ``q95``, ``ess``, ``n_draws``
        (total pooled draws), plus ``rhat`` and ``n_chains`` for multi-chain
        input. Statistics are per field element (scalars for scalar fields,
        nested lists matching ``field_shape`` otherwise). ``ess`` is ``None``
        when there are fewer than 4 draws per chain.
    """
    if isinstance(result_or_samples, InferenceResult):
        samples = result_or_samples.samples
        if chains is None:
            chains = "n_chains" in result_or_samples.metadata
    else:
        samples = result_or_samples
        if chains is None:
            chains = False
    if not isinstance(samples, Mapping):
        raise GeoBrainError(
            "summarize expects an InferenceResult or a {field: tensor} mapping",
            object_name="summarize",
            field="result_or_samples",
            expected="InferenceResult | Mapping[str, Tensor]",
            actual=type(result_or_samples),
        )

    out: dict[str, dict[str, Any]] = {}
    for name, tensor in samples.items():
        x = _as_chain_major(tensor, chains, "summarize")   # (M, N, E)
        m, n, _e = x.shape
        shape = _field_shape(tensor, chains)
        stats: dict[str, Any] = {"n_draws": m * n}
        if chains:
            stats["n_chains"] = m
        if n == 0:
            out[name] = stats
            continue

        pooled = x.reshape(m * n, -1)                      # (M*N, E)
        q = torch.quantile(
            pooled,
            torch.tensor(
                [0.05, 0.50, 0.95], dtype=torch.float64, device=pooled.device
            ),
            dim=0,
        )
        stats["mean"] = _json_safe(pooled.mean(dim=0).reshape(shape))
        stats["std"] = _json_safe(
            pooled.std(dim=0, unbiased=True).reshape(shape)
            if m * n > 1
            else torch.full(shape or (), float("nan"), dtype=torch.float64)
        )
        stats["q05"] = _json_safe(q[0].reshape(shape))
        stats["q50"] = _json_safe(q[1].reshape(shape))
        stats["q95"] = _json_safe(q[2].reshape(shape))
        stats["ess"] = (
            _json_safe(ess(tensor, chains=chains)) if n >= 4 else None
        )
        if chains and m > 1:
            stats["rhat"] = (
                _json_safe(split_rhat(tensor, chains=True)) if n >= 4 else None
            )
        out[name] = stats
    return out


# =========================================================================
# multi-chain runner
# =========================================================================


def run_chains(
    sampler_factory: Callable[[int, torch.Generator | None], Sampler],
    n_chains: int,
    n_iters: int,
    ctx: ForwardContext | None = None,
    *,
    seeds: Sequence[int] | None = None,
    generators: Sequence[torch.Generator] | None = None,
    n_workers: int | None = None,
) -> InferenceResult:
    """Run ``n_chains`` independent chains and stack them.

    The minimal multi-chain entry point that makes :func:`split_rhat` and
    pooled :func:`ess` usable: builds one sampler per chain from
    ``sampler_factory``, runs each for ``n_iters``, and stacks the per-field
    samples to the library's ``(n_chains, n_draws, *field_shape)`` convention.

    Args:
        sampler_factory: ``(chain_index, generator) -> Sampler``. Called once
            per chain with the chain's index (0-based; use it to disperse
            initial points) and its RNG (``None`` when neither ``seeds`` nor
            ``generators`` is given, chains then differ only through the
            global RNG). Sampler-constructor knobs, including the MCMC
            chain-storage / burn-in controls ``warmup`` / ``thin`` /
            ``store_dtype``: pass through here: build each sampler with
            them and every chain records ``floor(n_iters / thin)``
            post-warmup draws (give all chains the SAME ``warmup`` / ``thin``
            so the per-chain draw counts agree and stack).
        n_chains: number of chains (>= 1).
        n_iters: iterations per chain (>= 0).
        ctx: optional :class:`ForwardContext` forwarded to every ``run``.
        seeds: per-chain integer seeds (length ``n_chains``); mutually
            exclusive with ``generators``.
        generators: per-chain ``torch.Generator`` objects (length
            ``n_chains``); mutually exclusive with ``seeds``.
        n_workers: ``None`` (default) runs the chains sequentially, exactly as
            before. An integer ``>= 1`` runs them in a thread pool of that
            size (capped at ``n_chains``). Torch releases the GIL inside its
            kernels, so chains dominated by tensor math (PDE forwards) overlap
            effectively; the per-chain intra-op thread count still applies, so
            consider ``torch.set_num_threads`` to avoid oversubscription. Every
            sampler is CONSTRUCTED on the calling thread in chain order (so a
            factory drawing from the global RNG stays deterministic); only
            ``run`` executes in the pool. With per-chain ``seeds`` /
            ``generators`` the results are bitwise identical to the sequential
            path (samplers draw exclusively from their injected generator);
            chains sharing the global RNG remain statistically valid but are
            no longer reproducible (true of any global-RNG chain). On a
            failure, the exception of the LOWEST failing chain index is
            raised (matching sequential error identity) after already-started
            chains finish; any ``partial_result`` the sampler attached rides
            the raised error unchanged.

    Returns:
        An :class:`InferenceResult` whose ``samples`` are stacked
        ``(n_chains, n_draws, *field_shape)`` tensors and whose
        ``log_post_history`` is stacked ``(n_chains, n_draws)`` (2-D, one row
        per chain, unlike the 1-D single-chain contract). ``acceptance_rate``
        is the mean of the per-chain rates; ``metadata`` carries
        ``n_chains``, ``chain_acceptance_rates`` (per-chain list) and
        ``chain_metadata`` (per-chain metadata dicts).

    Raises:
        GeoBrainError: on invalid ``n_chains`` / ``n_iters``, both ``seeds``
            and ``generators`` given, wrong sequence lengths, or chains whose
            sample fields / shapes disagree.
    """
    n_chains = validate_int(
        n_chains, owner="run_chains", field="n_chains",
        minimum=0, minimum_inclusive=False,
    )
    n_iters = validate_int(n_iters, owner="run_chains", field="n_iters", minimum=0)
    if seeds is not None and generators is not None:
        raise GeoBrainError(
            "run_chains takes seeds OR generators, not both",
            object_name="run_chains",
            field="seeds/generators",
            expected="at most one of the two",
            actual="both",
        )
    gens: list[torch.Generator | None]
    if seeds is not None:
        if len(seeds) != n_chains:
            raise GeoBrainError(
                "run_chains seeds length must equal n_chains",
                object_name="run_chains",
                field="seeds",
                expected=f"length {n_chains}",
                actual=len(seeds),
            )
        gens = [torch.Generator().manual_seed(int(s)) for s in seeds]
    elif generators is not None:
        if len(generators) != n_chains:
            raise GeoBrainError(
                "run_chains generators length must equal n_chains",
                object_name="run_chains",
                field="generators",
                expected=f"length {n_chains}",
                actual=len(generators),
            )
        gens = list(generators)
    else:
        gens = [None] * n_chains

    if n_workers is not None:
        n_workers = validate_int(
            n_workers, owner="run_chains", field="n_workers", minimum=1
        )

    # Construct every sampler on the calling thread, in chain order, factory
    # side effects (global-RNG init dispersal, registry lookups) stay
    # deterministic regardless of how the runs are scheduled below.
    samplers: list[Sampler] = []
    for c in range(n_chains):
        sampler = sampler_factory(c, gens[c])
        if not hasattr(sampler, "run"):
            raise GeoBrainError(
                "run_chains sampler_factory must return a Sampler",
                object_name="run_chains",
                field="sampler_factory",
                expected="object with .run(n_iters, ctx)",
                actual=type(sampler),
            )
        samplers.append(sampler)

    results: list[InferenceResult] = []
    if n_workers is None:
        for sampler in samplers:
            results.append(sampler.run(n_iters, ctx))
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(n_workers, n_chains)) as pool:
            futures = [pool.submit(s.run, n_iters, ctx) for s in samplers]
            # Collect in chain-index order: the lowest failing index raises,
            # matching the sequential path's error identity.
            results = [f.result() for f in futures]

    fields = list(results[0].samples.keys())
    for c, res in enumerate(results[1:], start=1):
        if list(res.samples.keys()) != fields:
            raise GeoBrainError(
                "run_chains chains returned different sample fields",
                object_name="run_chains",
                field="samples",
                expected=fields,
                actual=list(res.samples.keys()),
            )
        for name in fields:
            if res.samples[name].shape != results[0].samples[name].shape:
                raise GeoBrainError(
                    "run_chains chains returned different sample shapes",
                    object_name="run_chains",
                    field=f"samples[{name!r}]",
                    expected=tuple(results[0].samples[name].shape),
                    actual=tuple(res.samples[name].shape),
                )

    samples = {
        name: torch.stack([r.samples[name] for r in results], dim=0)
        for name in fields
    }
    log_post = torch.stack([r.log_post_history for r in results], dim=0)
    acc = [float(r.acceptance_rate) for r in results]
    return InferenceResult(
        samples=samples,
        log_post_history=log_post,
        acceptance_rate=sum(acc) / n_chains if n_iters > 0 else 0.0,
        requested_iters=n_iters,
        completed_iters=n_iters,
        stop_reason=SamplerStopReason.COMPLETED,
        metadata={
            "sample_layout": "chains",
            "stored_draws": int(next(iter(samples.values())).shape[1]),
            "n_chains": n_chains,
            "chain_acceptance_rates": acc,
            "chain_metadata": [dict(r.metadata) for r in results],
            **({"n_workers": n_workers} if n_workers is not None else {}),
        },
    )
