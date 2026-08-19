# pyright: reportPrivateImportUsage=false
"""
Neural network layers for GeoBrain.

Provides custom layer implementations including basic utility layers
and Bayesian neural network layers using the Flipout reparameterization
technique (Wen et al. 2018, ICLR) for efficient weight sampling.

:class:`LinearFlipout` is the single canonical Flipout linear layer; its
:class:`Conv2dFlipout` / :class:`Conv3dFlipout` siblings share a private
:class:`_ConvNdFlipout` base, and all three derive from
:class:`BaseVariationalLayer`. Every layer's ``forward`` returns a plain
tensor (so it composes with :class:`torch.nn.Sequential`); the KL
divergence is obtained only via :meth:`~BaseVariationalLayer.kl_div` /
each layer's ``kl_loss``, never as a forward tuple.

KL divergence across a whole model is aggregated with :func:`get_kl_loss`
(walks ``model.modules()`` and sums every :class:`BaseVariationalLayer`
leaf). The standalone :func:`gaussian_kl` exposes the same closed-form
Gaussian KL as a free function for callers that want it directly.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from geobrain.core.errors import GeoBrainError

__all__ = [
    "Reshape",
    "BaseVariationalLayer",
    "LinearFlipout",
    "Conv2dFlipout",
    "Conv3dFlipout",
    "gaussian_kl",
    "get_kl_loss",
    "kl_regularizer",
    "count_variational_parameters",
]


def _gaussian_kl_terms(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
) -> torch.Tensor:
    """Element-wise ``KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2))`` (unreduced).

    The single source of the closed-form Gaussian-KL integrand, shared by the
    :func:`gaussian_kl` free function and :meth:`BaseVariationalLayer.kl_div`
    (which differ only in the final reduction).
    """
    return (
        torch.log(sigma_p)
        - torch.log(sigma_q)
        + (sigma_q.pow(2) + (mu_q - mu_p).pow(2)) / (2 * sigma_p.pow(2))
        - 0.5
    )


def gaussian_kl(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
) -> torch.Tensor:
    """
    Closed-form ``KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2))``.

    Computed element-wise and summed over all entries of the (broadcast-
    compatible) inputs. All four arguments are tensors so the same routine
    works for both weights and biases without scalar/tensor branching.

    This is the free-function form of
    :meth:`BaseVariationalLayer.kl_div` with ``reduction="sum"``.

    Args:
        mu_q, sigma_q: Posterior mean and (positive) standard deviation.
        mu_p, sigma_p: Prior mean and (positive) standard deviation.

    Returns:
        Zero-dim tensor; the summed KL.
    """
    return _gaussian_kl_terms(mu_q, sigma_q, mu_p, sigma_p).sum()


class Reshape(nn.Module):
    """
    Reshape layer for changing tensor dimensions.

    Args:
        *args: Target shape for the tensor (including batch dimension).

    Example:
        >>> reshape = Reshape(-1, 16, 8, 8)
        >>> x = torch.randn(32, 1024)
        >>> output = reshape(x)
        >>> print(output.shape)  # torch.Size([32, 16, 8, 8])
    """

    def __init__(self, *args: int) -> None:
        super().__init__()
        self.shape = args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape input tensor.

        Uses ``Tensor.reshape`` rather than ``Tensor.view`` so that
        non-contiguous inputs (post-transpose / slice) work without an
        explicit ``.contiguous()`` call upstream.
        """
        return x.reshape(self.shape)


class BaseVariationalLayer(nn.Module, ABC):
    """
    Base class for variational Bayesian layers.

    Implements common functionality for computing the KL divergence
    between posterior and prior distributions, shared by every Flipout
    layer (linear + conv).

    Every concrete subclass must implement :meth:`kl_loss`; it is the
    contract :func:`get_kl_loss` relies on when aggregating KL across a
    model. Declaring it ``@abstractmethod`` enforces that at the type
    boundary, a subclass that forgets the override fails to instantiate
    (``TypeError``) rather than surfacing an ``AttributeError`` only when
    aggregation walks the module.
    """

    def __init__(self) -> None:
        super().__init__()
        self._generator: torch.Generator | None = None
        # Lazily-built per-device generators derived from ``self._generator``,
        # keyed by ``torch.device``. Populated on demand by ``_generator_for``
        # so a CPU-seeded layer can draw device-native randomness on CUDA
        # (PyTorch forbids a device-mismatched generator) without disturbing
        # CPU behaviour or seed reproducibility.
        self._device_generators: dict[torch.device, torch.Generator] = {}

    @abstractmethod
    def kl_loss(self) -> torch.Tensor:  # pragma: no cover
        """KL divergence of this layer's posterior vs. prior (summed).

        Concrete layers return the total ``KL(Q || P)`` over their
        variational weights (and bias, if present); :func:`get_kl_loss`
        sums this across every :class:`BaseVariationalLayer` in a model.
        """
        ...

    def _generator_for(self, device: torch.device) -> torch.Generator | None:
        """Generator matching ``device`` (or ``None`` for the global RNG).

        A seeded layer stores a single injected generator (``self._generator``)
        whose device is fixed at construction (typically CPU). PyTorch requires
        a random op's generator to live on the op's device, so when ``device``
        differs (e.g. the layer was moved to CUDA) we lazily create, and cache,
        a device-native generator seeded deterministically from the injected
        generator's seed. When the devices match, the injected generator is
        returned unchanged, so CPU behaviour and seed reproducibility are
        byte-for-byte identical. If no generator was injected the global RNG is
        used (``None``), exactly as before.
        """
        if self._generator is None:
            return None
        if self._generator.device == device:
            return self._generator
        cached = self._device_generators.get(device)
        if cached is None:
            cached = torch.Generator(device=device)
            cached.manual_seed(self._generator.initial_seed())
            self._device_generators[device] = cached
        return cached

    def _randn(self, ref: torch.Tensor) -> torch.Tensor:
        """``randn`` like ``ref`` honouring the injected generator (if any).

        ``torch.randn_like`` takes no ``generator``, so build the draw from
        ``ref``'s shape / dtype / device explicitly; one shared helper keeps
        every Flipout layer (linear + conv) seed-reproducible. The generator is
        reconciled to ``ref``'s device via :meth:`_generator_for`, so a
        CPU-seeded layer still draws correctly once moved to CUDA.
        """
        return torch.randn(
            ref.shape,
            dtype=ref.dtype,
            device=ref.device,
            generator=self._generator_for(ref.device),
        )

    def kl_div(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        mu_p: torch.Tensor,
        sigma_p: torch.Tensor,
        *,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """
        KL divergence between two Gaussian distributions.

        Computes ``KL(Q || P)`` where ``Q ~ N(mu_q, sigma_q^2)`` and
        ``P ~ N(mu_p, sigma_p^2)``.

        The element-wise KL is reduced according to ``reduction``:

            - ``"sum"`` (default): total KL divergence across all
              variational weights. This is the convention required by
              the ELBO (``NLL_total + KL_total``).
            - ``"mean"``: element-wise mean. Useful for diagnostics
              but **not** the right quantity to add to NLL.
        """
        if reduction == "sum":
            return gaussian_kl(mu_q, sigma_q, mu_p, sigma_p)
        if reduction == "mean":
            return _gaussian_kl_terms(mu_q, sigma_q, mu_p, sigma_p).mean()
        raise GeoBrainError(
            "kl_div reduction must be 'sum' or 'mean'",
            object_name=type(self).__name__, field="reduction",
            expected="'sum' or 'mean'", actual=reduction,
        )


class LinearFlipout(BaseVariationalLayer):
    """
    Linear layer with Flipout reparameterization.

    Implements a variational linear layer using the Flipout technique
    for more efficient gradient estimation through decorrelated weight
    perturbations.

    The layer maintains a posterior distribution over weights::

        w ~ N(mu_weight, sigma_weight^2)

    against a prior ``p(w) ~ N(prior_mean, prior_variance)``.

    Args:
        in_features: input feature count.
        out_features: output feature count.
        prior_mean: mean of the Gaussian weight prior.
        prior_variance: variance of the Gaussian weight prior.
        posterior_mu_init: init value for the posterior means.
        posterior_rho_init: init value for the posterior ``rho``
            (``sigma = softplus(rho)``; ``-3.0`` gives a small initial
            sigma of about ``0.05``).
        bias: include a (variational) bias term.
        generator: optional RNG for reproducible perturbation draws.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self._generator = generator
        if not isinstance(in_features, int) or in_features <= 0:
            raise GeoBrainError(
                "LinearFlipout in_features must be a positive integer",
                object_name="LinearFlipout", field="in_features",
                expected="positive int", actual=in_features,
            )
        if not isinstance(out_features, int) or out_features <= 0:
            raise GeoBrainError(
                "LinearFlipout out_features must be a positive integer",
                object_name="LinearFlipout", field="out_features",
                expected="positive int", actual=out_features,
            )
        prior_var_f = float(prior_variance)
        if not prior_var_f > 0:
            raise GeoBrainError(
                "LinearFlipout prior_variance must be strictly positive "
                "(zero / negative variance gives nan / undefined KL)",
                object_name="LinearFlipout", field="prior_variance",
                expected="> 0", actual=prior_variance,
            )

        self.in_features = in_features
        self.out_features = out_features

        self.prior_mean = prior_mean
        self.prior_variance = prior_var_f
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        self.mu_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer(
            "prior_weight_mu",
            torch.Tensor(out_features, in_features),
            persistent=False,
        )
        self.register_buffer(
            "prior_weight_sigma",
            torch.Tensor(out_features, in_features),
            persistent=False,
        )

        if bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_features))
            self.rho_bias = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer(
                "prior_bias_mu", torch.Tensor(out_features), persistent=False
            )
            self.register_buffer(
                "prior_bias_sigma", torch.Tensor(out_features), persistent=False
            )
        else:
            self.register_buffer("prior_bias_mu", None, persistent=False)
            self.register_buffer("prior_bias_sigma", None, persistent=False)
            self.register_parameter("mu_bias", None)
            self.register_parameter("rho_bias", None)

        self.init_parameters()

    def init_parameters(self) -> None:
        """Initialize layer parameters."""
        # Store std, not variance: kl_div expects sigma.
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance**0.5)

        self.mu_weight.data.normal_(mean=self.posterior_mu_init, std=0.1, generator=self._generator)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init, std=0.1, generator=self._generator)

        if self.mu_bias is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance**0.5)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1, generator=self._generator)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1, generator=self._generator)

    def kl_loss(self) -> torch.Tensor:
        """KL divergence summed over weight + bias entries."""
        sigma_weight = F.softplus(self.rho_weight)
        kl = self.kl_div(
            self.mu_weight,
            sigma_weight,
            self.prior_weight_mu,
            self.prior_weight_sigma,
        )
        if self.mu_bias is not None:
            sigma_bias = F.softplus(self.rho_bias)
            kl = kl + self.kl_div(
                self.mu_bias,
                sigma_bias,
                self.prior_bias_mu,
                self.prior_bias_sigma,
            )
        return kl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Flipout sampling.

        Returns a plain tensor (composes with :class:`torch.nn.Sequential`).
        KL divergence is obtained separately via :meth:`kl_loss` or
        aggregated across a model with :func:`get_kl_loss`, the forward
        pass never returns it.
        """
        sigma_weight = F.softplus(self.rho_weight)
        delta_weight = sigma_weight * self._randn(self.mu_weight)

        bias = None
        if self.mu_bias is not None:
            sigma_bias = F.softplus(self.rho_bias)
            bias = sigma_bias * self._randn(self.mu_bias)

        outputs = F.linear(x, self.mu_weight, self.mu_bias)

        sign_input = (
            torch.empty_like(x)
            .uniform_(-1, 1, generator=self._generator_for(x.device))
            .sign()
        )
        sign_output = (
            torch.empty_like(outputs)
            .uniform_(-1, 1, generator=self._generator_for(outputs.device))
            .sign()
        )
        perturbed_outputs = (
            F.linear(x * sign_input, delta_weight, bias) * sign_output
        )

        return outputs + perturbed_outputs


class _ConvNdFlipout(BaseVariationalLayer):
    """
    Shared base for Conv2dFlipout and Conv3dFlipout.

    Subclasses set ``_conv_fn`` (e.g., :func:`F.conv2d`) and ``_ndim``
    (2 or 3) and call ``super().__init__()`` with the appropriate
    kernel shape.
    """

    _conv_fn = None  # type: ignore[assignment]
    _ndim: int = 0

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self._generator = generator
        cls_name = type(self).__name__
        for label, val in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("kernel_size", kernel_size),
            ("groups", groups),
        ):
            if not isinstance(val, int) or val <= 0:
                raise GeoBrainError(
                    f"{cls_name} {label} must be a positive integer",
                    object_name=cls_name, field=label,
                    expected="positive int", actual=val,
                )
        if in_channels % groups != 0:
            raise GeoBrainError(
                f"{cls_name} in_channels must be divisible by groups",
                object_name=cls_name, field="in_channels",
                expected=f"divisible by groups ({groups})", actual=in_channels,
            )
        if out_channels % groups != 0:
            raise GeoBrainError(
                f"{cls_name} out_channels must be divisible by groups",
                object_name=cls_name, field="out_channels",
                expected=f"divisible by groups ({groups})", actual=out_channels,
            )
        prior_var_f = float(prior_variance)
        if not prior_var_f > 0:
            raise GeoBrainError(
                f"{cls_name} prior_variance must be strictly positive "
                "(zero / negative variance gives nan / undefined KL)",
                object_name=cls_name, field="prior_variance",
                expected="> 0", actual=prior_variance,
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        self.prior_mean = prior_mean
        self.prior_variance = prior_var_f
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        kernel_shape = (
            out_channels,
            in_channels // groups,
            *([kernel_size] * self._ndim),
        )

        self.mu_kernel = nn.Parameter(torch.Tensor(*kernel_shape))
        self.rho_kernel = nn.Parameter(torch.Tensor(*kernel_shape))
        self.register_buffer(
            "prior_weight_mu", torch.Tensor(*kernel_shape), persistent=False
        )
        self.register_buffer(
            "prior_weight_sigma", torch.Tensor(*kernel_shape), persistent=False
        )

        if self.use_bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_channels))
            self.rho_bias = nn.Parameter(torch.Tensor(out_channels))
            self.register_buffer(
                "prior_bias_mu", torch.Tensor(out_channels), persistent=False
            )
            self.register_buffer(
                "prior_bias_sigma", torch.Tensor(out_channels), persistent=False
            )
        else:
            self.register_parameter("mu_bias", None)
            self.register_parameter("rho_bias", None)
            self.register_buffer("prior_bias_mu", None, persistent=False)
            self.register_buffer("prior_bias_sigma", None, persistent=False)

        self.init_parameters()

    def init_parameters(self) -> None:
        """Initialize layer parameters."""
        self.prior_weight_mu.data.fill_(self.prior_mean)
        self.prior_weight_sigma.data.fill_(self.prior_variance**0.5)
        self.mu_kernel.data.normal_(mean=self.posterior_mu_init, std=0.1, generator=self._generator)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init, std=0.1, generator=self._generator)

        if self.use_bias:
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1, generator=self._generator)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1, generator=self._generator)
            self.prior_bias_mu.data.fill_(self.prior_mean)
            self.prior_bias_sigma.data.fill_(self.prior_variance**0.5)

    def kl_loss(self) -> torch.Tensor:
        """KL divergence between posterior and prior (summed)."""
        sigma_weight = F.softplus(self.rho_kernel)
        kl = self.kl_div(
            self.mu_kernel,
            sigma_weight,
            self.prior_weight_mu,
            self.prior_weight_sigma,
        )
        if self.use_bias:
            sigma_bias = F.softplus(self.rho_bias)
            kl = kl + self.kl_div(
                self.mu_bias,
                sigma_bias,
                self.prior_bias_mu,
                self.prior_bias_sigma,
            )
        return kl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Flipout sampling.

        Returns a plain tensor. KL divergence is obtained separately via
        :meth:`kl_loss` or :func:`get_kl_loss`, the forward pass never
        returns it.
        """
        conv_fn = self._conv_fn
        if conv_fn is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set _conv_fn (e.g., F.conv2d)."
            )
        conv_kwargs = dict(
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        outputs = conv_fn(
            x, weight=self.mu_kernel, bias=self.mu_bias, **conv_kwargs
        )

        sign_input = (
            torch.empty_like(x)
            .uniform_(-1, 1, generator=self._generator_for(x.device))
            .sign()
        )
        sign_output = (
            torch.empty_like(outputs)
            .uniform_(-1, 1, generator=self._generator_for(outputs.device))
            .sign()
        )

        sigma_weight = F.softplus(self.rho_kernel)
        delta_kernel = sigma_weight * self._randn(self.mu_kernel)

        bias = None
        if self.use_bias:
            sigma_bias = F.softplus(self.rho_bias)
            bias = sigma_bias * self._randn(self.mu_bias)

        perturbed_outputs = (
            conv_fn(
                x * sign_input,
                weight=delta_kernel,
                bias=bias,
                **conv_kwargs,
            )
            * sign_output
        )

        return outputs + perturbed_outputs


class Conv2dFlipout(_ConvNdFlipout):
    """2D convolutional layer with Flipout reparameterization.

    Args:
        in_channels / out_channels / kernel_size / stride / padding /
            dilation / groups: standard conv semantics
            (:class:`torch.nn.Conv2d`).
        prior_mean / prior_variance / posterior_mu_init /
            posterior_rho_init / bias / generator: variational semantics as
            in :class:`LinearFlipout`.
    """

    _conv_fn = staticmethod(F.conv2d)
    _ndim = 2


class Conv3dFlipout(_ConvNdFlipout):
    """3D convolutional layer with Flipout reparameterization.

    Args:
        in_channels / out_channels / kernel_size / stride / padding /
            dilation / groups: standard conv semantics
            (:class:`torch.nn.Conv3d`).
        prior_mean / prior_variance / posterior_mu_init /
            posterior_rho_init / bias / generator: variational semantics as
            in :class:`LinearFlipout`.
    """

    _conv_fn = staticmethod(F.conv3d)
    _ndim = 3


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def get_kl_loss(model: nn.Module) -> torch.Tensor:
    """
    Collect total KL divergence from variational layers in a model.

    Iterates ``model.modules()`` and sums
    :meth:`BaseVariationalLayer.kl_loss` only on instances of
    :class:`BaseVariationalLayer`. This is stricter than
    ``hasattr(module, "kl_loss")`` so that a user's wrapper module that
    defines its own aggregating ``kl_loss()`` method does **not**
    double-count the layers it contains.

    Returns ``torch.tensor(0.0)`` if the model has no variational
    layers.
    """
    kl = None
    for module in model.modules():
        if not isinstance(module, BaseVariationalLayer):
            continue
        term = module.kl_loss()
        kl = term if kl is None else kl + term
    if kl is None:
        return torch.tensor(0.0)
    return kl


class _KLLossModule(nn.Module):
    """``forward() = get_kl_loss(net)``: a functional_call target.

    ``torch.func.functional_call`` substitutes parameters only for the
    duration of a *forward* call, so aggregating KL over externally-supplied
    θ needs the aggregation to BE a forward.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self) -> torch.Tensor:
        return get_kl_loss(self.net)


def kl_regularizer(
    network: nn.Module, weight: float = 1.0, *, param_prefix: str = ""
) -> Callable[[Mapping[str, torch.Tensor]], torch.Tensor]:
    """ELBO bridge: an :class:`~geobrain.optim.Inverter` ``regularizer=``
    computing ``weight * KL(q(θ) || p(θ))`` from the CURRENT params dict.

    The Inverter clones its ``params`` and hands the live clones to the
    regularizer each step, so the KL is evaluated by routing those tensors
    through the network via ``torch.func.functional_call``, never by
    reading the module's own (stale) parameters. Entries named
    ``param_prefix + <network parameter name>`` are substituted; any other
    params-dict entries are ignored, and network parameters absent from the
    dict keep their module values.

    Args:
        network: the variational network whose KL is summed.
        weight: multiplier on the summed KL (the ELBO trade-off knob).
        param_prefix: prepended to network parameter names when looking
            them up in the params dict.

    With :class:`~geobrain.nn.WeightReparameterization` this turns a
    Flipout network into a Bayesian DIP::

        rp = WeightReparameterization(net, fixed_input=x0, outputs="m")
        inv = Inverter(problem, params=rp.initial_params(),
                       regularizer=kl_regularizer(net, weight=1e-3))
        # loss = physics NLL + weight * KL(q(θ) || p(θ))
    """
    if not isinstance(network, nn.Module):
        raise GeoBrainError(
            "kl_regularizer network must be a torch.nn.Module",
            object_name="kl_regularizer",
            field="network",
            expected=nn.Module,
            actual=type(network),
        )
    weight_f = float(weight)
    if not weight_f >= 0.0:
        raise GeoBrainError(
            "kl_regularizer weight must be >= 0",
            object_name="kl_regularizer",
            field="weight",
            expected=">= 0",
            actual=weight,
        )
    wrapper = _KLLossModule(network)
    names = tuple(name for name, _ in network.named_parameters())

    def _reg(params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        weights = {
            f"net.{name}": params[param_prefix + name]
            for name in names
            if (param_prefix + name) in params
        }
        return weight_f * torch.func.functional_call(wrapper, weights, ())

    return _reg


def count_variational_parameters(model: nn.Module) -> int:
    """Count total number of variational parameters in a model."""
    total = 0
    for module in model.modules():
        if isinstance(module, BaseVariationalLayer):
            for param in module.parameters():
                total += param.numel()
    return total
