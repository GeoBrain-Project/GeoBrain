# pyright: reportPrivateImportUsage=false
"""
VAE-based geological modeling using LDMC's AutoencoderKL.

The AutoencoderKL provides a KL-regularized latent space for 3D geological
volumes. It compresses 64^3 three-phase volumes to a 16^3 latent space (4x
spatial compression per axis), enabling:
    - Direct generation by sampling in latent space
    - Encoding real volumes to latent representations
    - Smooth interpolation between geological models
    - Integration with the optim module for latent-space inversion

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import importlib
import os
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn

from .base import GenerativeSimulator, GenerativeConfig, _SoftFieldDecoder
from .dependencies import LDMC_PROVIDER, require_provider


# ---------------------------------------------------------------------------
# Lazy LDMC import helper
# ---------------------------------------------------------------------------

def _import_ldmc_vae():
    """Lazily import LDMC's AutoencoderKL."""
    require_provider(LDMC_PROVIDER)
    models = importlib.import_module("LDMC.models")
    return models.AutoencoderKL


class VAESimulator(GenerativeSimulator):
    """
    VAE-based simulator using LDMC's AutoencoderKL.

    **Experimental API**; may change in a minor release before it
    stabilizes.

    Provides generation, encoding, decoding, reconstruction, and latent
    interpolation for 3D geological volumes (64^3, 3-phase).

    The VAE decoder can also be used with the optim module for
    latent-space inversion (LatentParameterization).

    Args:
        model: Pre-trained AutoencoderKL instance.
        checkpoint_path: Path to saved VAE checkpoint file (.pth).
        num_classes: Number of phase classes (default 3).
        latent_ch: Number of latent channels (default 3).
        transform: Optional output transform function.

    Example:
        >>> # From checkpoint
        >>> sim = VAESimulator(checkpoint_path='./ckpt/autoencoder_final.pth')
        >>> config = GenerativeConfig(n_realizations=4, seed=42)
        >>> labels = sim.simulate(config)  # (4, 64, 64, 64), values in {0, 1, 2}
        >>>
        >>> # Encode-decode round trip
        >>> z_mu, z_sigma = sim.encode(one_hot_volume)
        >>> reconstructed = sim.decode(z_mu)
        >>>
        >>> # Latent interpolation
        >>> interp = sim.interpolate_fields(volume_a, volume_b, n_steps=10)

    Integration with optim module (the operator-native reparameterization
    seam, supports every solver, not just Inverter)::

        >>> from geobrain.nn import LatentReparameterization
        >>> from geobrain import InverseProblem
        >>> from geobrain.optim import Inverter
        >>> reparam = LatentReparameterization(
        ...     sim.differentiable_decoder(), outputs="facies",
        ... )
        >>> problem = InverseProblem(
        ...     forward=seismic_op @ reparam,
        ...     observed=observed_data,
        ...     likelihood=likelihood,
        ... )
        >>> inverter = Inverter(
        ...     problem,
        ...     params=reparam.initial_params(torch.zeros(1, 3, 16, 16, 16)),
        ... )
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        num_classes: int = 3,
        latent_ch: int = 3,
        latent_dim: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            checkpoint_path=checkpoint_path,
            latent_dim=latent_dim,
            transform=transform,
        )
        self.num_classes = num_classes
        self.latent_ch = latent_ch

    # =========================================================================
    # Model loading
    # =========================================================================

    def _load_checkpoint(self, device: torch.device) -> None:
        """
        Load AutoencoderKL from checkpoint file.

        Args:
            device: Target device.
        """
        AutoencoderKL = _import_ldmc_vae()

        ckpt_path = self.checkpoint_path
        if ckpt_path is None or not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"VAE checkpoint not found: {ckpt_path}"
            )

        self._model = AutoencoderKL(
            in_ch=self.num_classes,
            out_ch=self.num_classes,
            latent_ch=self.latent_ch,
        ).to(device)
        self._model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        self._model.eval()

    # =========================================================================
    # Core generation
    # =========================================================================

    def differentiable_decoder(self) -> nn.Module:
        """Soft-field decoder ``z → softmax(decode_stage_2_outputs(z))``,
        differentiable (no ``argmax``, no ``no_grad``), unlike :meth:`decode` /
        :meth:`decode_logits`. Use with ``geobrain.nn.LatentReparameterization``
        to invert in the VAE latent space through the physics forward.
        """
        def decode(z: torch.Tensor) -> torch.Tensor:
            return self._ensure_model_loaded(z.device).decode_stage_2_outputs(z)
        return _SoftFieldDecoder(decode)

    def _simulate(self, config: GenerativeConfig) -> torch.Tensor:
        """
        Generate 3D geological volumes by sampling from the VAE latent space.

        Samples z ~ N(0, I) and decodes to phase labels. Note that samples
        from a VAE latent space may be less coherent than those from the
        full latent diffusion pipeline (DiffusionSimulator).

        Args:
            config: Generation configuration.

        Returns:
            Phase labels tensor of shape (n_realizations, D, H, W) with
            values in {0, 1, ..., num_classes-1}. dtype is torch.long.
        """
        device = config.torch_device
        model = self._ensure_model_loaded(device)

        n = config.n_realizations

        # Sample in 3D latent space: (B, latent_ch, 16, 16, 16). The generator
        # (set by _set_seed from config.seed) keeps seeding off the global RNG.
        z = torch.randn(
            n, self.latent_ch, 16, 16, 16,
            device=device, dtype=config.torch_dtype, generator=self._generator,
        )

        with torch.no_grad():
            logits = model.decode_stage_2_outputs(z)  # (B, C, D, H, W)

        labels = logits.argmax(dim=1)  # (B, D, H, W)
        return labels

    # =========================================================================
    # VAE-specific interface methods
    # =========================================================================

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input volume to latent distribution parameters.

        Args:
            x: One-hot encoded volume (B, num_classes, D, H, W).
               For 3-phase: (B, 3, 64, 64, 64).

        Returns:
            Tuple of (z_mu, z_sigma) each of shape (B, latent_ch, 16, 16, 16).
        """
        device = x.device
        model = self._ensure_model_loaded(device)
        return model.encode(x)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent tensor to phase labels.

        Args:
            z: Latent tensor (B, latent_ch, 16, 16, 16).

        Returns:
            Phase labels (B, D, H, W) with values in {0, ..., num_classes-1}.
        """
        device = z.device
        model = self._ensure_model_loaded(device)
        logits = model.decode_stage_2_outputs(z)  # (B, C, D, H, W)
        return logits.argmax(dim=1)

    @torch.no_grad()
    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent tensor to raw logits (before argmax).

        Useful for differentiable pipelines (e.g., latent-space inversion).

        Args:
            z: Latent tensor (B, latent_ch, 16, 16, 16).

        Returns:
            Logits (B, num_classes, D, H, W).
        """
        device = z.device
        model = self._ensure_model_loaded(device)
        return model.decode_stage_2_outputs(z)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input through full VAE (encode -> reparameterize -> decode).

        Args:
            x: One-hot encoded volume (B, num_classes, D, H, W).

        Returns:
            Reconstructed phase labels (B, D, H, W).
        """
        device = x.device
        model = self._ensure_model_loaded(device)
        z_mu, z_sigma = model.encode(x)
        z = model.reparameterize(z_mu, z_sigma)
        logits = model.decode_stage_2_outputs(z)
        return logits.argmax(dim=1)

    @torch.no_grad()
    def interpolate_latent(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """
        Linearly interpolate in latent space between two points.

        Args:
            z1: Starting latent tensor (1, latent_ch, 16, 16, 16) or
                (latent_ch, 16, 16, 16).
            z2: Ending latent tensor (same shape as z1).
            n_steps: Number of interpolation steps.

        Returns:
            Interpolated latent tensors (n_steps, latent_ch, 16, 16, 16).
        """
        if z1.dim() == 4:
            z1 = z1.unsqueeze(0)
        if z2.dim() == 4:
            z2 = z2.unsqueeze(0)

        alphas = torch.linspace(0.0, 1.0, n_steps, device=z1.device)
        # Broadcast: (n_steps, 1, 1, 1, 1) * (1, C, D, H, W)
        alphas = alphas.view(n_steps, 1, 1, 1, 1)
        return z1 * (1.0 - alphas) + z2 * alphas

    @torch.no_grad()
    def interpolate_fields(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """
        Interpolate between two geological volumes via latent space.

        Encodes both volumes, linearly interpolates in latent space,
        and decodes each intermediate point.

        Args:
            x1: Starting one-hot volume (B, num_classes, D, H, W) or
                (num_classes, D, H, W).
            x2: Ending one-hot volume (same shape as x1).
            n_steps: Number of interpolation steps.

        Returns:
            Interpolated phase labels (n_steps, D, H, W).
        """
        if x1.dim() == 4:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 4:
            x2 = x2.unsqueeze(0)

        device = x1.device
        model = self._ensure_model_loaded(device)

        # Encode to latent means (no sampling for deterministic interpolation)
        z1, _ = model.encode(x1)
        z2, _ = model.encode(x2)

        # Interpolate
        z_interp = self.interpolate_latent(z1.squeeze(0), z2.squeeze(0), n_steps)

        # Decode each step
        logits = model.decode_stage_2_outputs(z_interp)
        return logits.argmax(dim=1)  # (n_steps, D, H, W)

    @torch.no_grad()
    def compute_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute latent representation (encoder mean, no sampling).

        Args:
            x: One-hot encoded volume (B, num_classes, D, H, W).

        Returns:
            Latent mean z_mu of shape (B, latent_ch, 16, 16, 16).
        """
        device = x.device
        model = self._ensure_model_loaded(device)
        z_mu, _ = model.encode(x)
        return z_mu
