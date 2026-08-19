# pyright: reportPrivateImportUsage=false
"""
GAN-based 3D facies generation.

Provides a simple DCGAN-style 3D generator (Generator3D) and the
GANSimulator interface for facies modeling. The generator maps a flat
latent vector z to a 64^3 volume with num_classes channels (logits),
then argmax gives discrete phase labels (0/1/2).

Architecture (Generator3D, default config):
    z (B, 512)  ->  FC  ->  (B, 256, 4, 4, 4)
        -> ConvT 4->8   (256 -> 128)
        -> ConvT 8->16  (128 -> 64)
        -> ConvT 16->32 (64  -> 32)
        -> ConvT 32->64 (32  -> num_classes)
    Output: (B, num_classes, 64, 64, 64) logits

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import os
from geobrain.core.errors import GeoBrainError
from typing import Optional, Literal, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import GenerativeSimulator, GenerativeConfig, _SoftFieldDecoder


# ============================================================================
# Generator architecture
# ============================================================================

class Generator3D(nn.Module):
    """
    DCGAN-style 3D generator for facies modeling.

    Maps a flat latent vector z to a 64^3 volume with *num_classes* output
    channels (logits).  ``argmax(dim=1)`` yields discrete phase labels.

    Args:
        latent_dim: Length of the input z vector.
        num_classes: Number of facies classes (output channels).
        base_ch: Channel count at the 4^3 stage; halved at each upsample.

    Example:
        >>> g = Generator3D(latent_dim=512, num_classes=3)
        >>> z = torch.randn(4, 512)
        >>> logits = g(z)          # (4, 3, 64, 64, 64)
        >>> labels = logits.argmax(1)  # (4, 64, 64, 64)
    """

    def __init__(self, latent_dim: int = 512, num_classes: int = 3,
                 base_ch: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_ch = base_ch

        # z -> (base_ch, 4, 4, 4)
        self.project = nn.Sequential(
            nn.Linear(latent_dim, base_ch * 4 * 4 * 4),
            nn.ReLU(True),
        )

        # 4 -> 8 -> 16 -> 32 -> 64
        self.net = nn.Sequential(
            self._block(base_ch,      base_ch // 2),   # 4  -> 8
            self._block(base_ch // 2, base_ch // 4),   # 8  -> 16
            self._block(base_ch // 4, base_ch // 8),   # 16 -> 32
            # final stage – no BN, no ReLU
            nn.ConvTranspose3d(base_ch // 8, num_classes, 4, 2, 1, bias=False),
        )

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) or (B, latent_dim, 1, 1, 1).
        Returns:
            Logits of shape (B, num_classes, 64, 64, 64).
        """
        if z.dim() > 2:
            z = z.view(z.size(0), -1)
        h = self.project(z).view(-1, self.base_ch, 4, 4, 4)
        return self.net(h)


# ============================================================================
# Simulator
# ============================================================================

class GANSimulator(GenerativeSimulator):
    """
    GAN-based simulator for 3D facies generation.

    **Experimental API**; may change in a minor release before it
    stabilizes.

    Uses a DCGAN-style ``Generator3D`` that maps a latent vector z to a
    64^3 volume of discrete phase labels {0, 1, …, num_classes-1}.

    Args:
        model: Pre-built generator network (any ``nn.Module`` whose
               ``forward(z)`` returns ``(B, num_classes, 64, 64, 64)``
               logits).  If *None* and *checkpoint_path* is given, a
               ``Generator3D`` is created automatically.
        checkpoint_path: Path to a saved generator state-dict (``.pth``).
        latent_dim: Dimension of the latent z vector (default 512).
        num_classes: Number of facies classes (default 3).
        base_ch: Base channel count for the built-in ``Generator3D``.
        truncation: Truncation parameter for sampling (1.0 = no truncation).
                   Values < 1 clamp z to ``[-truncation, truncation]``,
                   reducing diversity but improving average quality.
        transform: Optional output transform applied after generation.

    Example:
        >>> # With pre-trained checkpoint
        >>> sim = GANSimulator(
        ...     checkpoint_path='generator.pth', latent_dim=512)
        >>> config = GenerativeConfig(n_realizations=8, seed=42)
        >>> labels = sim.simulate(config)   # (8, 64, 64, 64)
        >>>
        >>> # From latent vector
        >>> z = torch.randn(1, 512)
        >>> labels = sim.generate_from_latent(z)
        >>>
        >>> # Latent interpolation
        >>> z1, z2 = torch.randn(512), torch.randn(512)
        >>> z_path = sim.interpolate_latent(z1, z2, n_steps=10, method='slerp')
        >>> labels = sim.generate_from_latent(z_path)
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        latent_dim: int = 512,
        num_classes: int = 3,
        base_ch: int = 256,
        truncation: float = 1.0,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            checkpoint_path=checkpoint_path,
            latent_dim=latent_dim,
            transform=transform,
        )

        if not 0 < truncation <= 1.0:
            raise GeoBrainError(f"truncation must be in (0, 1], got {truncation}")

        self.truncation = truncation
        self.num_classes = num_classes
        self.base_ch = base_ch

    # -----------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------

    def _load_checkpoint(self, device: torch.device) -> None:
        """Load Generator3D from a state-dict checkpoint file."""
        ckpt_path = self.checkpoint_path
        if ckpt_path is None or not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Generator checkpoint not found: {ckpt_path}"
            )

        latent_dim = self.latent_dim if self.latent_dim is not None else 512
        self._model = Generator3D(
            latent_dim=latent_dim,
            num_classes=self.num_classes,
            base_ch=self.base_ch,
        ).to(device)
        self._model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        self._model.eval()

    def differentiable_decoder(self) -> nn.Module:
        """Soft-field decoder ``z → softmax(Generator3D(z))``: differentiable
        (no ``argmax``, no ``no_grad``), for
        ``geobrain.nn.LatentReparameterization``.
        """
        def decode(z: torch.Tensor) -> torch.Tensor:
            return self._ensure_model_loaded(z.device)(z)
        return _SoftFieldDecoder(decode)

    # -----------------------------------------------------------------
    # Core generation
    # -----------------------------------------------------------------

    def _simulate(self, config: GenerativeConfig) -> torch.Tensor:
        """
        Generate 3D facies volumes.

        Returns:
            Phase labels (n_realizations, 64, 64, 64), dtype ``torch.long``,
            values in {0, …, num_classes-1}.
        """
        device = config.torch_device
        model = self._ensure_model_loaded(device)
        n = config.n_realizations

        z = self._sample_latent_truncated(n, device)

        with torch.no_grad():
            logits = model(z)  # (B, num_classes, 64, 64, 64)

        return logits.argmax(dim=1)  # (B, 64, 64, 64)

    # -----------------------------------------------------------------
    # Truncated sampling
    # -----------------------------------------------------------------

    def _sample_latent_truncated(
        self, n: int, device: torch.device,
    ) -> torch.Tensor:
        """
        Sample latent vectors, optionally applying the truncation trick.

        Args:
            n: Number of samples.
            device: Target device.

        Returns:
            Latent vectors (n, latent_dim).
        """
        z = self._sample_latent(n, device)
        if self.truncation < 1.0:
            z = z.clamp(-self.truncation, self.truncation)
        return z

    # -----------------------------------------------------------------
    # Generation from explicit latent
    # -----------------------------------------------------------------

    @torch.no_grad()
    def generate_from_latent(
        self,
        z: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate facies volume(s) from explicit latent vector(s).

        Args:
            z: Latent vector(s) of shape ``(B, latent_dim)`` or
               ``(latent_dim,)``.
            device: Target device (uses z.device by default).

        Returns:
            Phase labels (B, 64, 64, 64).
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        device = device or z.device
        model = self._ensure_model_loaded(device)
        logits = model(z.to(device))
        return logits.argmax(dim=1)

    # -----------------------------------------------------------------
    # Latent interpolation
    # -----------------------------------------------------------------

    def interpolate_latent(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        n_steps: int = 10,
        method: Literal['linear', 'slerp'] = 'slerp',
    ) -> torch.Tensor:
        """
        Interpolate between two latent vectors.

        Args:
            z1: Starting latent vector (latent_dim,).
            z2: Ending latent vector (latent_dim,).
            n_steps: Number of interpolation steps.
            method: ``'linear'`` or ``'slerp'`` (spherical linear).

        Returns:
            Interpolated latent vectors (n_steps, latent_dim).
        """
        if z1.dim() > 1:
            z1 = z1.squeeze(0)
        if z2.dim() > 1:
            z2 = z2.squeeze(0)

        ts = torch.linspace(0.0, 1.0, n_steps, device=z1.device)

        if method == 'linear':
            return torch.stack(
                [z1 * (1.0 - t) + z2 * t for t in ts], dim=0
            )

        # slerp
        z1n = F.normalize(z1, dim=0)
        z2n = F.normalize(z2, dim=0)
        omega = torch.acos(
            (z1n * z2n).sum().clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        )
        sin_omega = omega.sin()

        if sin_omega.abs() < 1e-6:
            # Vectors (anti-)parallel – fall back to linear
            return torch.stack(
                [z1 * (1.0 - t) + z2 * t for t in ts], dim=0
            )

        return torch.stack([
            (((1.0 - t) * omega).sin() / sin_omega) * z1
            + ((t * omega).sin() / sin_omega) * z2
            for t in ts
        ], dim=0)
