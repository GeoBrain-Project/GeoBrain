# pyright: reportPrivateImportUsage=false
"""
Latent Diffusion Model for 3D geological generation.

Based on the LDMC (Latent Diffusion Model for Conditional generation) architecture:
    - Stage 1: AutoencoderKL compresses 64^3 volumes to 16^3 latent space
    - Stage 2: DiffusionUNet3D generates in latent space via DDPM/DDIM

Supports unconditional and conditional (well-constrained) generation with
optional classifier-free guidance.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import importlib
import os
from geobrain.core.errors import GeoBrainError
import contextlib
from typing import Optional, Literal, Callable, List, cast

import torch
import torch.nn as nn

from .base import GenerativeSimulator, GenerativeConfig, _SoftFieldDecoder
from .dependencies import LDMC_PROVIDER, require_provider


# ---------------------------------------------------------------------------
# Lazy LDMC import helper
# ---------------------------------------------------------------------------

def _import_ldmc():
    """Lazily import LDMC through the structured optional-provider seam."""
    require_provider(LDMC_PROVIDER)
    models = importlib.import_module("LDMC.models")
    conditioning = importlib.import_module("LDMC.conditioning")
    return (
        models.AutoencoderKL,
        models.DiffusionUNet3D,
        models.DDPMScheduler,
        models.DDIMScheduler,
        conditioning,
    )


class DiffusionSimulator(GenerativeSimulator):
    """
    Latent Diffusion Model simulator for 3D geological field generation.

    **Experimental API**; may change in a minor release before it
    stabilizes.

    Uses a two-stage architecture from LDMC:
        1. AutoencoderKL: 64^3 input -> 16^3 latent (3-phase one-hot encoding)
        2. DiffusionUNet3D: Generates in 16^3 latent space via DDPM/DDIM

    Supports:
        - Unconditional generation of 3-phase geological volumes
        - Conditional generation with well constraints (vertical boreholes)
        - Classifier-free guidance for improved conditioning quality
        - DDPM (full, 1000 steps) and DDIM (accelerated, ~50 steps) sampling

    Args:
        checkpoint_path: Path to checkpoint directory containing:
            - autoencoder_final.pth (VAE weights)
            - diffusion_final.pth or diffusion_cond_final.pth (UNet weights)
            - ldm_meta.pth or ldm_cond_meta.pth (scale_factor)
        model: Pre-built DiffusionUNet3D (alternative to checkpoint).
        autoencoder: Pre-built AutoencoderKL (alternative to checkpoint).
        scale_factor: Latent scaling factor (default 1.0).
        sampler: Sampling method ('ddpm' or 'ddim').
        inference_steps: Number of denoising steps (default: 50 for ddim, 1000 for ddpm).
        timesteps: Number of training timesteps.
        beta_start: Starting noise level.
        beta_end: Ending noise level.
        cond_ch: Conditioning channels (0=unconditional, 4=conditional).
        guidance_scale: Classifier-free guidance scale (1.0=no guidance).
        num_classes: Number of phase classes (default 3).
        batch_size: Max samples per batch during generation.
        transform: Optional output transform function.

    Example:
        >>> # Unconditional generation
        >>> sim = DiffusionSimulator(checkpoint_path='./checkpoints')
        >>> config = GenerativeConfig(n_realizations=4, seed=42)
        >>> labels = sim.simulate(config)  # (4, 64, 64, 64), values in {0, 1, 2}
        >>>
        >>> # Conditional with well constraints
        >>> wells = [{"x": 10, "y": 20, "labels": [0, 0, 1, ...]}]  # len(labels)=64
        >>> sim = DiffusionSimulator(
        ...     checkpoint_path='./checkpoints',
        ...     cond_ch=4, guidance_scale=3.0)
        >>> labels = sim.set_conditioning(wells).simulate(config)
    """

    SUPPORTED_SAMPLERS = ("ddpm", "ddim")

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        autoencoder: Optional[nn.Module] = None,
        scale_factor: float = 1.0,
        sampler: Literal["ddpm", "ddim"] = "ddim",
        inference_steps: Optional[int] = None,
        timesteps: int = 1000,
        beta_start: float = 0.0015,
        beta_end: float = 0.0195,
        cond_ch: int = 0,
        guidance_scale: float = 1.0,
        num_classes: int = 3,
        batch_size: int = 4,
        latent_dim: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            checkpoint_path=checkpoint_path,
            latent_dim=latent_dim,
            transform=transform,
        )

        if sampler not in self.SUPPORTED_SAMPLERS:
            raise GeoBrainError(
                f"Unknown sampler: '{sampler}'. "
                f"Supported: {self.SUPPORTED_SAMPLERS}"
            )

        self.sampler = sampler
        self.timesteps = timesteps
        self.inference_steps = inference_steps or (50 if sampler == "ddim" else timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.cond_ch = cond_ch
        self.guidance_scale = guidance_scale
        self.num_classes = num_classes
        self.batch_size = batch_size

        # VAE and scale factor
        self._autoencoder = autoencoder
        self._scale_factor = scale_factor

        # Well conditioning data
        self._wells: Optional[List[dict]] = None

    # =========================================================================
    # Model loading
    # =========================================================================

    def _load_checkpoint(self, device: torch.device) -> None:
        """
        Load VAE and UNet from checkpoint directory.

        Expected directory structure:
            checkpoint_path/
                autoencoder_final.pth
                diffusion_final.pth  (or diffusion_cond_final.pth)
                ldm_meta.pth  (or ldm_cond_meta.pth)
        """
        AutoencoderKL, DiffusionUNet3D, _, _, _ = _import_ldmc()

        ckpt_dir = self.checkpoint_path
        if ckpt_dir is None:
            raise GeoBrainError(
                "DiffusionSimulator requires checkpoint_path to be set."
            )
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {ckpt_dir}"
            )

        # --- Load VAE ---
        vae_path = os.path.join(ckpt_dir, "autoencoder_final.pth")
        if not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE checkpoint not found: {vae_path}")

        self._autoencoder = AutoencoderKL(
            in_ch=self.num_classes,
            out_ch=self.num_classes,
        ).to(device)
        self._autoencoder.load_state_dict(
            torch.load(vae_path, map_location=device, weights_only=True)
        )
        self._autoencoder.eval()

        # --- Load UNet ---
        if self.cond_ch > 0:
            diff_path = os.path.join(ckpt_dir, "diffusion_cond_final.pth")
        else:
            diff_path = os.path.join(ckpt_dir, "diffusion_final.pth")
        if not os.path.exists(diff_path):
            raise FileNotFoundError(f"UNet checkpoint not found: {diff_path}")

        self._model = DiffusionUNet3D(cond_ch=self.cond_ch).to(device)
        self._model.load_state_dict(
            torch.load(diff_path, map_location=device, weights_only=True)
        )
        self._model.eval()

        # --- Load scale factor ---
        if self.cond_ch > 0:
            meta_path = os.path.join(ckpt_dir, "ldm_cond_meta.pth")
            if not os.path.exists(meta_path):
                meta_path = os.path.join(ckpt_dir, "ldm_meta.pth")
        else:
            meta_path = os.path.join(ckpt_dir, "ldm_meta.pth")

        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location=device, weights_only=True)
            self._scale_factor = meta["scale_factor"]

    def _ensure_models_loaded(self, device: torch.device) -> None:
        """Ensure both VAE and UNet are loaded and on the correct device."""
        if self._model is None or self._autoencoder is None:
            if self.checkpoint_path is None:
                raise GeoBrainError(
                    "No models available. Provide 'checkpoint_path' pointing "
                    "to the LDMC checkpoint directory, or pass pre-built "
                    "'model' (UNet) and 'autoencoder' (VAE) instances."
                )
            self._load_checkpoint(device)
        else:
            self._model = self._model.to(device).eval()
            self._autoencoder = self._autoencoder.to(device).eval()

    # =========================================================================
    # Scheduler
    # =========================================================================

    def _create_scheduler(self):
        """Create and configure the noise scheduler."""
        _, _, DDPMScheduler, DDIMScheduler, _ = _import_ldmc()

        sched_cls = DDIMScheduler if self.sampler == "ddim" else DDPMScheduler
        scheduler = sched_cls(
            num_train_timesteps=self.timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
        )
        scheduler.set_timesteps(self.inference_steps)
        return scheduler

    # =========================================================================
    # Core generation
    # =========================================================================

    @torch.no_grad()
    def _generate_batch(self, noise, device, cond=None):
        """
        Reverse diffusion in latent space, then decode to phase labels.

        Args:
            noise: Initial noise (B, 3, 16, 16, 16).
            device: Target device.
            cond: Optional conditioning tensor (B, cond_ch, 16, 16, 16).

        Returns:
            Phase labels (B, D, H, W) with values in {0, ..., num_classes-1}.
        """
        scheduler = self._create_scheduler()
        x = noise
        model = self._model
        autoencoder = self._autoencoder
        assert model is not None and autoencoder is not None

        for t in scheduler.timesteps:
            t_batch = torch.full(
                (x.shape[0],), t, device=device, dtype=torch.long
            )

            if cond is not None and self.guidance_scale != 1.0:
                # Classifier-free guidance
                eps_cond = model(x, t_batch, cond=cond)
                eps_uncond = model(
                    x, t_batch, cond=torch.zeros_like(cond)
                )
                eps = eps_uncond + self.guidance_scale * (
                    eps_cond - eps_uncond
                )
            elif cond is not None:
                eps = model(x, t_batch, cond=cond)
            else:
                eps = model(x, t_batch)

            x, _ = scheduler.step(eps, t, x)

        # Decode latent -> 3-channel logits -> argmax -> phase labels
        logits = autoencoder.decode_stage_2_outputs(  # pyright: ignore[reportCallIssue]
            x / self._scale_factor
        )
        labels = logits.argmax(dim=1)  # (B, D, H, W)
        return labels

    def _build_conditioning(self, batch_size, device):
        """Build latent conditioning tensor from well data."""
        if self._wells is None:
            return None

        _, _, _, _, cond_module = _import_ldmc()

        cond_vol, mask_vol = cond_module.wells_to_conditioning_volumes(
            self._wells,
            num_classes=self.num_classes,
            batch_size=batch_size,
        )
        cond_vol = cond_vol.to(device)
        mask_vol = mask_vol.to(device)

        cond = cond_module.build_latent_conditioning(
            cond_vol, mask_vol, self._autoencoder, self._scale_factor
        )
        return cond

    def differentiable_decoder(self) -> nn.Module:
        """Soft-field decoder ``z → softmax(autoencoder.decode(z / scale))``,
        differentiable (no ``argmax``, no ``no_grad``). Wraps the **autoencoder
        decode**, NOT the diffusion UNet (``self._model``): latent inversion
        runs the autoencoder decode, not the iterative sampler. Use with
        ``geobrain.nn.LatentReparameterization``.
        """
        def decode(z: torch.Tensor) -> torch.Tensor:
            if self._autoencoder is None:
                # Load BOTH UNet and autoencoder (the base singular
                # _ensure_model_loaded only loads the UNet, leaving the
                # autoencoder None). _load_checkpoint populates _autoencoder.
                self._ensure_models_loaded(z.device)
            if self._autoencoder is None:
                # Constructed with model=<UNet> but no autoencoder/checkpoint:
                # surface a structured error instead of a bare NoneType.to.
                raise GeoBrainError(
                    "DiffusionSimulator: no autoencoder available; pass "
                    "autoencoder= or checkpoint_path=",
                    object_name="DiffusionSimulator",
                    field="autoencoder",
                    expected="AutoencoderKL instance or checkpoint_path",
                    actual=None,
                )
            ae = self._autoencoder.to(z.device).eval()
            return ae.decode_stage_2_outputs(z / self._scale_factor)
        return _SoftFieldDecoder(decode)

    def _simulate(self, config: GenerativeConfig) -> torch.Tensor:
        """
        Generate 3D geological volumes using latent diffusion.

        Args:
            config: Generation configuration.

        Returns:
            Phase labels tensor of shape (n_realizations, D, H, W) with
            values in {0, 1, ..., num_classes-1}. dtype is torch.long.
        """
        device = config.torch_device
        self._ensure_models_loaded(device)

        n_total = config.n_realizations
        all_labels = []
        generated = 0

        while generated < n_total:
            bs = min(self.batch_size, n_total - generated)

            # Sample noise in latent space: (B, latent_ch, 16, 16, 16). The
            # generator (set by _set_seed from config.seed) keeps the seed off
            # the global RNG.
            noise = torch.randn(
                bs, 3, 16, 16, 16, device=device, dtype=config.torch_dtype,
                generator=self._generator
            )

            # Build conditioning from wells (if set)
            cond = (
                self._build_conditioning(bs, device)
                if self._wells is not None
                else None
            )

            # AMP context
            if config.use_amp and device.type == "cuda":
                amp_ctx = torch.amp.autocast("cuda")
            else:
                amp_ctx = contextlib.nullcontext()

            with amp_ctx:
                labels = self._generate_batch(noise, device, cond=cond)

            # Apply pixel-level well correction
            if self._wells is not None:
                _, _, _, _, cond_module = _import_ldmc()
                labels = cond_module.apply_well_correction(
                    labels, self._wells
                )

            all_labels.append(labels if config.output_device == "compute" else labels.to("cpu"))
            generated += bs

        return torch.cat(all_labels, dim=0)  # (n_total, D, H, W)

    # =========================================================================
    # Conditioning
    # =========================================================================

    def set_conditioning(self, data) -> 'DiffusionSimulator':
        """
        Set well conditioning data for conditional generation.

        Args:
            data: Well conditioning data. Either a list of well dicts
                ``[{"x": int, "y": int, "labels": [int, ...]}]`` (labels
                length = volume depth, 64), or a path to a wells JSON
                file (string).

        Returns:
            Self for method chaining.

        Example:
            >>> wells = [
            ...     {"x": 10, "y": 20, "labels": [0, 0, 1, 1, 2, ...]},
            ...     {"x": 30, "y": 40, "labels": [0, 1, 1, 2, 2, ...]},
            ... ]
            >>> sim.set_conditioning(wells).simulate(config)
        """
        if isinstance(data, str):
            _, _, _, _, cond_module = _import_ldmc()
            self._wells = cond_module.load_wells_json(data)
        elif isinstance(data, list):
            self._wells = data
        else:
            # Fall back to parent for tensor conditioning
            return cast("DiffusionSimulator", super().set_conditioning(data))
        return self

    @property
    def supports_conditioning(self) -> bool:
        """Whether this simulator supports well conditioning."""
        return self.cond_ch > 0

    # =========================================================================
    # Diffusion-specific utility methods
    # =========================================================================

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent tensor to phase labels.

        Args:
            z: Latent tensor of shape (B, 3, 16, 16, 16).

        Returns:
            Phase labels of shape (B, 64, 64, 64) with values in {0, 1, 2}.
        """
        autoencoder = self._autoencoder
        assert autoencoder is not None
        logits = autoencoder.decode_stage_2_outputs(  # pyright: ignore[reportCallIssue]
            z / self._scale_factor
        )
        return logits.argmax(dim=1)

    @torch.no_grad()
    def encode_volume(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a volume to latent space.

        Args:
            x: One-hot encoded volume (B, num_classes, D, H, W).

        Returns:
            Latent tensor (B, 3, 16, 16, 16).
        """
        autoencoder = self._autoencoder
        assert autoencoder is not None
        return autoencoder.encode_stage_2_inputs(x) * self._scale_factor  # pyright: ignore[reportCallIssue]

    @torch.no_grad()
    def forward_diffusion(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple:
        """
        Forward diffusion process q(x_t | x_0).

        Args:
            x0: Clean latent data (B, 3, 16, 16, 16).
            t: Timestep indices (B,).

        Returns:
            Tuple of (x_t, noise) - noisy sample and the noise that was added.
        """
        scheduler = self._create_scheduler()
        noise = torch.randn_like(x0)
        x_t = scheduler.add_noise(x0, noise, t)
        return x_t, noise

    @torch.no_grad()
    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: int,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single denoising step in latent space.

        Args:
            x_t: Current noisy latent (B, 3, 16, 16, 16).
            t: Current timestep (integer).
            cond: Optional conditioning tensor.

        Returns:
            Denoised latent x_{t-1}.
        """
        device = x_t.device
        t_batch = torch.full(
            (x_t.shape[0],), t, device=device, dtype=torch.long
        )
        model = self._model
        assert model is not None

        if cond is not None and self.guidance_scale != 1.0:
            eps_cond = model(x_t, t_batch, cond=cond)
            eps_uncond = model(
                x_t, t_batch, cond=torch.zeros_like(cond)
            )
            eps = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
        elif cond is not None:
            eps = model(x_t, t_batch, cond=cond)
        else:
            eps = model(x_t, t_batch)

        scheduler = self._create_scheduler()
        prev_sample, _ = scheduler.step(eps, t, x_t)
        return prev_sample
