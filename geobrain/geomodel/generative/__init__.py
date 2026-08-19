"""
Generative AI methods for geological modeling.

Status: **experimental**; these neural simulators may change before 1.0 (they
depend on external checkpoints and their API is shaped by the research workflow).
The simulator classes carry the ``**Experimental API**`` marker: they may
change in a minor release before they stabilize.

Implemented:
    - DiffusionSimulator: Latent Diffusion Model (VAE + UNet + DDPM/DDIM)
    - VAESimulator: 3D AutoencoderKL (encode, decode, interpolate)
    - GANSimulator: DCGAN-style 3D generator for facies

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .base import GenerativeConfig, GenerativeSimulator, SoftFieldDecoder
from .gans import GANSimulator, Generator3D
from .vae import VAESimulator
from .diffusion import DiffusionSimulator
from .checkpoints import ModelCard, load_verified_state_dict
from .dependencies import LDMC_PROVIDER, OptionalProvider, require_provider

__all__ = [
    "GenerativeConfig",
    "GenerativeSimulator",
    "SoftFieldDecoder",
    "GANSimulator",
    "Generator3D",
    "VAESimulator",
    "DiffusionSimulator",
    "LDMC_PROVIDER",
    "ModelCard",
    "OptionalProvider",
    "load_verified_state_dict",
    "require_provider",
]
