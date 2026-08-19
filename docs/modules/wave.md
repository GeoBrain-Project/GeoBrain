# Wave physics (`geobrain.physics.wave`)

Seismic wave propagation, reflectivity, and the wavelets that drive them.

| Component | What is there |
|---|---|
| Time domain | `Acoustic2D/3D`, `Elastic2D/3D`, `ViscoAcoustic2D`, `ViscoElastic2D`, `AnisotropicElastic2D/3D` |
| Frequency domain | `Helmholtz2D`, with an implicit adjoint |
| Reflectivity | `Zoeppritz`, `AkiRichards`, `Shuey`, `ConvolutionalAVO`, `Convolutional1D` |
| Wavelets | `ricker`, `GaussianWavelet`, `OrmsbyWavelet`, `KlauderWavelet`, `create_wavelet`, `shared_wavelet` |
| Surveys | `Seismic2DSurvey`, `Seismic3DSurvey`, `Helmholtz2DSurvey` |
| Configuration | `WaveSimulationConfig`, `WaveBoundaryConfig`, `WaveDiscretizationConfig`, `WaveOutputConfig`, `WaveMemoryConfig`, `WaveBackendConfig` |

A wavelet and a reflectivity kernel. Between them, a convolutional forward
model:

```python
import torch
from geobrain.physics.wave import AkiRichards, ricker

wavelet = ricker(64, 0.002, 25.0, dtype=torch.float64)
print("wavelet", tuple(wavelet.shape), "peak at sample", int(wavelet.argmax()))

avo = AkiRichards(angles_deg=[0.0, 15.0, 30.0])
spec = avo.differentiability
print("AkiRichards:", spec.trainable_inputs, "->", spec.output_keys)
```

```text
wavelet (64,) peak at sample 31
AkiRichards: ('vp', 'vs', 'rho') -> ('reflectivity',)
```

`ricker` is zero-phase by default: the peak sits in the middle of the
window, not at its start, and `causal=True` shifts it if that is what your
recording convention expects.

Note what the kernel asks for: `vp`, `vs` **and** `rho`. Put it at the end of a
chain that starts from `vp` alone and the chain's contract still says `vp`:
the other two are produced upstream by rock physics, and the chain parameterizes
on its entry link.

```{figure} /_figures/01_fwi_climb.gif
:class: full-width
:alt: FWI sharpening as the frequency bands are added

Full-waveform inversion climbing the frequency bands, which is the one place
in the gallery where time is genuinely the subject. From
`examples/03_physics/01_seismic_fwi.py`.
```

## The engines

The time-domain engines take a survey and a wavelet and return a gather. They
declare an adjoint rather than taping every timestep, which is what makes FWI
on a real model affordable.

```{note}
`Helmholtz2D` declares `IMPLICIT_VJP`: the forward pass solves a linear system,
and the gradient comes from an implicit-function adjoint at the price of one
extra solve, not a tape through the factorisation.
```

`WaveOutputConfig` selects what comes back. Ask for the seismic data and you
get the data; asking for wavefield snapshots is a separate, heavier request.

## See also

- `examples/03_physics/01_seismic_fwi.py`: multi-scale FWI on Marmousi II,
  2 to 11 Hz.
- `examples/00_showcase/02_operator_composition.py`: an acoustic chain and a
  gravity chain as two channels of one bundle, sharing a rock-physics link.
- `examples/01_architecture/03_composition_rules.py`: Aki-Richards as the
  terminal link, and what the chain derives from it.
