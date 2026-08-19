# Neural networks (`geobrain.nn`)

Not a separate mode, and not a solver. A network here is an **operator you
compose in front of the physics with `@`**, which changes what the unknowns
are without touching the forward model.

| Piece | What is there |
|---|---|
| Reparameterization | `WeightReparameterization`, `LatentReparameterization` |
| Decoders | `ConvDecoder2d`, `ConvDecoder3d`, `CoordinateMLP` |
| Variational layers | `LinearFlipout`, `Conv2dFlipout`, `Conv3dFlipout`, `BaseVariationalLayer` |
| KL machinery | `get_kl_loss`, `kl_regularizer`, `gaussian_kl`, `count_variational_parameters` |
| Utilities | `Reshape`, `ClippedLinearActivation` |

## Three answers to "what are the unknowns?"

```python
# sketch: runnable version in examples/00_showcase/06
explicit = seismic                                     # the image itself
network  = seismic @ WeightReparameterization(decoder, ...)
latent   = seismic @ LatentReparameterization(decoder, ...)
```

| | The unknown is | The prior is |
|---|---|---|
| **Explicit** | the property image, one number per cell | none, so wherever the data are weak, the answer is whatever the optimiser drifted into |
| **Network** | a decoder's **weights**, fed a frozen random code | whatever that architecture can draw. This is the deep image prior |
| **Latent** | the **code**, decoder frozen | the strongest of the three: the image cannot leave the decoder's range |

All three then go through the same `InverseProblem` and the same
`create_inverter().run()`. Not one line of the forward model changes between
them, and the chain reports the switch honestly: ask the network version for
its trainable inputs and it answers with the decoder's weight names.

## What the comparison actually shows

```{admonition} The unknown count does not order the results
:class: warning

In the worked example the deep image prior wins with the **most** unknowns, and
the latent run beats the explicit one with a fifth of them. The prior is not a
term in the objective; it is the set of images the decoder can draw at all.
```

And the result is a property of that earth, not a law. A convolutional prior is
built for fields that are smooth on the scale the data resolve; where bedding
is finer than the wavelength, the same prior costs resolution instead of buying
it. The claim worth making is narrower and more useful: trying all three cost
three lines and one `@` apiece, on the same physics, through the same door.

```{figure} /_figures/06_neural_network_integration.png
:class: gb-tall
:alt: One inversion, three parameterizations

The same inversion run with the unknown as an image, as a decoder's weights,
and as its latent code. From
`examples/00_showcase/06_neural_network_integration.py`.
```

## See also

- `examples/00_showcase/06_neural_network_integration.py`: all three
  parameterizations on one seismic problem, scored against the truth.
