"""Input embeddings for PINN backbones.

Callables applied to raw (r, t) coordinates before they reach a BaseMLP
subclass, passed in as a backbone's `input_transformation`. Distinct from
`preprocessing/`, which handles data-side scaling of physical values --
these live in the network's forward pass (e.g. FourierFeatures is a
registered buffer, sized via `output_dim` when the backbone builds its
first layer).

    FourierFeatures  -- random Fourier feature embedding (eq. 4.3,
                         Wang, Sankaran, Wang & Perdikaris, 2023)
"""