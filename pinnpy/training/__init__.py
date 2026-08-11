"""Training techniques for physics-informed neural networks.

pinnpy.training collects optional, composable strategies for improving
PINN training beyond plain uniform-weighted MSE minimization -- each
one addresses a specific, well-documented failure mode of naive PINN
training (see each module's own docstring for the underlying paper
and motivation). None of these are required: a PINN trains fine
without any of them; they're opt-in improvements you reach for when
you hit the specific problem each one solves.

    causal.py     -- temporal causal weighting. Fixes: a PINN fitting
                      late timesteps before it has correctly learned
                      early ones, violating the physical causality that
                      information should only propagate forward in
                      time. Opt in via PINN(..., causal=True, n_chunks=...).

    weighting.py  -- gradient-norm adaptive loss balancing. Fixes: one
                      loss term (e.g. the PDE residual) dominating
                      training because its gradients are naturally
                      much larger than another term's (e.g. a boundary
                      condition), so equal nominal weights don't mean
                      equal influence. Opt in via a GradNormWeighter,
                      driven from Trainer(..., use_gradnorm=True).

    sampling.py   -- RAR-G (residual-based adaptive refinement).
                      Fixes: a fixed collocation set that never
                      concentrates points where the network is
                      currently violating the PDE most. Opt in by
                      calling training.sampling.rar_g(...) instead of
                      (or after) a plain Trainer.train(...) run.

These three compose: it's common to train with causal=True AND
use_gradnorm=True AND periodic RAR-G refinement, all at once, on the
same PINN.

Example -- causal weighting + gradnorm together:

    from pinnpy.pinns import ForwardPinn
    from pinnpy.trainer import Trainer
    import torch

    pinn = ForwardPinn(
        pde=my_pde, n_species=3,
        initial_conditions=[my_ic], boundary_conditions=[my_bc],
        causal=True, n_chunks=24, causal_eps=1.0,
    )
    trainer = Trainer(pinn, weights=my_weights, use_gradnorm=True, gradnorm_update_every=1000)
    optimizer = torch.optim.Adam(pinn.net.parameters(), lr=1e-3)
    trainer.train(optimizer, epochs=20_000, L=1.0)
    # trainer.weights has been rebalanced by gradnorm every 1000 steps;
    # pinn.meta['chunk_losses']/['chunk_weights'] hold the latest
    # causal-weighting diagnostics (see causal_plotting.py to visualize).

Example -- RAR-G refinement on top of an already-warmed-up PINN:

    from pinnpy.training.sampling import rar_g

    history = rar_g(
        pinn, trainer, optimizer,
        n_rounds=10, n_dense=20_000, m_add=500,
        rnd_epochs=1000, warmup_epochs=20_000, L=1.0,
    )
    # history: list of per-round dicts (n_points, max_score, mean_score)
    # -- confirms the refinement is actually chasing and reducing the
    # worst PDE residual round over round.
"""