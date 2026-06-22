import pytest
import numpy as np
from utils.fdm import get_diffusion_differential_operator
from core.context import PhysicsContext, cfg

@pytest.fixture
def ctx():
    return PhysicsContext(cfg=cfg)

def test_diffucion_differential_operator(ctx):
    x, L, b = get_diffusion_differential_operator(m=100, x_domain=[0, 1], ctx=ctx, phase='uptake')
    # assert len(x) == 101, "spacial vector is of size (m+1, )."
    assert x.shape == (101,)
    assert L.shape == (101, 101)
    assert b.shape == (101, )
    print('shape test passed succesfully!!')