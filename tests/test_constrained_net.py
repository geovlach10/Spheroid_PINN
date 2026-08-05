import torch
from pinnpy.neural_nets import FCNN
from pinnpy.constrained_net import ConstrainedNet
from trastuzumab.residuals import phi

BETA, CSOL, EPS = 2.3, 1.0, 0.01

def _cnet(enforce=('ic','neumann','robin'), seed=1):
    inner_net = FCNN(in_dim=2, out_dim=3, n_layers=4, n_neurons=16, seed=seed)
    return ConstrainedNet(inner_net=inner_net, beta=BETA, c_sol_star=CSOL, eps=EPS, enforce=enforce)   

def test_ic_zero_all_channels_untrained():
    net = _cnet()
    r = torch.rand(100, 1)
    for c in net(r, t=torch.zeros_like(r)):
        assert c.abs().max().item() < 1e-6, "ConstrainedNet output is not zero at t=0"

def test_non_trivial_away_from_zero():
    net = _cnet()
    r = torch.rand(32, 1)
    t = torch.rand(32, 1).clamp(min=0.1)
    c0, c1, c2 = net(r, t)
    assert (max(c.abs().max().item() for c in (c0, c1, c2)))> 1e-4, "ConstrainedNet output is trivial away from t=0"

def test_spatial_grad_includes_anchor():
    net = _cnet()
    r = torch.rand(8, 1, requires_grad=True)
    t = torch.full_like(r, 0.3)
    c0, _, _ = net(r, t)
    gr = torch.autograd.grad(c0.sum(), r)[0]
    eps = 1e-4
    with torch.no_grad():
        cp, _, _ = net(r + eps, t)
        cm, _, _ = net(r - eps, t)
        fd = (cp - cm) / (2 * eps)
    assert torch.allclose(gr, fd, atol=1e-2), "autograd d/dr disagrees with finite diff -> anchor r-gradient dropped"

def test_checkpoint_schema_unchanged():
    net = _cnet(seed=3)
    sd = net.state_dict()
    assert all(not k.startswith('inner.') for k in sd), "state_dict keys carry a submodule prefix -> delegation missing"
    new_fcnn = FCNN(in_dim=2, out_dim=3, n_layers=1, n_neurons=64, initialization='xavier_normal', seed=3)
    new_fcnn.load_state_dict(sd)


def test_neumann_holds_untrained():
    net = _cnet()
    t = torch.rand(128,1).clamp(min=0.02)
    r0 = torch.zeros_like(t).requires_grad_(True)
    c0,_,_ = net(r0, t)
    c0r = torch.autograd.grad(c0.sum(), r0)[0]
    assert c0r.abs().max().item() < 1e-5, "Neumann dc0/dr(0,t) != 0"

def test_robin_holds_untrained():
    net = _cnet()
    t = torch.rand(128,1).clamp(min=0.02)
    r1 = torch.ones_like(t).requires_grad_(True)
    c0,_,_ = net(r1, t)
    u0 = c0 / phi(r1)                    # phi(1)=1
    u0r = torch.autograd.grad(u0.sum(), r1)[0]
    lhs = u0r + BETA*u0
    gamma = BETA*CSOL*(1 - torch.exp(-t/EPS))
    assert torch.allclose(lhs, gamma, atol=1e-4), "Robin operator != gamma(t)"