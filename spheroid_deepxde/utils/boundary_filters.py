import deepxde as dde

### Boundary location filters.
def initial_time_filter(x, on_initial):
    return on_initial

def center_filter(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.0)

def surface_filter(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1.0)