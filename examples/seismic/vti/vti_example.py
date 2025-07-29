import numpy as np

try:
    import pytest
except ImportError:
    pass

from devito import Function, smooth, norm, info, Constant
from examples.seismic.plotting import plot_shotrecord
from examples.seismic import demo_model, setup_geometry, seismic_args
from examples.seismic.vti import VTIWaveSolver
import matplotlib.pyplot as plt


def vti_setup(
    shape=(301, 301),
    spacing=(10.0, 10.0),
    tn=250.0,
    kernel="centered",
    space_order=4,
    nbl=50,
    preset="layers-tti",
    vti=True,
    **kwargs,
):
    # Two layer model for true velocity
    model = demo_model(
        preset,
        shape=shape,
        spacing=spacing,
        space_order=space_order,
        nbl=nbl,
        vti=vti,
        **kwargs,
    )
    # Source and receiver geometries
    geometry = setup_geometry(model, tn, f0=0.02)

    return VTIWaveSolver(model, geometry, space_order=space_order, kernel=kernel, **kwargs)


def run(
    shape=(50, 50),
    spacing=(20.0, 20.0),
    tn=250.0,
    autotune=False,
    space_order=4,
    nbl=0,
    preset="layers-tti",
    kernel="centered",
    full_run=False,
    checkpointing=False,
    **kwargs,
):

    solver = vti_setup(
        shape=shape,
        spacing=spacing,
        tn=tn,
        space_order=space_order,
        nbl=nbl,
        kernel=kernel,
        preset=preset,
        **kwargs,
    )

    info("Applying Forward")
    save = full_run and not checkpointing
    rec, u, summary = solver.forward(save=save)

    # if not full_run:
    #     return summary.gflopss, summary.oi, summary.timings, solver.model, [rec, u]

    # # Smooth velocity
    # initial_vp = Function(name='v0', grid=solver.model.grid, space_order=space_order)
    # smooth(initial_vp, solver.model.vp)
    # dm = np.float32(initial_vp.data**(-2) - solver.model.vp.data**(-2))

    # info("Applying Adjoint")
    # solver.adjoint(rec, autotune=autotune)
    # info("Applying Born")
    # solver.jacobian(dm, autotune=autotune)
    # info("Applying Gradient")
    # solver.jacobian_adjoint(rec, u, autotune=autotune, checkpointing=checkpointing)

    return summary.gflopss, summary.oi, summary.timings, [rec, u]


@pytest.mark.parametrize("shape", [(51, 51), (16, 16)])
@pytest.mark.parametrize("kernel", ["centered"])
def test_tti_stability(shape, kernel):
    spacing = tuple([20] * len(shape))
    _, _, _, [rec, _, _] = run(shape=shape, spacing=spacing, kernel=kernel, tn=16000.0, nbl=0)
    assert np.isfinite(norm(rec))


if __name__ == "__main__":
    description = "Example script to execute a TTI forward operator."
    parser = seismic_args(description)
    parser.add_argument(
        "-k",
        dest="kernel",
        default="centered",
        choices=["centered"],
        help="Choice of finite-difference kernel",
    )
    args = parser.parse_args()

    preset = "constant-vti"

    # Preset parameters
    ndim = 2
    shape = (301, 301)
    spacing = tuple(ndim * [10.0])
    tn = args.tn if args.tn > 0 else (350.0 if ndim < 3 else 1250.0)

    gflop, oi, timings, par = run(
        shape=shape,
        spacing=spacing,
        nbl=args.nbl,
        tn=tn,
        space_order=args.space_order,
        autotune=args.autotune,
        dtype=args.dtype,
        opt=args.opt,
        kernel=args.kernel,
        preset=preset,
        checkpointing=args.checkpointing,
        full_run=args.full,
        vti=True,
    )

    plot_shotrecord(par[1].data[0], 0, tn)
