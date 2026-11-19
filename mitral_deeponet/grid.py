"""Reference scales, model frame and branch-input grid of the published DeepONet (values from the training pipeline).

Non-dimensionalization (the network sees and returns only non-dimensional quantities):
    x* = x / L_C                 L_C   = 0.1 m          (all three coordinates)
    t* = (T mod T_C) / T_C       T_C   = 0.75 s         (phase within the cardiac cycle, 80 bpm; t* in [0, 1))
    u* = u / U_C                 U_C   = 4.988348484039307 m/s   (max |velocity component| over the CFD database)
    p* = p / P_C                 P_C   = 13439.0517578125 Pa     (max static pressure over the CFD database)
    nu_t* = nu_t / NU_T_C        NU_T_C = MU_T_C / RHO = 3.917e-4 m^2/s (max eddy viscosity over the database; the training
                                  pipeline divided the dynamic turbulent viscosity exported by STAR-CCM+ by MU_T_C =
                                  0.4254174828529358 Pa s, which gives the same non-dimensional value, RHO = 1086 kg/m^3)

Model frame: the rotated Cartesian frame of the CFD export, origin at the centre of the flat orifice opening, x along the
regurgitant jet (from LV into LA), z normal to the PIV / branch-input plane (z* = 0). For Slot-L-Bent the deformed plate moves
the opening, and the model was adapted in a frame shifted by SHIFT_SLOTLBENT: x*_model = x*_raw - 0.064 (6.4 mm).
"""
import numpy as np

L_C = 0.1
T_C = 0.75
U_C = 4.988348484039307
P_C = 13439.0517578125
MU_T_C = 0.4254174828529358
RHO = 1086.0
NU_T_C = MU_T_C / RHO
SHIFT_SLOTLBENT = 0.064

NX, NY = 119, 102                       # 17*7 x 17*6 equidistant grid
X_RANGE = (0.005, 0.69)                 # x* (0.05 ... 6.9 cm)
Y_RANGE = (-0.22, 0.22)                 # y* (-2.2 ... 2.2 cm)


def branch_grid(return_mask=False):
    """Coordinates (x*, y*) of the 10738 branch-input points, in the order the network expects.

    The 119 x 102 grid (np.meshgrid of x and y, row-major flatten, y slowest) is cropped by two straight lines through
    (x*, y*) = (0.69, 0.075) at -30 deg and (0.69, 0.01) at 60 deg, exactly as in the training pipeline. Returns xy of shape
    (10738, 2); with return_mask=True also the boolean mask of shape (NY, NX) on the full grid."""
    x = np.linspace(*X_RANGE, NX)
    y = np.linspace(*Y_RANGE, NY)
    X, Y = np.meshgrid(x, y)
    m1 = Y < 0.075 + np.tan(np.radians(-30)) * (X - 0.69)
    m2 = Y > 0.01 + np.tan(np.radians(60)) * (X - 0.69)
    mask = m1 & m2
    xy = np.c_[X[mask], Y[mask]]
    return (xy, mask) if return_mask else xy


def to_model(x_m=None, t_s=None, u_ms=None):
    """Convenience: dimensional SI values -> non-dimensional model values (any argument may be None)."""
    return (None if x_m is None else np.asarray(x_m) / L_C,
            None if t_s is None else (np.asarray(t_s) % T_C) / T_C,
            None if u_ms is None else np.asarray(u_ms) / U_C)


def to_si(out):
    """(N, 5) non-dimensional (u, v, w, p, nu_t) -> dict of SI fields (velocity m/s, p Pa, nu_t m^2/s)."""
    out = np.asarray(out, dtype='float64')
    return dict(u=out[:, 0] * U_C, v=out[:, 1] * U_C, w=out[:, 2] * U_C, p=out[:, 3] * P_C, nu_t=out[:, 4] * NU_T_C)
