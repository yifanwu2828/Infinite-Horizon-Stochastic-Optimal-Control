import numpy as np
from casadi import Opti, cos, sqrt, vertcat
from icecream import ic


def CEC(
        ref_states,
        control_seq,
        error_seq,
        obstacles=None,
        verbose=False
) -> np.ndarray:
    """
    CEC is a suboptimal control scheme that applies,
    at each stage, the control that would be optimal
    if the noise variables w_t were fixed at their expected values
    :param ref_states:
    :param control_seq:
    :param error_seq:
    :param obstacles:
    :param verbose:
    :return:
    """
    # obstacles
    circle1 = obstacles[0, :]  # [-2. , -2. ,  0.5]
    circle2 = obstacles[1, :]  # [1. , 2. , 0.5]
    rad1 = circle1[-1]
    rad2 = circle2[-1]

    # control Limit
    v_min, v_max = (0.0,  1.0)
    w_min, w_max = (-1.0, 1.0)

    # Hyperparameter
    q = 1e-6
    Q = 1e-6 * np.array([
        [1, -1],
        [-1, 1],
    ])
    R = 1e-6 * np.array([
        [1, -1],
        [-1, 1],
    ])
    gamma = 1

    pQp = error_seq[:2, 0].T @ Q @ error_seq[:2, 0]
    uRu = control_seq[:, 0].T @ R @ control_seq[:, 0]
    cos_sq = q * (1 - cos(error_seq[2, :][0])) ** 2
    ic(pQp)
    ic(cos_sq)
    ic(uRu)
    assert pQp >= 0
    assert cos_sq >= 0
    assert uRu >= 0

    T = error_seq.shape[1]
    r_t = ref_states[:2, :]

    # Optimization problem
    opti = Opti()
    # variable
    u = opti.variable(2, T)
    e = opti.variable(3, T)

    # set initial value of variable
    opti.set_initial(u, control_seq)
    opti.set_initial(e, error_seq)

    terminal_cost = 0

    f = 0
    for i in range(T):
        a = e[:2, i].T @ Q @ e[:2, i]
        b = q * (1 - cos(e[2, i])) ** 2
        c = u[:, i].T @ R @ u[:, i]
        if i == T - 1 and T != 1:
            f += terminal_cost
        else:
            f += gamma ** i * (a + b + c)

        # control limit
        opti.subject_to(opti.bounded(v_min, u[0, i], v_max))
        opti.subject_to(opti.bounded(w_min, u[1, i], w_max))

        # # position boundary
        opti.subject_to(opti.bounded(-3, (e[:2, :] + r_t)[0, i], 3))
        opti.subject_to(opti.bounded(-3, (e[:2, :] + r_t)[1, i], 3))

        # obstacles
        opti.subject_to(
            sqrt(((e[:2, :] + r_t)[0, i] - circle1[0]) ** 2 + ((e[:2, :] + r_t)[1, i] - circle1[1]) ** 2) >= rad1
        )
        opti.subject_to(
            sqrt(((e[:2, :] + r_t)[0, i] - circle2[0]) ** 2 + ((e[:2, :] + r_t)[1, i] - circle1[1]) ** 2) >= rad2
        )

    # ---- objective---------
    opti.minimize(f)

    p_opts = {"expand": False, "ipopt.print_level": 0}
    s_opts = {"max_iter": 100}
    opti.solver("ipopt", p_opts, s_opts)

    sol = opti.solve()
    if verbose:
        print(f"solved value: {sol.value(u)}")
    ic(sol.value(u))
    if T > 1:
        control = np.asarray(sol.value(u))[:, 0]
    else:
        control = np.asarray(sol.value(u))
    return control
