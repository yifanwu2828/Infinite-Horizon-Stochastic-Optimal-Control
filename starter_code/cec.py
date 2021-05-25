import numpy as np
from casadi import Opti, cos, sqrt
from icecream import ic


def CEC(
        car_states,
        ref_states,
        control_seq,
        error_seq,
        obstacles=None,
        verbose=False):
    """
    CEC is a suboptimal control scheme that applies,
    at each stage, the control that would be optimal
    if the noise variables w_t were fixed at their expected values
    :param car_states:
    :param ref_states:
    :param control_seq:
    :param error_seq:
    :param obstacles:
    :param verbose:
    :return:
    """
    circle1 = obstacles[0, :]  # [-2. , -2. ,  0.5]
    circle2 = obstacles[1, :]  # [1. , 2. , 0.5]

    rad1 = circle1[-1]
    rad2 = circle2[-1]

    # control Limit
    v_min, v_max = (0, 1)
    w_min, w_max = (-1, 1)

    # Hyperparameter
    q = 1
    Q = 1 * np.eye(2)
    R = 1 * np.eye(2)
    gamma = 0.99

    T = car_states.shape[1]
    r_t = ref_states[:2, :]

    # Optimization problem
    opti = Opti()
    # variable
    u = opti.variable(2, T)
    e = opti.variable(3, T)

    # opti.set_initial(u, control_seq)
    # opti.set_initial(e, error_seq)

    terminal_cost = 0

    pos_err = e[:2, :]
    ori_err = e[2, :]
    p = pos_err + r_t

    f = 0
    for i in range(T):
        a = pos_err[:, i].T @ Q @ pos_err[:, i]
        b = q * (1 - cos(ori_err[i])) ** 2
        c = u[:, i].T @ R @ u[:, i]
        if i == T - 1:
            f += terminal_cost
        else:
            f += gamma ** i * (a + b + c)

        # control limit
        opti.subject_to(opti.bounded(v_min, u[0, i], v_max))
        opti.subject_to(opti.bounded(w_min, u[1, i], w_max))

        # # position boundary
        # opti.subject_to(opti.bounded(-3, p[0, i], 3))
        # opti.subject_to(opti.bounded(-3, p[1, i], 3))
        #
        # # obstacles
        # opti.subject_to(
        #     sqrt((p[0, i] - circle1[0]) ** 2 + (p[1, i] - circle1[1]) ** 2) >= rad1
        # )
        # opti.subject_to(
        #     sqrt((p[0, i] - circle2[0]) ** 2 + (p[1, i] - circle1[1]) ** 2) >= rad2
        # )

    # ---- objective---------
    opti.minimize(f)

    p_opts = {"expand": False, "ipopt.print_level": 0}
    s_opts = {"max_iter": 100}
    opti.solver("ipopt", p_opts, s_opts)

    sol = opti.solve()
    if verbose:
        print(f"solved value: {sol.value(u)}")
    # ic(opti.debug.value)

    if T > 1:
        control = np.asarray(sol.value(u))[:, 0]
    else:
        control = np.asarray(sol.value(u))
    return control
