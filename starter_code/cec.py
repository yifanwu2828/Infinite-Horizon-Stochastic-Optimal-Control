from typing import Optional
import numpy as np
from casadi import Opti, pi, sin, cos, sqrt, hcat, vcat
from icecream import ic


def CEC(
        X0: np.ndarray,
        ref_X: np.ndarray,
        obstacles: np.ndarray,
        control_seq: Optional[np.ndarray] = None,
        error_seq: Optional[np.ndarray] = None,
        verbose=False
) -> np.ndarray:
    """
    CEC is a suboptimal control scheme that applies,
    at each stage, the control that would be optimal
    if the noise variables w_t were fixed at their expected values(zero in this case)
    :param X0:
    :param ref_X: (3,T)
    :param obstacles: (2,3)
    :param control_seq:
    :param error_seq:
    :param verbose:
    :return: control (2,)
    """
    X0 = X0.reshape(3, -1)
    # obstacles
    circle1 = obstacles[0, :]  # [-2. , -2. ,  0.5]
    circle2 = obstacles[1, :]  # [1. , 2. , 0.5]
    rad1 = circle1[-1]
    rad2 = circle2[-1]

    # position boundary
    x_min, x_max = (-3, 3)
    y_min, y_max = (-3, 3)

    # orientation boundary (yaw angle)
    theta_min, theta_max = (-pi, pi - 1e-8)

    # control Limit
    v_min, v_max = (0.0,  1.0)  # linear velocity
    w_min, w_max = (-1.0, 1.0)  # angular velocity

    # time step
    tau = 0.5

    # Hyperparameter
    q = 1e1
    Q = 1e-2 * np.array([
        [1, 1],
        [1, 1],
    ])
    R = 7e-3 * np.array([
        [1, 0],
        [0, 1],
    ])

    pQp = error_seq[:2, 0].T @ Q @ error_seq[:2, 0]
    uRu = control_seq[:, 0].T @ R @ control_seq[:, 0]
    cos_sq = q * (1 - cos(error_seq[2, :][0])) ** 2
    ic(pQp)
    ic(cos_sq)
    ic(uRu)
    # assert pQp >= 0
    # assert cos_sq >= 0
    # assert uRu >= 0

    gamma = 0.99
    T = ref_X.shape[1]

    # Optimization problem
    opti = Opti()
    # variable
    u = opti.variable(2, T)
    e = opti.variable(3, T)

    # set initial value of variable
    # opti.set_initial(u, control_seq)
    # opti.set_initial(e[:, 0], error_seq[:, 0])

    f = 0
    for i in range(T):
        p_t = e[:2, i]
        theta_t = e[2, i]
        u_t = u[:, i]

        pos_err = e[:2, i]
        ori_err = e[2, i]
        r_t = ref_X[:2, i]
        a_t = ref_X[2, i]

        a = p_t.T @ Q @ p_t
        b = q * (1 - cos(theta_t)) ** 2
        c = u_t.T @ R @ u_t

        if i == T - 1 and i != 0:
            # terminal cost
            f += 0
        else:
            f += gamma ** i * (a + b + c)

            # control limit constraint
            opti.subject_to(opti.bounded(v_min, u[0, i], v_max))
            opti.subject_to(opti.bounded(w_min, u[1, i], w_max))

            # position boundary constraint
            opti.subject_to(opti.bounded(x_min, (pos_err + r_t)[0], x_max))
            opti.subject_to(opti.bounded(y_min, (pos_err + r_t)[1], y_max))
            # orientation boundary constraint
            opti.subject_to(opti.bounded(theta_min, ori_err, theta_max))

            # obstacles collision constraint
            opti.subject_to(
                sqrt(((pos_err + r_t)[0] - circle1[0]) ** 2 + ((pos_err + r_t)[1] - circle1[1]) ** 2) >= rad1
            )
            opti.subject_to(
                sqrt(((pos_err + r_t)[0] - circle2[0]) ** 2 + ((pos_err + r_t)[1] - circle1[1]) ** 2) >= rad2
            )

            x = hcat([tau * cos(ori_err), 0])
            y = hcat([tau * sin(ori_err), 0])
            z = hcat([0, tau])

            G = vcat([x, y, z])
            ref_diff = (ref_X[:, i] - ref_X[:, i + 1])

            # error constraint (error dynamics)
            nxt_err = e[:, i] + G @ u[:, i] + ref_diff + 0
            opti.subject_to(e[:, i+1] == nxt_err)



    # ---- objective---------
    opti.minimize(f)

    p_opts = {"expand": True, "ipopt.print_level": 0}
    s_opts = {"max_iter": 100}
    opti.solver("ipopt", p_opts, s_opts)

    sol = opti.solve()
    if verbose:
        print(f"solved value: {sol.value(u)}")
    if T > 1:
        control = np.asarray(sol.value(u))[:, 0]
    else:
        control = np.asarray(sol.value(u))

    return control



