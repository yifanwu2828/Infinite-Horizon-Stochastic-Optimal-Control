from typing import Optional
import numpy as np
from casadi import Opti, pi, sin, cos, hcat, vcat
from icecream import ic
import matplotlib.pyplot as plt

# Position boundary
x_min, x_max = (-3, 3)
y_min, y_max = (-3, 3)

# Orientation boundary (yaw angle)
theta_min, theta_max = (-pi, pi)

# Control Limit
v_min, v_max = (0.0, 1.0)  # linear velocity
w_min, w_max = (-1.0, 1.0)  # angular velocity

'''
# Hyperparameter
    q = 1.5
    Q = np.array([
        [1, 0],
        [0, 1],
    ])
    R = np.array([
        [10, 0],
        [0, 1],
    ])

# Hyperparameter
    q = 100
    Q = np.array([
        [1, 0],
        [0, 2],
    ])
    R = np.array([
        [5, 0],
        [0, 2],
    ])
'''



def PSD_check(Q, R, q, p, u, theta):

    pQp = p.T @ Q @ p
    uRu = u.T @ R @ u
    cos_sq = q * (1 - cos(theta)) ** 2
    assert pQp >= 0
    assert uRu >= 0
    assert cos_sq >= 0


def CEC(
        cur_states: np.ndarray,
        ref_X: np.ndarray,
        obstacles: np.ndarray,
        tau: float,
        control_seq: Optional[np.ndarray] = None,
        error_seq: Optional[np.ndarray] = None,
        verbose=False
) -> np.ndarray:
    """
    CEC is a suboptimal control scheme that applies,
    at each stage, the control that would be optimal
    if the noise variables w_t were fixed at their expected values(zero in this case)
    :param cur_states:
    :param ref_X: (3,T)
    :param obstacles: (2,3)
    :param tau : time_step
    :param control_seq:
    :param error_seq:
    :param verbose:
    :return: control (2,)
    """

    # Obstacles
    circle1 = obstacles[0, :]  # [-2. , -2. ,  0.5]
    circle2 = obstacles[1, :]  # [1. , 2. , 0.5]
    rad1 = circle1[-1]
    rad2 = circle2[-1]

    # total number of control intervals
    T = ref_X.shape[1]  # 10

    # Current State
    X0 = cur_states[:3, 0].reshape(3, -1)
    # Current ref
    ref_X0 = ref_X[:3, 0].reshape(3, -1)
    # Current
    e0 = X0 - ref_X0

    # Hyperparameter
    q = 1.5
    Q = np.array([
        [1, 0],
        [0, 1],
    ])
    R = np.array([
        [10, 0],
        [0, 1],
    ])

    gamma = 0.95
    # PSD_check(Q, R, q, p=error_seq[:2, 0], u=control_seq[:, 0], theta=error_seq[2, 0])

    # Optimization problem
    opti = Opti()

    # variable
    u = opti.variable(2, T)     # (2,10)
    e = opti.variable(3, T-1)   # (3, 9)
    X = e + ref_X[:, 1:]        # (3, 9)

    # set initial value of variable
    # opti.set_initial(u, control_seq)
    # opti.set_initial(e, error_seq)

    f = e0[:2].T @ Q @ e0[:2] + q*(1 - e0[2])**2 + u[:, 0].T @ R @ u[:, 0]

    theta_0 = e0[2]
    a_0 = ref_X[2, 0]

    G_0 = np.array([
        [tau * cos(theta_0 + a_0), 0],
        [tau * sin(theta_0 + a_0), 0],
        [0, tau]
    ])
    ref_diff_0 = ref_X[:, 0] - ref_X[:, 1]
    e1 = e0 + G_0 @ u[:, 0] + ref_diff_0
    opti.subject_to(e[:, 0] == e1)


    for i in range(T):
        if i == T-1:
            break
        p_t = e[:2, i]
        theta_t = e[2, i]
        u_t = u[:, i+1]

        pQp = p_t.T @ Q @ p_t
        q_sq = q * (1 - cos(theta_t)) ** 2
        uRu = u_t.T @ R @ u_t


        if i == T-2:
            # terminal cost
            f += pQp + q_sq
        else:
            f += (gamma ** (i+1)) * (pQp + q_sq + uRu)

            a_t = ref_X[2, i+1]

            x = hcat([tau * cos(theta_t + a_t), 0])
            y = hcat([tau * sin(theta_t + a_t), 0])
            z = hcat([0, tau])
            G = vcat([x, y, z])
            ref_diff = ref_X[:, i] - ref_X[:, i + 1]

            # error constraint (error dynamics)
            nxt_err = e[:, i] + G @ u[:, i] + ref_diff + 0
            opti.subject_to(e[:, i + 1] == nxt_err)

    # control constraint
    opti.subject_to(opti.bounded(v_min, u[0, :], v_max))
    opti.subject_to(opti.bounded(w_min, u[1, :], w_max))

    # state constraint
    opti.subject_to(opti.bounded(x_min, X[0, :], x_max))
    opti.subject_to(opti.bounded(y_min, X[1, :], y_max))

    # obstacles collision constraint
    opti.subject_to(
        (X[0, :] - circle1[0]) ** 2 + (X[1, :] - circle1[1]) ** 2 > rad1**2
    )
    opti.subject_to(
        (X[0, :] - circle2[0]) ** 2 + (X[1, :] - circle1[1]) ** 2 > rad2**2
    )

    # ---- objective---------
    opti.minimize(f)

    p_opts = {"expand": True, "ipopt.print_level": 0}
    # p_opts = {"expand": True}
    s_opts = {"max_iter": 100}
    opti.solver("ipopt", p_opts, s_opts)

    # sol1 = opti.solve()
    # print(sol1.stats()["iter_count"])
    # opti.set_initial(sol1.value_variables())

    control = None
    try:
        sol2 = opti.solve()
        print(f"NLP itr: {sol2.stats()['iter_count']}")
        control = sol2.value(u)[:, 0]
        if verbose:
            print(f"solved value: {sol2.value(u)}")
    except RuntimeError:
        ic(opti.debug.value(u)[:, 0])

    return control



