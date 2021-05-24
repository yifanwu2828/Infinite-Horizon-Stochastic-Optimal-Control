import numpy as np
from casadi import Opti, cos
from icecream import ic


def CEC(cur_state, cur_ref, obstacles, u_init=(0, 0), verbose=False):

    rad = 0.5
    v_min, v_max = (0, 1)
    w_min, w_max = (-1, 1)
    e = cur_state - cur_ref

    # Hyperparameter
    q = 1e-9
    Q = 1e-9*np.eye(2)
    R = 1e-9 * np.eye(2)
    gamma = 0.99

    # Optimization problem
    opti = Opti()
    # variable
    u = opti.variable(2)
    pos_err = opti.variable(2)
    ori_err = opti.variable(1)

    opti.set_initial(pos_err, e[:2])
    opti.set_initial(ori_err, e[2])
    # opti.set_initial(u, np.asarray(u_init))

    # parameter
    r_t = opti.parameter(2)
    opti.set_value(r_t, cur_ref[:2])

    # ---- objective---------
    a = pos_err.T @ Q @ pos_err
    b = q * (1 - cos(ori_err)) ** 2
    c = u.T @ R @ u
    opti.minimize(gamma*(a + b + c))

    # control boundary
    opti.subject_to(opti.bounded(v_min, u[0], v_max))
    opti.subject_to(opti.bounded(w_min, u[1], w_max))

    # boundary
    opti.subject_to(opti.bounded(-3, (pos_err + r_t)[0], 3))
    opti.subject_to(opti.bounded(-3, (pos_err + r_t)[1], 3))

    # obstacles
    # opti.subject_to(
    #     ((pos_err + r_t)[0] - obstacles[0, :][0]) ** 2 + ((pos_err + r_t)[1] - obstacles[0, :][1]) ** 2 > rad**2
    # )
    # opti.subject_to(
    #     ((pos_err + r_t)[0] - obstacles[1, :][0]) ** 2 + ((pos_err + r_t)[1] - obstacles[1, :][1]) ** 2 > rad ** 2
    # )



    # p_opts = {"expand": True}
    # s_opts = {"max_iter": 100}
    # opti.solver("ipopt", p_opts, s_opts)
    opti.solver("ipopt")

    sol = opti.solve()
    if verbose:
        print(f"solved value: {sol.value(u)}")
    return np.asarray(sol.value(u))
