from typing import Optional, Callable
from functools import lru_cache
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation



def tic(message: Optional[str] = None) -> float:
    """ Timing Function """
    if message:
        print(message)
    else:
        print("############ Time Start ############")
    return time.time()


def toc(t_start: float, name: Optional[str] = "Operation", ftime=False) -> None:
    """ Timing Function """
    assert isinstance(t_start, float)
    sec: float = time.time() - t_start
    if ftime:
        duration = time.strftime("%H:%M:%S", time.gmtime(sec))
        print(f'\n############ {name} took: {str(duration)} ############\n')
    else:
        print(f'\n############ {name} took: {sec:.4f} sec. ############\n')


############################################
############################################

def visualize(
        car_states: np.ndarray,
        ref_traj: np.ndarray,
        obstacles: np.ndarray,
        t: np.ndarray,
        time_step: float,
        save=False,
) -> None:
    init_state = car_states[0, :]

    def create_triangle(state=(0, 0, 0), h=0.5, w=0.25, update=False):
        x, y, th = state
        triangle = np.array([
            [h, 0],
            [0, w / 2],
            [0, -w / 2],
            [h, 0]
        ]).T
        rotation_matrix = np.array([
            [np.cos(th), -np.sin(th)],
            [np.sin(th), np.cos(th)]
        ])

        coords = np.array([[x, y]]) + (rotation_matrix @ triangle).T
        if update:
            return coords
        else:
            return coords[:3, :]

    def init():
        return path, current_state, target_state,

    def animate(i):
        # get variables
        x = car_states[i, 0]
        y = car_states[i, 1]
        th = car_states[i, 2]

        # update path
        if i == 0:
            path.set_data(np.array([]), np.array([]))
        x_new = np.hstack((path.get_xdata(), x))
        y_new = np.hstack((path.get_ydata(), y))
        path.set_data(x_new, y_new)

        # update horizon
        # x_new = car_states[0, :, i]
        # y_new = car_states[1, :, i]
        # horizon.set_data(x_new, y_new)

        # update current_state
        current_state.set_xy(create_triangle([x, y, th], update=True))

        # update current_target
        x_ref = ref_traj[i, 0]
        y_ref = ref_traj[i, 1]
        th_ref = ref_traj[i, 2]
        target_state.set_xy(create_triangle([x_ref, y_ref, th_ref], update=True))

        # update target_state
        # xy = target_state.get_xy()
        # target_state.set_xy(xy)            

        return path, current_state, target_state,

    circles = [plt.Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.5) for obs in obstacles]

    # create figure and axes
    fig, ax = plt.subplots(figsize=(9, 9))
    min_scale_x = min(init_state[0], np.min(ref_traj[:, 0])) - 1.5
    max_scale_x = max(init_state[0], np.max(ref_traj[:, 0])) + 1.5
    min_scale_y = min(init_state[1], np.min(ref_traj[:, 1])) - 1.5
    max_scale_y = max(init_state[1], np.max(ref_traj[:, 1])) + 1.5
    # TODO: change this back
    # ax.set_xlim(left=min_scale_x, right=max_scale_x)
    # ax.set_ylim(bottom=min_scale_y, top=max_scale_y)
    ax.set_xlim(left=-10, right=10)
    ax.set_ylim(bottom=-10, top=10)
    for circle in circles:
        ax.add_patch(circle)
    # create lines:
    #   path
    path, = ax.plot([], [], 'k', linewidth=2)

    # current_state
    current_triangle = create_triangle(init_state[:3])
    current_state = ax.fill(current_triangle[:, 0], current_triangle[:, 1], color='r')
    current_state = current_state[0]

    # target_state
    target_triangle = create_triangle(ref_traj[0, 0:3])
    target_state = ax.fill(target_triangle[:, 0], target_triangle[:, 1], color='b')
    target_state = target_state[0]

    # reference trajectory
    ax.scatter(ref_traj[:, 0], ref_traj[:, 1], marker='x')

    sim = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=len(t),
        interval=time_step * 100,
        blit=True,
        repeat=True
    )
    plt.show()

    if save:
        sim.save('./fig/animation' + str(time.time()) + '.gif', writer='ffmpeg', fps=15)


# ----------------------------------------------------------------------------------
@lru_cache(maxsize=1000)
def lissajous(k: int, time_step: float):
    """
    This function returns the reference point at time step k
    :param k: time step
    :param time_step:
    :return:
    """
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2 * np.pi / 50
    b = 3 * a
    T = np.round(2 * np.pi / (a * time_step))
    k = k % T
    delta = np.pi / 2
    xref = xref_start + A * np.sin(a * k * time_step + delta)
    yref = yref_start + B * np.sin(b * k * time_step)
    v = [A * a * np.cos(a * k * time_step + delta), B * b * np.cos(b * k * time_step)]
    thetaref = np.arctan2(v[1], v[0])
    return np.array([xref, yref, thetaref])


def simple_controller(cur_state, ref_state, v_limit=(0.0, 1.0), w_limit=(-1.0, 1.0)):
    """This function implements a simple P controller"""
    v_min, v_max = v_limit
    w_min, w_max = w_limit
    k_v = 0.55
    k_w = 1.0
    v = k_v * np.sqrt((cur_state[0] - ref_state[0])**2 + (cur_state[1] - ref_state[1])**2)
    v = np.clip(v, v_min, v_max)
    angle_diff = ref_state[2] - cur_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    w = k_w * angle_diff
    w = np.clip(w, w_min, w_max)
    return np.array([v, w])


def car_next_state(time_step: float, cur_state, control, noise=False):
    """
    The discrete-time kinematic model of the differential-drive robot
    :param time_step: \tau -> t
    :param cur_state: [x_t, y_t, theta_t].T
    :param control: [v_t, w_t]
    :param noise: Gaussian Motion Noise W_t
    :return: new_state
    """
    theta = cur_state[2]
    # Yaw
    rot_3d_z = np.array([
        [np.cos(theta), 0],
        [np.sin(theta), 0],
        [0,             1]])
    f = rot_3d_z @ control

    if noise:
        # Gaussian Motion Noise (w_t ∈ R^{3}) with N(0, diag(σ)^2 )
        # where σ = [0.04, 0.04, 0.004] ∈ R^{3}
        mu, sigma = 0, 0.04  # mean and standard deviation for (x,y)
        w_xy = np.random.normal(mu, sigma, 2)
        mu, sigma = 0, 0.004  # mean and standard deviation for theta
        w_theta = np.random.normal(mu, sigma, 1)
        w = np.concatenate((w_xy, w_theta))
        new_state = cur_state + time_step * f.flatten() + w
    else:
        new_state = cur_state + time_step * f.flatten()
    return new_state


def error_dynamics(cur_err, cur_ref, nxt_ref, ctrl, tau):
    """
    :param cur_err:
    :param cur_ref:
    :param nxt_ref:
    :param ctrl: control [v_t, w_t]
    :param tau: time_step
    :return: error next step
    """
    cur_err = cur_err.reshape(3, -1)
    theta_err = cur_err[2, 0]
    ref_diff = (cur_ref - nxt_ref).reshape(3, -1)

    a_t = cur_ref[2]
    u = ctrl.reshape(2, -1)

    G = np.array([
        [tau * np.cos(theta_err + a_t), 0],
        [tau * np.sin(theta_err + a_t), 0],
        [0,                           tau],
    ])
    nxt_err = cur_err + G @ u + ref_diff
    return nxt_err.reshape(-1)


def predict_T_step(cur_X: np.ndarray, traj: Callable, cur_iter: int, T: int, time_step: float):
    """
    :param cur_X: current car state
    :param traj: reference trajectory function
    :param cur_iter: idx to track traj
    :param T: number of steps
    :param time_step:
    :return:
    """
    state_seq = np.empty((3, T))
    ref_seq = np.empty((3, T))
    control_seq = np.empty((2, T))
    cur_error_seq = np.empty((3, T))
    nxt_error_seq = np.empty((3, T-1))

    # current state
    state = cur_X

    for i in range(T):
        state_seq[:, i] = state

        # Reference state
        cur_ref = traj(cur_iter + i, time_step=time_step)
        nxt_ref = traj(cur_iter + i + 1, time_step=time_step)
        ref_seq[:, i] = cur_ref

        # Control
        u = simple_controller(state, cur_ref)
        control_seq[:, i] = u

        # True error (p̃ t := p t − r t and θ̃ t := θ t − α t)
        true_error = state - cur_ref
        cur_error_seq[:, i] = true_error

        # Predict error (noise free)
        pred_nxt_error = error_dynamics(true_error, cur_ref, nxt_ref, u, tau=time_step)
        if i + 1 >= T-1:
            continue
        else:
            nxt_error_seq[:, i] = pred_nxt_error

        # Car next state (noise free)
        nxt_state = car_next_state(time_step, cur_state=state, control=u, noise=False)
        state = nxt_state


    return state_seq, ref_seq, control_seq, cur_error_seq, nxt_error_seq


# ----------------------------------------------------------------------------------


