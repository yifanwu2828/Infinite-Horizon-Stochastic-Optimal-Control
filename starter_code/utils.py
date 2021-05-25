from time import time

import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from matplotlib import animation



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
            [cos(th), -sin(th)],
            [sin(th), cos(th)]
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
        sim.save('./fig/animation' + str(time()) + '.gif', writer='ffmpeg', fps=15)


# ----------------------------------------------------------------------------------

def simple_controller(cur_state, ref_state, v_limit=(0, 1), w_limit=(-1, 1)):
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


def error_dynamics(cur_state, ref_cur_state, ref_nxt_state, control, tau=0.5, Wt=None):
    """
    :param cur_state: [x_t, y_t, theta_t].T
    :param ref_cur_state:
    :param ref_nxt_state:
    :param control: control [v_t, w_t]
    :param tau: time_step
    :param Wt: Gaussian Motion Noise W_t
    :return: error next step
    """
    # assert isinstance(cur_state, np.ndarray)
    # assert isinstance(ref_cur_state, np.ndarray)
    # assert isinstance(ref_nxt_state, np.ndarray)
    # assert isinstance(control, np.ndarray)
    # assert isinstance(tau, float)
    cur_err = (cur_state - ref_cur_state).reshape(-1, 1)
    theta = cur_state[2]

    ref_diff = (ref_nxt_state - ref_cur_state).reshape(-1, 1)

    u = control.reshape(-1, 1)

    G = tau * np.array([
        [np.cos(theta), 0],
        [np.sin(theta), 0],
        [0,             1],
    ])

    if Wt is not None:
        assert isinstance(Wt, np.ndarray), "Noise should be np.ndarray"
        nxt_err = cur_err + G @ u + ref_diff + Wt.reshape(-1, 1)
    else:
        nxt_err = cur_err + G @ u + ref_diff
    return nxt_err


