from time import time

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import animation

from icecream import ic


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
        sim.save('./fig/animation' + str(time()) + '.gif', writer='ffmpeg', fps=15)


# ----------------------------------------------------------------------------------
def lissajous(k, tau):
    """
    This function returns the reference point at time step k
    :param k: time step
    :param tau:
    :return:
    """
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2 * np.pi / 50
    b = 3 * a
    a_t = a * tau
    T = np.round(2 * np.pi / a_t)

    k = k % T
    delta = np.pi / 2
    a_k_t = k * a_t
    b_k_t = b * k * tau

    xref = xref_start + A * np.sin(a_k_t + delta)
    yref = yref_start + B * np.sin(b_k_t)
    v = [A*a*np.cos(a_k_t + delta), B*b*np.cos(b_k_t)]
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


def car_next_state(time_step, cur_state, control, noise=True):
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


def error_dynamics(cur_error, ref_cur_state, ref_nxt_state, control, tau, noise=False):
    """
    :param cur_error:
    :param ref_cur_state:
    :param ref_nxt_state:
    :param control: control [v_t, w_t]
    :param tau: time_step
    :param noise: if True: add Gaussian Motion Noise W_t
    :return: error next step
    """
    assert isinstance(cur_error, np.ndarray)
    assert isinstance(ref_cur_state, np.ndarray)
    assert isinstance(ref_nxt_state, np.ndarray)
    assert isinstance(control, np.ndarray)
    assert isinstance(tau, float)

    cur_error = cur_error.reshape(3, -1)
    theta_err = cur_error[2, 0]
    ref_diff = (ref_cur_state - ref_nxt_state).reshape(3, -1)

    u = control.reshape(2, -1)

    G = np.array([
        [tau * np.cos(theta_err), 0],
        [tau * np.sin(theta_err), 0],
        [0,                     tau],
    ])
    w = 0
    if noise:
        '''
        Gaussian Motion Noise (w_t ∈ R^{3}) with N(0, diag(σ)^2 )
        where σ = [0.04, 0.04, 0.004] ∈ R^{3}
        '''
        # mean and standard deviation for (x,y)
        mu_xy, sigma_xy = 0, 0.04
        w_xy = np.random.normal(mu_xy, sigma_xy, 2)
        # mean and standard deviation for theta
        mu_theta, sigma_theta = 0, 0.004
        w_theta = np.random.normal(mu_theta, sigma_theta, 1)
        w = np.concatenate((w_xy, w_theta))
    nxt_err = cur_error + G @ u + ref_diff + w
    return nxt_err


def predict_T_step(cur_state, traj, cur_iter, T, time_step):
    """

    :param cur_state: current car state
    :param traj: reference trajectory
    :param cur_iter: idx to track traj
    :param T: number of steps
    :param time_step:
    :return:
    """
    state_seq = []
    ref_seq = []
    act_seq = []
    true_error_seq = []
    pred_error_seq = []

    # current state
    state = cur_state
    for i in range(T):
        state_seq.append(state)

        # Reference state
        cur_ref = traj(cur_iter + i, tau=time_step)
        ref_seq.append(cur_ref)

        # Control
        u = simple_controller(state, cur_ref)
        act_seq.append(u)

        # True error (p̃ t := p t − r t and θ̃ t := θ t − α t)
        true_error = state - cur_ref
        true_error_seq.append(true_error)

        # # Predict error (noise free)
        # pred_nxt_error = utils.error_dynamics(true_error, cur_ref, traj(cur_iter + i+1), u, tau=time_step)
        # if i == 0:
        #     pred_error_seq.append(true_error)
        # pred_error_seq.append(pred_nxt_error)

        # Car next state (noise free)
        nxt_state = car_next_state(time_step, state, u, noise=False)
        state = nxt_state

    state_seq = np.array(state_seq).reshape(3, -1)
    ref_seq = np.array(ref_seq).reshape(3, -1)
    act_seq = np.array(act_seq).reshape(2, -1)
    true_error_seq = np.array(true_error_seq).reshape(3, -1)
    # pred_error_seq = np.array(true_error_seq).reshape(3, -1)

    return state_seq, ref_seq, act_seq, true_error_seq, pred_error_seq


# ----------------------------------------------------------------------------------

class MDP(object):
    def __init__(
            self,
            dt: float,
            t: int = 100,  # should be fixed at 100 since reference trajectory is periodic 100
            x_lim=(-3, 3), y_lim=(-3, 3), theta_limit=(-np.pi, np.pi),
            v_lim=(-0, 1), w_lim=(-1, 1),
            gamma: float = 0.99,
            res: float = 0.1
    ):
        # Discrete time horizon
        self.dt = dt
        self.t = np.arange(0, t, self.dt)

        # position
        self.x_min, self.x_max = x_lim
        self.y_min, self.y_max = y_lim
        # orientation
        self.theta_min, self.theta_max = theta_limit

        # velocity
        self.v_min, self.v_max = v_lim
        self.w_min, self.w_max = w_lim
        self.gamma = gamma
        self.res = res

        nX = np.ceil(((self.x_max - self.x_min) / self.res) + 1).astype('int')
        nY = np.ceil(((self.y_max - self.y_min) / self.res) + 1).astype('int')
        nTheta = np.ceil(((self.theta_max - self.theta_min) / self.res) + 1).astype('int')
        self.nS = nX * nY * nTheta

        nV = np.ceil(((self.v_max - self.v_min) / self.res) + 1).astype('int')
        nW = np.ceil(((self.w_max - self.w_min) / self.res) + 1).astype('int')
        self.nA = nV * nW

        self.x = np.linspace(self.x_min, self.x_max, nX)
        self.y = np.linspace(self.y_min, self.y_max, nY)
        self.theta = np.linspace(self.theta_min, self.theta_max, nTheta)  # endpoint=False)

        self.v = np.linspace(self.v_min, self.v_max, nV)  # linear velocity
        self.w = np.linspace(self.w_min, self.w_max, nW)  # angular velocity


    @staticmethod
    def wrap_angle(angles):
        n = int(np.abs((angles / np.pi)))
        if n % 2 == 0:
            angles = angles - n * np.pi * np.sign(angles)
        else:
            angles = angles - (n + 1) * np.pi * np.sign(angles)
        return angles

    def build_MDP(self):
        # P(s, a, s')
        P = np.zeros((self.nS, self.nA, self.nS))
        # L(x, u)
        L = np.zeros((self.nS, self.nA))


if __name__ == '__main__':
    mdp = MDP(dt=0.5)
