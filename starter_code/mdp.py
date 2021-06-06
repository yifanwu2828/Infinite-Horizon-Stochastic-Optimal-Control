import sys
import time
from functools import lru_cache
from math import pi

import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import softmax
from matplotlib import pyplot as plt
from numba import jit
from tqdm import tqdm
from icecream import ic

import gaussian
import utils

class MDP(object):
    def __init__(
            self,
            dt: float = 0.5,
            nT: int = 100,  # should be fixed at 100 since reference trajectory is periodic 100
            x_lim=(-3, 3), y_lim=(-3, 3), theta_limit=(-pi, pi),
            v_lim=(0, 1), w_lim=(-1, 1),
            sigma=(0.04, 0.04, 0.004),
            gamma: float = 0.99,
            res: float = 0.3,
            expert='ref_traj.npy',
            init_V0=None,
            init_PI=None,
    ):
        # load traj
        if expert:
            with open(f'{expert}', 'rb') as f:
                self.traj = np.load(f)

        self.init_V0=init_V0
        self.init_PI=init_PI

        # Discrete time horizon
        self.dt = dt  # time step
        self.nT = nT

        # Position
        self.x_min, self.x_max = x_lim
        self.y_min, self.y_max = y_lim
        # Orientation
        self.theta_min, self.theta_max = theta_limit

        # Linear&Angular velocity
        self.v_min, self.v_max = v_lim
        self.w_min, self.w_max = w_lim

        # discount
        self.gamma = gamma
        # resolution
        self.res = res

        # number of states
        self.nX = np.ceil(((self.x_max - self.x_min) / self.res) + 1).astype('int')
        self.nY = np.ceil(((self.y_max - self.y_min) / self.res) + 1).astype('int')
        self.nTheta = np.ceil((self.theta_max - self.theta_min) / self.res).astype('int')

        # number of actions
        self.nV = np.ceil(((self.v_max - self.v_min) / self.res) + 1).astype('int')
        self.nW = np.ceil(((self.w_max - self.w_min) / self.res) + 1).astype('int')

        # noise cov
        sigma = np.array(sigma, dtype=np.float64)
        self.cov = np.diag(sigma ** 2)

        # State
        S_t = np.arange(0, self.nT)
        S_x = np.linspace(self.x_min, self.x_max, self.nX)
        S_y = np.linspace(self.y_min, self.y_max, self.nY)
        S_theta = np.linspace(self.theta_min, self.theta_max, self.nTheta)

        # Control
        V = np.linspace(self.v_min, self.v_max, self.nV)  # linear velocity
        W = np.linspace(self.w_min, self.w_max, self.nW)  # angular velocity

        # (nt, nx, ny, nθ)
        t, x, y, theta = np.meshgrid(S_t, S_x, S_y, S_theta, indexing='ij')
        self.E = np.stack((t, x, y, theta), axis=-1)
        # self.E = np.column_stack([t.flat, x.flat, y.flat, theta.flat])

        del t, x, y, theta

        # state space independent of time
        x, y, theta = np.meshgrid(S_x, S_y, S_theta, indexing='ij')
        self.pos = np.stack((x, y, theta), axis=-1)
        # self.pos = np.column_stack([x.flat, y.flat, theta.flat])
        del x, y, theta

        # (nv , nw)
        u, w = np.meshgrid(V, W, indexing='ij')
        self.U = np.stack((u, w), axis=-1)  # (11, 21, 2)
        # self.U = np.column_stack([u.flat, w.flat])

        self.L = np.empty(self.pos.shape[:-1] + self.U.shape[:-1])
        q = 1.5
        Q = np.array([
            [1, 0],
            [0, 1],
        ])
        R = np.array([
            [10, 0],
            [0, 1],
        ])
        for i in range(self.pos.shape[0]):
            for j in range(self.pos.shape[1]):
                for k in range(self.pos.shape[3]):
                    for v in range(self.U.shape[0]):
                        for w in range(self.U.shape[1]):
                            e = self.pos[i, j, k]
                            u_t = self.U[v, w]
                            p_t = e[:2]
                            theta_t = e[2]
                            pQp = p_t.T @ Q @ p_t
                            q_sq = q * (1 - np.cos(theta_t)) ** 2
                            uRu = u_t.T @ R @ u_t
                            cost = pQp + q_sq + uRu
                            assert pQp >= 0 and q_sq >= 0 and uRu >= 0
                            self.L[i, j, k, v, w] = cost

        ic(self.E.shape)
        ic(self.pos.shape)
        ic(self.U.shape)
        ic(self.L.shape)

    @staticmethod
    @lru_cache(maxsize=100)
    def wrap_angle(angles):
        while angles < -pi:
            angles += 2 * pi
        while angles >= np.pi:
            angles -= 2 * pi
        return angles

    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def lissajous(k: int, time_step=0.5):
        """
        This function returns the reference point at time step k
        :param k:
        :param time_step
        :return:
        """
        # time_step = self.dt

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

    def simple_controller(self, cur_state, ref_state):
        """This function implements a simple P controller"""
        k_v = 0.55
        k_w = 1.0
        v = k_v * np.sqrt((cur_state[0] - ref_state[0]) ** 2 + (cur_state[1] - ref_state[1]) ** 2)
        v = np.clip(v, self.v_min, self.v_max)
        angle_diff = ref_state[2] - cur_state[2]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        w = k_w * angle_diff
        w = np.clip(w, self.w_min, self.w_max)
        return np.array([v, w])

    def car_next_state(self, cur_state, control, noise=False):
        """
        The discrete-time kinematic model of the differential-drive robot
        :param cur_state: [x_t, y_t, theta_t].T
        :param control: [v_t, w_t]
        :param noise: Gaussian Motion Noise W_t
        :return: new_state
        """
        time_step = self.dt
        cur_state = cur_state.reshape(-1)
        theta = cur_state[2]
        # Yaw
        rot_3d_z = np.array([
            [np.cos(theta), 0],
            [np.sin(theta), 0],
            [0, 1]
        ], dtype=np.float64)
        f = rot_3d_z @ control
        w = 0
        if noise:
            # Gaussian Motion Noise (w_t ∈ R^{3}) with N(0, diag(σ)^2 ) where σ = [0.04, 0.04, 0.004] ∈ R^{3}
            # mean and standard deviation for (x,y)
            mu_pos, sigma_pos = 0, 0.04
            w_xy = np.random.normal(mu_pos, sigma_pos, 2)
            # mean and standard deviation for yaw angle (theta)
            mu_yaw, sigma_yaw = 0, 0.004
            w_theta = np.random.normal(mu_yaw, sigma_yaw, 1)
            w = np.concatenate((w_xy, w_theta))
        new_state = cur_state + time_step * f.flatten() + w
        return new_state

    def g(self, cur_err, cur_ref, nxt_ref, ctrl):
        """
        Error Dynamics
        :param cur_err:
        :param cur_ref:
        :param nxt_ref:
        :param ctrl: control [v_t, w_t]
        :return: error next step
        """
        tau = self.dt
        cur_err = cur_err.reshape(3, -1)
        theta_err = cur_err[2, 0]
        ref_diff = (cur_ref - nxt_ref).reshape(3, -1)

        a_t = cur_ref[2]
        u = ctrl.reshape(2, -1)

        G = np.array([
            [tau * np.cos(theta_err + a_t), 0],
            [tau * np.sin(theta_err + a_t), 0],
            [0, tau],
        ], dtype=np.float64)
        nxt_err = cur_err + G @ u + ref_diff
        return nxt_err.reshape(-1)

    def pf(self, cur_E, cur_U, T: int):
        """
        Motion Model for error state
        :param cur_E: state (3,)
        :param cur_U: control (2,)
        :param T:
        :return:
        """
        # current reference state (3,)
        cur_ref = self.traj[T]
        # next reference state (3,)
        nxt_ref = self.traj[T + 1]

        # Obtain next error state e' from error dynamic
        nxt_err_mean = self.g(cur_E, cur_ref, nxt_ref, ctrl=cur_U)  # (3,)

        # Enforce state in bound and wrap angle [-pi, pi)
        nxt_err_mean[0] = np.clip(nxt_err_mean[0], self.x_min, self.x_max)
        nxt_err_mean[1] = np.clip(nxt_err_mean[1], self.y_min, self.y_max)
        nxt_err_mean[2] = self.wrap_angle(nxt_err_mean[2])

        # scipy implementation of Multivariate Normal
        # self.pos: (nx, ny, ntheta) should be the same across time
        log_prob = multivariate_normal.logpdf(self.pos, mean=nxt_err_mean, cov=self.cov)

        # own implementation of Fast and Numerically Stable Multivariate Normal
        # Compute in log space for numerical stability
        # log_prob = gaussian.logpdf(self.pos, mean=nxt_err_mean)
        # Normalize to ensure outgoing transition probability sum to 1
        P_nxt_e = gaussian.softmax(log_prob)
        assert abs(np.sum(P_nxt_e) - 1) <= 1e-4
        return P_nxt_e

    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def calc_Q(L, gamma, P, V_pi):
        return L + gamma * np.sum(P * V_pi)


    def policy_iteration(self, num_iter=100):
        V0 = np.zeros(self.E.shape[:-1])
        PI = np.zeros(self.E.shape[:-1] + (2,), dtype='int')
        Q0 = np.zeros(self.E.shape[:-1] + self.U.shape[:-1])
        ic(V0.shape)
        ic(PI.shape)
        ic(Q0.shape)

        nT, nX, nY, nThe, nV, nW = Q0.shape
        for _ in tqdm(range(num_iter)):
            V_prev = V0.copy()
            for _ in tqdm(range(5)):
                for t in range(nT):
                    for x in range(nX):
                        for y in range(nY):
                            for theta in range(nThe):
                                cur_e = self.E[t, x, y, theta][1:]
                                v, w = PI[t, x, y, theta, :]
                                cur_u = self.U[v, w]
                                cost_eu = self.L[x, y, theta, v, w]
                                P_e_prim = self.pf(cur_e, cur_u, t)
                                V0[t, x, y, theta] = self.calc_Q(cost_eu, self.gamma, P_e_prim, V0[t, :, :, :])
            for t in tqdm(range(nT)):
                for x in range(nX):
                    for y in range(nY):
                        for theta in range(nThe):
                            cur_e = self.E[t, x, y, theta][1:]
                            for v in range(nV):
                                for w in range(nW):
                                    cur_u = self.U[v, w]
                                    cost_eu = self.L[x, y, theta, v, w]
                                    P_e_prim = self.pf(cur_e, cur_u, t)
                                    Q0[t, x, y, theta, v, w] = self.calc_Q(cost_eu, self.gamma, P_e_prim, V0[t, :, :, :])
                            idx = np.argmin(Q0[t, x, y, theta])
                            r, c = np.unravel_index(idx, (nV, nW))
                            PI[t, x, y, theta] = [int(r), int(c)]
                            V0[t, x, y, theta] = np.min(Q0[t, x, y, theta])
            ic(np.sum((V0 - V_prev) ** 2))
            if np.allclose(V0, V_prev, equal_nan=True):
                break

        return V0, PI


if __name__ == '__main__':
    mdp = MDP(dt=0.5, res=0.5)
    x_init = 1.5
    y_init = 0.0
    theta_init = np.pi / 2
    obstacles = np.array([
        [-2, -2, 0.5],
        [1, 2, 0.5]
    ])

    # V, policy = mdp.policy_iteration(num_iter=100, V_k='V.npy', pi_k='policy.npy')
    # with open('V.npy', 'wb') as f:
    #     np.save(f, V)
    # with open('policy.npy', 'wb') as f:
    #     np.save(f, policy)

    print('<--------- Loading V & PI ----------->')
    with open('V.npy', 'rb') as f1:
        V0 = np.load(f1)
    with open('policy.npy', 'rb') as f2:
        PI = np.load(f2)

    S_x = np.linspace(mdp.x_min, mdp.x_max, mdp.nX)
    S_y = np.linspace(mdp.y_min, mdp.y_max, mdp.nY)
    S_theta = np.linspace(mdp.theta_min, mdp.theta_max, mdp.nTheta)

    V = np.linspace(mdp.v_min, mdp.v_max, mdp.nV)  # linear velocity
    W = np.linspace(mdp.w_min, mdp.w_max, mdp.nW)  # angular velocity

    cur_state = np.array([x_init, y_init, theta_init], dtype=np.float64)

    car_states, ref_traj, times = [], [], []
    error_lst = []
    error = 0.0

    for t in range(mdp.nT):
        t1 = time.time()
        cur_ref = mdp.lissajous(t)
        cur_err = cur_state - cur_ref
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        x = np.argmin(abs(cur_err[0] - S_x))
        y = np.argmin(abs(cur_err[1] - S_y))
        theta = np.argmin(abs(cur_err[2] - S_theta))
        ic(x, y, theta)

        ic(PI[t, x, y, theta].squeeze())
        v_idx, w_idx = PI[t, x, y, theta].squeeze()
        v, w = mdp.U[v_idx, w_idx]
        u = np.array([v, w])
        next_state = mdp.car_next_state(cur_state, u, noise=True)
        cur_state = next_state
        time_itr = time.time() - t1
        print(f"\n<----------{t}---------->")
        print(f"time: {time_itr: .3f}")
        times.append(time_itr)
        err = np.linalg.norm((cur_state - cur_ref), ord=2)
        error_lst.append(err)
        ic(err)
        error += err
    print(f'Final error: {error}')
    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)

    error_lst = np.array(error_lst)
    plt.scatter(range(len(error_lst)), error_lst, label='error')
    plt.legend()
    plt.show()

    plt.scatter(car_states[:, 0], car_states[:, 1], label='car')
    plt.scatter(ref_traj[:, 0], ref_traj[:, 1], label='reference')
    plt.legend()
    plt.show()

    try:
        utils.visualize(car_states, ref_traj, obstacles, times, 0.5, save=True)
    except KeyboardInterrupt:
        plt.close('all')
        sys.exit(0)