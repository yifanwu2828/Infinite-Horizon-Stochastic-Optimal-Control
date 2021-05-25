import argparse
import random
from time import time
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from icecream import ic

import utils
import cec


@lru_cache(maxsize=2048)
def lissajous(k):
    """
    This function returns the reference point at time step k
    :param k: time step
    :return:
    """
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2 * np.pi / 50
    b = 3 * a
    a_t = a*time_step
    T = np.round(2 * np.pi / a_t)

    k = k % T
    delta = np.pi / 2
    a_k_t = k * a_t
    b_k_t = b*k*time_step

    xref = xref_start + A * np.sin(a_k_t + delta)
    yref = yref_start + B * np.sin(b_k_t)
    v = [A*a*np.cos(a_k_t + delta), B*b*np.cos(b_k_t)]
    thetaref = np.arctan2(v[1], v[0])
    return np.array([xref, yref, thetaref])


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

    w = None
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
    return new_state, w


def predict_T_step(cur_state, traj, cur_iter, T=10, time_step=0.5):
    """

    :param cur_state: current car state
    :param traj: reference trajectory
    :param cur_iter: idx to track traj
    :param T: number of steps
    :param time_step: 0.5
    :return:
    """
    state_seq = []
    ref_seq = []
    act_seq = []
    error_seq = []

    state = cur_state
    for i in range(T):
        state_seq.append(state)

        # reference
        cur_ref = traj(cur_iter + i)
        ref_nxt_state = traj(cur_iter + i+1)
        ref_seq.append(cur_ref)

        # control
        u = utils.simple_controller(state, cur_ref)
        act_seq.append(u)

        # error
        err = utils.error_dynamics(state, cur_ref, ref_nxt_state, u, Wt=None)
        error_seq.append(err)

        # car next state
        nxt_state, _ = car_next_state(time_step, state, u, noise=False)

    state_seq = np.array(state_seq).reshape(3, -1)
    ref_seq = np.array(ref_seq).reshape(3, -1)
    act_seq = np.array(act_seq).reshape(2, -1)
    error_seq = np.array(error_seq).reshape(3, -1)
    return state_seq, ref_seq, act_seq, error_seq


if __name__ == '__main__':
    if True:
        p = argparse.ArgumentParser()
        p.add_argument("--sim_time", help="simulation time(not change)", type=float, default=120)
        p.add_argument("--time_step", help="time between steps in seconds", type=float, default=0.5)
        p.add_argument("-r", "--render", help="Visualize Env", action="store_true", default=True)
        p.add_argument("--save", help="Save Trajectory", action="store_true", default=False)
        p.add_argument("--seed", help="Random Generator Seed", type=int, default=42)
        p.add_argument(
            "-verb", "--verbose", action="store_true", default=True,
            help="Verbose mode (False: no output, True: INFO)"
        )
        args = p.parse_args()
    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Simulation params
    time_step = args.time_step

    # Car params
    x_init = 1.5
    y_init = 0.0
    theta_init = np.pi / 2

    # Control limits
    v_min, v_max = (0, 1)
    w_min, w_max = (-1, 1)

    # Obstacles in the environment [x, y, rad]
    obstacles = np.array([
        [-2, -2, 0.5],
        [1,   2, 0.5]
    ])

    # Params
    traj = lissajous
    ref_traj = []
    car_states = []
    times = []
    error = 0.0

    # -------------------- Start main loop------------------------------------------
    # Initialize state
    start_main_loop = time()
    cur_state = np.array([x_init, y_init, theta_init], dtype=np.float64)

    cur_iter = 0
    # Main loop
    while cur_iter * time_step < args.sim_time:
        t1 = time()

        # Get reference state
        cur_time_step = cur_iter * time_step
        cur_ref = traj(cur_iter)

        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller by your own controller
        # control = simple_controller(cur_state, cur_ref)

        # TODO: predict T=10 step
        states, ref_states, control_seq, err_seq = predict_T_step(cur_state, traj, cur_iter, T=10)
        control = cec.CEC(states, ref_states, control_seq, err_seq, obstacles)
        print(f"[v,w]: {control}")
        ################################################################

        # Apply control input
        next_state, Wt = car_next_state(time_step, cur_state, control, noise=True)

        # Update current state
        cur_state = next_state
        # Loop time
        t2 = time()
        if args.verbose:
            print(f"\n<----------{cur_iter}---------->")
            print(f"time: {t2 - t1: .3f}")
        times.append(t2 - t1)
        cur_error = np.linalg.norm(cur_state - cur_ref)
        error += cur_error
        cur_iter += 1
    end_mainloop_time = time()
    print('\n\n')
    print(f'Total duration: {end_mainloop_time - start_main_loop}')
    print(f'Average iteration time: {np.array(times).mean() * 1e3} ms')
    print(f'Final error: {error}')

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    if args.render:
        try:
            utils.visualize(car_states, ref_traj, obstacles, times, time_step, save=args.save)
        except KeyboardInterrupt:
            plt.close('all')
            pass




