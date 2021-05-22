import argparse
import random
from time import time
from functools import lru_cache

import numpy as np

import utils


# Simulation params
time_step = 0.5  # time between steps in seconds
sim_time = 120   # simulation time

# Car params
x_init = 1.5
y_init = 0.0
theta_init = np.pi / 2

v_max = 1
v_min = 0
w_max = 1
w_min = -1


@lru_cache(maxsize=256)
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
    return [xref, yref, thetaref]


def simple_controller(cur_state, ref_state):
    """This function implements a simple P controller"""
    k_v = 0.55
    k_w = 1.0
    v = k_v*np.sqrt((cur_state[0] - ref_state[0])**2 + (cur_state[1] - ref_state[1])**2)
    v = np.clip(v, v_min, v_max)
    angle_diff = ref_state[2] - cur_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    w = k_w*angle_diff
    w = np.clip(w, w_min, w_max)
    return [v, w]


# This function implement the car dynamics
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--render", help="Visualize Env", action="store_true", default=True)
    parser.add_argument("--save", help="Save Trajectory", action="store_true", default=False)
    parser.add_argument("--seed", help="Random Generator Seed", type=int, default=42)
    parser.add_argument(
        "-verb", "--verbose", action="store_true", default=True,
        help="Verbose mode (False: no output, True: INFO)"
    )
    args = parser.parse_args()

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Obstacles in the environment
    obstacles = np.array([[-2, -2, 0.5], [1, 2, 0.5]])

    # Params
    traj = lissajous
    ref_traj = []
    error = 0.0
    car_states = []
    times = []

    # Start main loop
    start_main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([x_init, y_init, theta_init], dtype=np.float64)
    cur_iter = 0
    # Main loop
    while cur_iter * time_step < sim_time:
        t1 = time()
        # Get reference state
        cur_time = cur_iter * time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller by your own controller
        control = simple_controller(cur_state, cur_ref)
        print(f"[v,w]: {control}")
        ################################################################

        # Apply control input
        next_state = car_next_state(time_step, cur_state, control, noise=True)
        # Update current state
        cur_state = next_state
        # Loop time
        t2 = time()
        if args.verbose:
            print(f"<----------{cur_iter}---------->")
            print(f"time: {t2 - t1}")
        times.append(t2-t1)
        error += np.linalg.norm(cur_state - cur_ref)
        cur_iter = cur_iter + 1
    end_mainloop_time = time()
    print('\n\n')
    print(f'Total time: {end_mainloop_time - start_main_loop}')
    print(f'Average iteration time: {np.array(times).mean() * 1000} ms')
    print(f'Final error: {error}')

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    if args.render:
        try:
            utils.visualize(car_states, ref_traj, obstacles, times, time_step, save=args.save)
        except KeyboardInterrupt:
            pass
