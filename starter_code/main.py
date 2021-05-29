import argparse
import random
from time import time


import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

import utils
import cec


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--sim_time", help="simulation time(not change)", type=float, default=120)
    p.add_argument("--time_step", help="time between steps in seconds", type=float, default=0.5)
    p.add_argument("-T", "--T", help="number of steps to collect", type=int, default=10)
    p.add_argument("-r", "--render", help="Visualize Env", action="store_true", default=False)
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
    # time_step = args.time_step
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
    traj = utils.lissajous
    car_states, ref_traj, times = [], [], []
    error_lst = []
    error = 0.0

    # -------------------- Start main loop------------------------------------------
    # Initialize state
    t0 = utils.tic()
    cur_state = np.array([x_init, y_init, theta_init], dtype=np.float64)

    cur_iter = 0
    # Main loop
    while cur_iter * time_step < args.sim_time:
        t1 = time()

        # Get reference state
        # cur_time_step = cur_iter * time_step
        cur_ref = traj(k=cur_iter, time_step=time_step)

        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)
        ################################################################
        # Generate control input
        states, ref_states, control_seq, cur_err_seq, nxt_err_seq = utils.predict_T_step(
            cur_state,
            traj, cur_iter,
            T=args.T,
            time_step=time_step
        )

        control = cec.CEC(states, ref_states, obstacles, time_step, control_seq, nxt_err_seq)
        if control is None:
            control = utils.simple_controller(cur_state, cur_ref)
        print(f"[v,w]: {control}")

        ################################################################

        # Apply control input
        next_state = utils.car_next_state(time_step, cur_state, control, noise=True)

        # Update current state
        cur_state = next_state

        # Loop time
        time_itr = time() - t1
        if args.verbose:
            print(f"\n<----------{cur_iter}---------->")
            print(f"time: {time_itr: .3f}")
        times.append(time_itr)
        cur_err = np.linalg.norm((cur_state - cur_ref), ord=2)
        error_lst.append(cur_err)
        ic(cur_err)
        error += cur_err
        cur_iter += 1
    utils.toc(t0, name='CEC')
    print(f'Average iteration time: {np.array(times).mean() * 1e3:.3f} ms')
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

    if args.render:
        try:
            utils.visualize(car_states, ref_traj, obstacles, times, time_step, save=args.save)
        except KeyboardInterrupt:
            plt.close('all')
            pass




