import argparse
import h5py
import multiprocessing as mp
import numpy as np
import os
import sys
import tensorflow as tf
import time
import julia
import sys
import scipy.io as io


backend = 'TkAgg'
import matplotlib

matplotlib.use(backend)
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')
from contexttimer import Timer

import hgail.misc.utils
import algorithms.utils

from envs import hyperparams, utils, build_env

from envs.utils import str2bool
from utils.math_utils import classify_traj, safetyindex
from utils.safeset import SafeSet
from algorithms.AGen import rls, validate_utils
from preprocessing.clean_holo import clean_data, csv2txt, create_lane
from preprocessing.extract_feature import extract_ngsim_features
from src.trajdata import convert_raw_ngsim_to_trajdatas
from numpy import cos, sin
# import pdb
import math
import tqdm
import torch

plt.style.use("ggplot")

N_ITERATION = 1  # the number of iterations of rls step
N_VEH = 1  # the number of controlled vehicles for each simulation


def online_adaption(
        env,
        policy,
        obs,
        mean,
        env_kwargs=dict(),
        lbd=0.99,
        adapt_steps=1,
        trajinfos=None,
        control=None):

    if len(obs.shape) == 2:
        obs = np.expand_dims(obs, axis=0)
        mean = np.expand_dims(mean, axis=0)
    assert trajinfos is not None


    # print("new theta: {}".format(new_theta))

    ego_start_frame = trajinfos[env_kwargs['egoid']]['ts']
    maxstep = trajinfos[env_kwargs['egoid']]['te'] - trajinfos[env_kwargs['egoid']]['ts'] - 52
    env_kwargs['start'] = ego_start_frame + 2
    x = env.reset(**env_kwargs)

    n_agents = x.shape[0]
    # print("Agent number: {}".format(n_agents))
    dones = [True] * n_agents
    policy.reset(dones)

    print("max steps")
    print(maxstep)

    lx = x
    phi_average = []
    for step in tqdm.tqdm(range(ego_start_frame, maxstep + ego_start_frame - 1)):
    # for step in tqdm.tqdm(range(ego_start_frame, ego_start_frame+1)):  #test one step
        # print("==================== start rollout now ====================")
        # phi_overall = rollout(x, adapnets, env, policy, prev_hiddens, n_agents, adapt_steps, control, maxstep)
        # skip until reach segment 3
        if env.check_segment() is not 3:
            maxstep = maxstep - 1
            env_kwargs['start'] += 1
            lx = x
            x = env.reset(**env_kwargs) #update the frame information 
            continue
        phi_overall = rollout(x, env, policy, n_agents, control, maxstep)
        phi_average.append(np.mean(phi_overall))
        env_kwargs['start'] += 1
        lx = x
        x = env.reset(**env_kwargs) #update the frame information 
        print("already started from one groundtruth")
        break
        
    print("the average phi is", phi_average)




    error_info = dict()
    return error_info


def rollout(x, env, policy, n_agents, control, maxstep):
    # use_ssa = True
    use_ssa = True
    # predict_span = maxstep
    predict_span = 100
    # predict_span = 1
    # predict_span = 30
    # predict_span = 2
    error_per_step = []  # size is (predict_span, n_agent) each element is a dict(dx: , dy: ,dist: )
    valid_data = True
    hi_speed_limit = 40
    lo_speed_limit = 10
    orig_trajectory = []
    pred_trajectory = []
    start_time = time.time()
    time_info = {}
    phi_overall = []
    front_cnt = 0
    orig_acc = []
    nossa_acc = []
    ssa_acc = []
    for j in range(predict_span):
        x[0][15] = 0
        a, a_info, hidden_vec = policy.get_actions(x)
        actions = a # use GAIL policy


        # use safety monitor to adjust to control input 
        # define the fx, fu, dot_Xh, Mr, Mh, p_Mr_p_Xr, p_Mh_p_Xh, u0
        # get current ego vehicle states 
        # print("============ load Xr state =============")
        posx, posy, postheta, posv, idnum = env.veh_state()
        # print("ego posx, posy, postheta, posv :=", posx, posy, postheta, posv)
        fx = np.array([[posv*cos(postheta)],[posv*sin(postheta)],[0],[0]])
        fu = np.array([[0, 0],
              [0, 0],
              [0, 1],
              [1, 0]])
        Mr = np.array([[posx],[posy],[posv*cos(postheta)],[posv*sin(postheta)]])

        p_Mr_p_Xr = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, -posv*sin(postheta), posv*cos(postheta)],
                            [0, 0, cos(postheta), sin(postheta)]])
        

        # print("============ load Xh state =============")
        # get the next vehicle states
        poshx, poshy, poshtheta, poshv = env.front_state()
        # print("front poshx, poshy, poshtheta, poshv :=", poshx, poshy, poshtheta, poshv)
        Mh = np.array([[poshx],[poshy],[poshv*cos(poshtheta)],[poshv*sin(poshtheta)]])
        p_Mh_p_Xh = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, -poshv*sin(poshtheta), poshv*cos(poshtheta)],
                     [0, 0, cos(poshtheta), sin(poshtheta)]])
        dot_Xh = np.array([[poshv*cos(poshtheta)],[poshv*sin(poshtheta)],[0],[0]])

        # print("============ SSA control =============")
        if poshx == 0. and poshy == 0. and poshtheta == 0. and poshv == 0.: # there is no front car
            # print("no front car, no SSA needed")
            pass
        else:
            # convert ndarray to np.matrix
            front_cnt = front_cnt + 1
            fx = np.asmatrix(fx)
            fu = np.asmatrix(fu)
            dot_Xh = np.asmatrix(dot_Xh)
            Mr = np.asmatrix(Mr)
            Mh = np.asmatrix(Mh)
            p_Mr_p_Xr = np.asmatrix(p_Mr_p_Xr)
            p_Mh_p_Xh = np.asmatrix(p_Mh_p_Xh)
            a = np.reshape(a,(2,1)) #reshape the action
            a = np.asmatrix(a)
            # print("detected foreforevelocity is {} and the acceleartion control is {}".format(x[0][64], a[0][0])) 
            assert poshv > 0 # car is moving back 
            print("the acceleartion control is {}".format(a[0][0]))  
            '''uncomment below to activate SSA'''    
            if use_ssa:      
                a_new = control.calc_control_input(fx, fu, dot_Xh, Mr, Mh, p_Mr_p_Xr, p_Mh_p_Xh, a)
                a_new = np.reshape(a_new,(1,2)) #shape back the action
                actions = np.asarray(a_new)
                print("the new acceleartion is {}".format(actions[0][0]))
            phi = control.eval_phi(fx, fu, dot_Xh, Mr, Mh, p_Mr_p_Xr, p_Mh_p_Xh, a)
            phi_overall.append(phi)

        nx, r, dones, e_info = env.step(actions)
        # record the current action acceleration and original acceleration 
        orig_acc.append(env.get_ori_acc())
        if use_ssa is True:
            ssa_acc.append(actions[0][0])
        else:
            nossa_acc.append(actions[0][0])
        # phi_overall.append(safetyindex(nx)) # recording safety index 
        print("----------------------- step {} --------------------".format(j))
        if any(dones):
            print("=== done at step {}".format(j))
            break
        x = nx
    # print("total step: {}, total front car {}".format(predict_span, front_cnt))
    # save to mat 
    trial = 3
    path1 = "result/ssa/veh{}_ssa_seg3_trial{}.mat".format(idnum, trial)
    path2 = "result/nossa/veh{}_nossa_seg3_trial{}.mat".format(idnum, trial)
    pathacc1 = "result/ssa_acc/veh{}_ssa_seg3_trial{}_acc.mat".format(idnum, trial)
    pathacc2 = "result/nossa_acc/veh{}_ssa_seg3_trial{}_acc.mat".format(idnum, trial)
    if use_ssa is True:
        io.savemat(path1, {"phi_data":np.asarray(phi_overall)})
        # io.savemat(pathacc1, {"orig_acc":np.asarray(orig_acc), "ssa_acc":np.asarray(ssa_acc)})
        print("look the ssa acc:", ssa_acc)
    else:
        io.savemat(path2, {"phi_data":np.asarray(phi_overall)})
        # io.savemat(pathacc2, {"orig_acc":np.asarray(orig_acc), "nossa_acc":np.asarray(nossa_acc)})
        print("look the nossa acc:", nossa_acc)

    safe_info = dict()
    safe_info["phi_overall"] = phi_overall
    utils.print_safety(safe_info)
    f = plt.figure(1)
    plt.plot(phi_overall)
    plt.ylabel('Safety index value')
    plt.xlabel('Frame number')
    f.show()

    g = plt.figure(2)
    if use_ssa is True:
        plt.plot(ssa_acc)
        plt.plot(orig_acc)
    else:
        plt.plot(nossa_acc)
        plt.plot(orig_acc)
    plt.ylabel('original acceleration')
    plt.xlabel('Frame number')
    g.show()

    input()
    # raw_input()
    return phi_overall




def prediction(x, adapnets, env, policy, prev_hiddens, n_agents, adapt_steps):
    predict_span = 50
    error_per_step = []  # size is (predict_span, n_agent) each element is a dict(dx: , dy: ,dist: )
    valid_data = True
    hi_speed_limit = 40
    lo_speed_limit = 10
    orig_trajectory = []
    pred_trajectory = []
    start_time = time.time()
    time_info = {}
    phi_overall = []
    for j in range(predict_span):
        # if j == 0:
        #     print("feature {}".format(j), x)
        x[0][15] = 0
        a, a_info, hidden_vec = policy.get_actions(x)
        feature_array = np.concatenate([feature_array, np.array(x)], axis=0)
        hidden_vec = np.random.randn(1, 64)
        if adapt_steps == 1:
            adap_vec = hidden_vec
        else:
            adap_vec = np.concatenate((hidden_vec, prev_hiddens, x), axis=1)

        means = np.zeros([n_agents, 2])
        log_std = np.zeros([n_agents, 2])
        for i in range(x.shape[0]):
            means[i] = adapnets[i].predict(np.expand_dims(adap_vec[i], 0))
            log_std[i] = np.log(np.std(adapnets[i].theta, axis=0))

        prev_hiddens = hidden_vec
        actions = means
        # print("random feature:", actions)
        # print("policy feature:", a)
        # print("predict step: {}".format(j+1))
        nx, r, dones, e_info = env.step(actions)


        error_per_agent = []  # length is n_agent, each element is a dict(dx: , dy: ,dist: )
        for i in range(n_agents):
            assert n_agents == 1
            # print("orig x: ", e_info["orig_x"][i])
            # print("orig y: ", e_info["orig_y"][i])
            # print("orig v: ", e_info["orig_v"][i])
            # print("predicted v:", e_info["v"][i])
            # print("orig theta: ", e_info["orig_theta"][i])
            # print("predicted x: ", e_info["x"][i])
            # print("predicted y: ", e_info["y"][i])
            dx = abs(e_info["orig_x"][i] - e_info["x"][i])
            dy = abs(e_info["orig_y"][i] - e_info["y"][i])
            dist = math.hypot(dx, dy)
            # print("dist: ", dist)
            if e_info["orig_v"][i] > hi_speed_limit or e_info["orig_v"][i] < lo_speed_limit:
                valid_data = False
            # print("{}-----> dx: {} dy: {} dist: {}".format(j, dx, dy, dist))
            if valid_data:
                if dist > 140:
                    exit(0)   # this is for debugging
                error_per_agent.append(dist)
                orig_trajectory.append([e_info["orig_x"][i], e_info["orig_y"][i]])
                pred_trajectory.append([e_info["x"][i], e_info["y"][i]])
        if valid_data:
            error_per_step += error_per_agent
        if any(dones):
            break
        x = nx
        end_time = time.time()
        if j == 19:
            time_info["20"] = end_time - start_time
        elif j == 49:
            time_info["50"] = end_time - start_time

    return error_per_step, time_info, orig_trajectory, pred_trajectory


def collect_trajectories(
        args,
        params,
        egoids,
        error_dict,
        pid,
        env_fn,
        policy_fn,
        use_hgail,
        random_seed,
        lbd,
        adapt_steps):
    print("=================== run in collect trajecotries ===============")
    print('env initialization args')
    print(args)
    env, trajinfos, _, _ = env_fn(args, n_veh=N_VEH, alpha=0.)
    # here trajinfos are reset to the already running cars
    # print(trajinfos[0])
    args.policy_recurrent = True
    policy = policy_fn(args, env, mode=1)
    if torch.cuda.is_available():
        policy = policy.cuda()
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())

        # then load parameters
        if use_hgail:
            for i, level in enumerate(policy):
                level.algo.policy.set_param_values(params[i]['policy'])
            policy = policy[0].algo.policy
        else:
            policy_param_path = "./data/experiments/NGSIM-gail/imitate/model/policy.pkl"
            policy.load_param(policy_param_path)
            print("load policy param from: {}".format(policy_param_path))
            # policy.set_param_values(params['policy'])

        normalized_env = hgail.misc.utils.extract_normalizing_env(env)
        if normalized_env is not None:
            normalized_env._obs_mean = params['normalzing']['obs_mean']
            normalized_env._obs_var = params['normalzing']['obs_var']

        # collect trajectories
        egoids = np.unique(egoids)
        nids = len(egoids)
        veh_2_index = {}
        if args.env_multiagent:
            data, index = validate_utils.get_multiagent_ground_truth(args.ngsim_filename, args.h5_filename)
            for i, idx in enumerate(index):
                veh_2_index[idx] = i
        else:
            data = validate_utils.get_ground_truth(args.ngsim_filename, args.h5_filename)
            sample = np.random.choice(data['observations'].shape[0], 2)

        kwargs = dict()

        # safe set controller
        SSA = SafeSet()
        # print(('Loading obs data Running time: %s Seconds' % (end_time - start_time)))
        if args.env_multiagent:
            # I add not because single simulation has no orig_x etc.
            # egoid = random.choice(egoids)
            trajinfos = trajinfos[0]
            error = {"overall": [],
                     "curve": [],
                     "lane_change": [],
                     "straight": [],
                     "time_info": [],
                     "orig_traj": [],
                     "pred_traj": []}
            print("========= start loading vehicle ids ==========")
            # for veh_id in trajinfos.keys():
            for veh_id in [13, 8]:
            # just test the vehicle 8
                if trajinfos[veh_id]["te"] - trajinfos[veh_id]["ts"] <= 52:
                    continue
                # no use of random_seed
                if random_seed:
                    kwargs = dict(random_seed=random_seed + veh_id)
                print("egoid: {}, ts: {}, te: {}".format(veh_id, trajinfos[veh_id]["ts"], trajinfos[veh_id]["te"]))
                print("data index is {}".format(veh_2_index[veh_id]))
                kwargs['egoid'] = veh_id
                kwargs['traj_idx'] = 0

                error_info = online_adaption(
                    env,
                    policy,
                    obs=data['observations'][[veh_2_index[veh_id]], :, :],
                    mean=data['actions'][[veh_2_index[veh_id]], :, :],
                    env_kwargs=kwargs,
                    lbd=lbd,
                    adapt_steps=adapt_steps,
                    trajinfos=trajinfos,
                    control = SSA
                )
                print("tested on one egoveh, now quit")
                sys.exit("quit")

                # error["overall"] += error_info["overall"]
                # error["curve"] += error_info["curve"]
                # error["lane_change"] += error_info["lane_change"]
                # error["straight"] += error_info["straight"]
                # error["time_info"] += error_info["time_info"]
                # error["orig_traj"] += error_info["orig_traj"]
                # error["pred_traj"] += error_info["pred_traj"]
            error_dict.append(error)
        else:
            # for i in sample:
            for i, egoid in enumerate(egoids):
                sys.stdout.write('\rpid: {} traj: {} / {}\n'.format(pid, i, nids))
                index = veh_2_index[egoid]
                traj = online_adaption(
                    env,
                    policy,
                    obs=data['observations'][index, :, :],
                    mean=data['actions'][index, :, :],
                    env_kwargs=dict(egoid=egoid, traj_idx=[0]),
                    lbd=lbd,
                    adapt_steps=adapt_steps,
                )

    return error_dict


def parallel_collect_trajectories(
        args,
        params,
        egoids,
        n_proc,
        env_fn=build_env.build_ngsim_env,
        use_hgail=False,
        random_seed=None,
        lbd=0.99,
        adapt_steps=1):
    # build manager and dictionary mapping ego ids to list of trajectories

    tf_policy = False
    parallel = False
    # set policy function
    policy_fn = validate_utils.build_policy if tf_policy else algorithms.utils.build_policy

    # partition egoids
    proc_egoids = utils.partition_list(egoids, n_proc)
    if parallel:
        manager = mp.Manager()
        error_dict = manager.list()
        # pool of processes, each with a set of ego ids
        pool = mp.Pool(processes=n_proc)
        # print(('Creating parallel env Running time: %s Seconds' % (end_time - start_time)))
        # run collection
        print("=========== run parallel collection ==========")
        print("the n_proc:", n_proc)
        results = []
        for pid in range(n_proc):
            res = pool.apply_async(
                collect_trajectories,
                args=(
                    args,
                    params,
                    proc_egoids[pid],
                    error_dict,
                    pid,
                    env_fn,
                    policy_fn,
                    use_hgail,
                    random_seed,
                    lbd,
                    adapt_steps
                )
            )
            results.append(res)
        [res.get() for res in results]
        pool.close()
    else:
        print("=========== run single process collection ==========")
        error_dict = []
        error_dict = collect_trajectories(
            args,
            params,
            proc_egoids[0],
            error_dict,
            0,
            env_fn,
            policy_fn,
            use_hgail,
            random_seed,
            lbd,
            adapt_steps
        )

    # wait for the processes to finish

    # let the julia processes finish up
    time.sleep(10)
    return error_dict[0]


def single_process_collect_trajectories(
        args,
        params,
        egoids,
        starts,
        n_proc,
        env_fn=build_env.build_ngsim_env,
        max_steps=200,
        use_hgail=False,
        random_seed=None):
    '''
    This function for debugging purposes
    '''
    # build list to be appended to
    trajlist = []

    # set policy function
    policy_fn = build_env.build_hierarchy if use_hgail else validate_utils.build_policy
    tf.reset_default_graph()

    # collect trajectories in a single process
    collect_trajectories(
        args,
        params,
        egoids,
        starts,
        trajlist,
        n_proc,
        env_fn,
        policy_fn,
        max_steps,
        use_hgail,
        random_seed
    )
    return trajlist


def collect(
        egoids,
        args,
        exp_dir,
        use_hgail,
        params_filename,
        n_proc,
        collect_fn=parallel_collect_trajectories,
        random_seed=None,
        lbd = 0.99,
        adapt_steps=1):
    '''
    Description:
        - prepare for running collection in parallel
        - multiagent note: egoids and starts are not currently used when running
            this with args.env_multiagent == True
    '''
    # load information relevant to the experiment
    params_filepath = os.path.join(exp_dir, 'imitate/{}'.format(params_filename))
    params = np.load(params_filepath)['params'].item()
    # validation setup
    validation_dir = os.path.join(exp_dir, 'imitate', 'test')
    utils.maybe_mkdir(validation_dir)

    with Timer():
        error = collect_fn(
            args,
            params,
            egoids,
            n_proc,
            use_hgail=use_hgail,
            random_seed=random_seed,
            lbd=lbd,
            adapt_steps=adapt_steps
        )

    return error

    # utils.write_trajectories(output_filepath, trajs)


def load_egoids(filename, args, n_runs_per_ego_id=10, env_fn=build_env.build_ngsim_env):
    offset = args.env_H + args.env_primesteps
    basedir = os.path.expanduser('~/Autoenv/data/')  # TODO: change the file path
    ids_filename = filename.replace('.txt', '-index-{}-ids.h5'.format(offset))
    print("ids_filename")
    print(ids_filename)
    ids_filepath = os.path.join(basedir, ids_filename)
    print("Creating ids file")
    # this should create the ids file
    env_fn(args)
    if not os.path.exists(ids_filepath):
        raise ValueError('file unable to be created, check args')
    ids = np.array(h5py.File(ids_filepath, 'r')['ids'].value)
    print("Creating starts file")
    ids_file = h5py.File(ids_filepath, 'r')
    ts = ids_file['ts'].value
    te = ids_file['te'].value
    length = np.array([e - s for (s, e) in zip(ts, te)])
    traj_num = length.sum()

    ids = np.tile(ids, n_runs_per_ego_id)
    return ids, traj_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validation settings')
    parser.add_argument('--n_proc', type=int, default=1)
    parser.add_argument('--exp_dir', type=str, default='./data/experiments/NGSIM-gail')
    parser.add_argument('--params_filename', type=str, default='itr_200.npz')
    parser.add_argument('--n_runs_per_ego_id', type=int, default=1)
    parser.add_argument('--use_hgail', type=str2bool, default=False)
    parser.add_argument('--use_multiagent', type=str2bool, default=False)
    parser.add_argument('--n_multiagent_trajs', type=int, default=10000)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--n_envs', type=int, default=None)
    parser.add_argument('--remove_ngsim_vehicles', type=str2bool, default=False)
    parser.add_argument('--lbd', type=float, default=0.99)
    parser.add_argument('--adapt_steps', type=int, default=1)

    run_args = parser.parse_args()
    j = julia.Julia()
    j.using("NGSIM")

    args_filepath = "./args/params.npz"
    if os.path.isfile(args_filepath):
        args = hyperparams.load_args(args_filepath)
    else:
        raise ValueError("No such params file")  # if no such file, please run save_args.py

    if run_args.use_multiagent:
        args.env_multiagent = True
        args.remove_ngsim_vehicles = run_args.remove_ngsim_vehicles

    if run_args.debug:
        collect_fn = single_process_collect_trajectories
    else:
        collect_fn = parallel_collect_trajectories

    prev_lane_name = None  # used to generate roadway information
    data_base_dir = "./preprocessing/data"  # the directory we used to processing raw data
    total_error = {
        "overall": [],
        "curve": [],
        "lane_change": [],
        "straight": [],
        "time_info": [],
        "orig_traj": [],
        "pred_traj": []
    }
    # directly run for once 
    dir_error = {
        "overall": [],
        "curve": [],
        "lane_change": [],
        "straight": [],
        "time_info": [],
        "orig_traj": [],
        "pred_traj": []
    }
    # run for once
    # convert_raw_ngsim_to_trajdatas()
    print("Finish generating roadway")
    # sys.exit("finish generating converting to trajdata")
    print("Start feature extraction")
    # extract_ngsim_features(output_filename="ngsim_holo_new.h5", n_expert_files=1)
    print("Finish converting and feature extraction")
    # sys.exit("finish generating converting to trajdata")

    fn = "trajdata_holo_trajectories.txt"

    hn = './data/trajectories/ngsim_holo_new.h5'

    if run_args.n_envs:
        args.n_envs = run_args.n_envs 
    # args.env_H should be 200
    sys.stdout.write('{} vehicles with H = {}\n'.format(args.n_envs, args.env_H))

    args.ngsim_filename = fn
    args.h5_filename = hn
    if args.env_multiagent:
        egoids, _ = load_egoids(fn, args, run_args.n_runs_per_ego_id)
    else:
        egoids, _ = load_egoids(fn, args, run_args.n_runs_per_ego_id)
    print("egoids")
    print(egoids)
    # print("starts")
    # print(starts)

    if len(egoids) == 0:
        print("No valid vehicles, skipping")
    error = collect(
        egoids,
        args,
        exp_dir=run_args.exp_dir,
        params_filename=run_args.params_filename,
        use_hgail=run_args.use_hgail,
        n_proc=run_args.n_proc,
        collect_fn=collect_fn,
        random_seed=run_args.random_seed,
        lbd=run_args.lbd,
        adapt_steps=run_args.adapt_steps
    )
    # sys.exit("Already run for one file..")
    # print("\n\nDirectory: {}, file: {} Statistical Info:\n\n".format(dir_name, file_name))
    # utils.print_error(error)
    # dir_error["overall"] += error["overall"]
    # dir_error["curve"] += error["curve"]
    # dir_error["lane_change"] += error["lane_change"]
    # dir_error["straight"] += error["straight"]
    # dir_error["time_info"] += error["time_info"]
    # dir_error["orig_traj"] += error["orig_traj"]
    # dir_error["pred_traj"] += error["pred_traj"]
# print("\n\nDirectory: {} Statistical Info:\n\n".format(dir_name))
# utils.print_error(dir_error)
# total_error["overall"] += dir_error["overall"]
# total_error["curve"] += dir_error["curve"]
# total_error["lane_change"] += dir_error["lane_change"]
# total_error["straight"] += dir_error["straight"]
# total_error["time_info"] += dir_error["time_info"]
# total_error["orig_traj"] += dir_error["orig_traj"]
# total_error["pred_traj"] += dir_error["pred_traj"]
# print("\n\nOverall Statistical Info up to now:\n\n")
# utils.print_error(total_error)

