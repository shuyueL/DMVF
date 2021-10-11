import numpy as np 
import os
import scipy.io as sio
import time
import matplotlib.pyplot as plt 
from scipy import spatial
from multiprocessing import Process, Queue 
from single_agent_func_embedded import single_agent_func
from itertools import combinations
from scipy.special import comb
# from compute_strategy_6v_optimized import compute_strategy_6v
from summary_and_save import interpret_summary_and_save_n_views


def central_controller(split, result_path, frame_num, servers, E_matrix, remote_views, remote_ip):
    # multi-processing
    n_agent = 6
    strategy_records = [0 for i in range(n_agent)]
    strategy_queues = [Queue() for i in range(n_agent)]
    feature_queues = [Queue() for i in range(n_agent)]
    index_queues = [Queue() for i in range(n_agent)]
    endflag_queues = [Queue() for i in range(n_agent)]
    procs = [Process(target=single_agent_func, args=(strategy_queues[i], feature_queues[i], endflag_queues[i],index_queues[i], i,split, servers[i], E_matrix, remote_views, remote_ip)) for i in range(n_agent) if i not in remote_views]

    # initialize strategy
    for i in range(n_agent):
        strategy_records[i] = 1 # all start in Normal.
        # strategy_queues[i].put(1)

    t = time.time()

    # run processes
    for p in procs:
        p.start()

    # container for summary from each view
    sums = {}
    for i in range(n_agent):
        if i in remote_views:
            continue
        sums[i] = []

    p_ends = [False for i in range(n_agent)]
    n_ends = 0
    while n_ends < n_agent - len(remote_views):
        new_content = False
        for i in range(n_agent):
            if i in remote_views:
                continue
            if (not p_ends[i]) and (not endflag_queues[i].empty()):
                if endflag_queues[i].get():
                    p_ends[i] = True
                    n_ends += 1
                    new_content = True
            while (not index_queues[i].empty()):
                sums[i] = np.concatenate((np.array(sums[i]), np.array(index_queues[i].get())), axis = None)
                new_content = True
        if not new_content:
            time.sleep(0.1)

    time.sleep(2) #(TODO: remove for computing the time)

    for p in procs:
        if p.is_alive():
            p.terminate()

    # generate summary and save
    # interpret_summary_and_save_6_views(sum_0, sum_1, sum_2, sum_3, sum_4, sum_5, frame_num, result_path)  
    interpret_summary_and_save_n_views(sums, frame_num, result_path)  
    t = time.time()- t
    return t

