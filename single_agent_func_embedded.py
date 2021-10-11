# test script for multi-case

import tensorflow as tf
import numpy as np
from nn_layer import Layer
from nn import NeuralNet
#from visualization import Plot
from read_data import Episode
from agent import Agent
import scipy.io as sio
from scipy.special import comb
import statistics
import time

from strategy_consensus_distribute import ConsensusAgent

def single_agent_func(inQueue, outQueue, endQueue,indQueue, view, split, server, E_matrix, remote_views, remote_ip):

	# data path and names.
	# local address
	feat_path = '/home/slan/Datasets/Videoweb/Day4/Inception_v1_feat/mat_feat/'
	gt_path = '/home/slan/Datasets/Videoweb/Day4/gt/'
	# server address
	# feat_path = '/home/sla8745/datasets/multi-ego/mat_feat/'
	# gt_path ='/home/sla8745/datasets/multi-ego/binary_gt/'

	# strategy update period T
	T = 1000
	# test data split 
	if split == 1:
		# split 1
		test_name = [\
			'd4_s6_c3','d4_s6_c4','d4_s6_c21','d4_s6_c22','d4_s6_c51','d4_s6_c61'
			]
	if split == 2:
		# split 2
		test_name = [\
		'd4_s1_c3','d4_s1_c4','d4_s1_c21','d4_s1_c22','d4_s1_c51','d4_s1_c61'
		]
	if split == 3:
		# split 3
		test_name = [\
		'd4_s2_c3','d4_s2_c4','d4_s2_c21','d4_s2_c22','d4_s2_c51','d4_s2_c61'
		]
	if split == 4:
		# split 4
		test_name = [\
		'd4_s3_c3','d4_s3_c4','d4_s3_c21','d4_s3_c22','d4_s3_c51','d4_s3_c61'
		]
	if split == 5:
		# split 5
		test_name = [\
		'd4_s4_c3','d4_s4_c4','d4_s4_c21','d4_s4_c22','d4_s4_c51','d4_s4_c61'
		]

	test_num = 1

	l1 = Layer(1024,400,'relu')
	l2 = Layer(400,200,'relu')
	l3 = Layer(200,100,'relu')
	l4_fast = Layer(100,35,'linear')
	l4_normal = Layer(100,25,'linear')
	l4_slow = Layer(100,15,'linear')
	layers_fast = [l1,l2,l3,l4_fast]
	layers_normal = [l1,l2,l3,l4_normal]
	layers_slow = [l1,l2,l3,l4_slow]
	learning_rate = 0.0002
	loss_type = 'mean_square'
	opt_type = 'RMSprop'
	# Load models
	# Models of different splits
	# Useful model names:
	# SingleFFNet_slow:
	# 	Model_sff_slow_0617/0621/0630/0702/0703
	# 		Sff_slow_s1_XXXX_400
	# SingleFFNet:
	# 	Model_sff_0617/0621/0630/0702/0703
	# 		Sff_s1_XXXX_500
	# SingleFFNet_fast:
	# 	Model_sff_fast_0617/0621/0630/0702/0703
	# 		Sff_fast_s1_XXXX_800
	Q_fast = NeuralNet(layers_fast,learning_rate,loss_type, opt_type)
	Q_normal = NeuralNet(layers_normal,learning_rate,loss_type, opt_type)
	Q_slow = NeuralNet(layers_slow,learning_rate,loss_type, opt_type)
	if split == 1:
		Q_fast.recover('/home/slan/multi_fastforward_videoweb/Model_training_day4/SingleFFNet_fast/model_sff_fast_0617/','sff_fast_s1_0617_800')
		Q_normal.recover('/home/slan/multi_fastforward_videoweb/Model_training_day4/SingleFFNet/model_sff_0617/','sff_s1_0617_500')
		Q_slow.recover('/home/slan/multi_fastforward_videoweb/Model_training_day4/SingleFFNet_slow/model_sff_slow_0617/','sff_slow_s1_0617_400')
	if split == 2:
		Q_fast.recover('/home/slan/multi_fastforward_videoweb/Model_training_day4/SingleFFNet_fast/model_sff_fast_0621/','sff_fast_s1_0621_800')
		Q_normal.recover('/home/slan/multi_fastforward_videoweb/Model_training_day4/SingleFFNet/model_sff_0621/','sff_s1_0621_500')
		Q_slow.recover('/home/slan/multi_fastforward_videoweb/Model_training_day4/SingleFFNet_slow/model_sff_slow_0621/','sff_slow_s1_0621_400')
	if split == 3:
		Q_fast.recover('/home/slan/multi_fastforward_videoweb/Model_training_day4/SingleFFNet_fast/model_sff_fast_0630/','sff_fast_s1_0630_800')
		Q_normal.recover('/home/slan/multi_fastforward_videoweb/Model_training_day4/SingleFFNet/model_sff_0630/','sff_s1_0630_500')
		Q_slow.recover('/home/slan/multi_fastforward_videoweb/Model_training_day4/SingleFFNet_slow/model_sff_slow_0630/','sff_slow_s1_0630_400')
	if split == 4:
		Q_fast.recover('/home/slan/multi_fastforward_videoweb/Model_training_day4/SingleFFNet_fast/model_sff_fast_0702/','sff_fast_s1_0702_800')
		Q_normal.recover('/home/slan/multi_fastforward_videoweb/Model_training_day4/SingleFFNet/model_sff_0702/','sff_s1_0702_500')
		Q_slow.recover('/home/slan/multi_fastforward_videoweb/Model_training_day4/SingleFFNet_slow/model_sff_slow_0702/','sff_slow_s1_0702_400')
	if split == 5:
		Q_fast.recover('/home/slan/multi_fastforward_videoweb/Model_training_day4/SingleFFNet_fast/model_sff_fast_0703/','sff_fast_s1_0703_800')
		Q_normal.recover('/home/slan/multi_fastforward_videoweb/Model_training_day4/SingleFFNet/model_sff_0703/','sff_s1_0703_500')
		Q_slow.recover('/home/slan/multi_fastforward_videoweb/Model_training_day4/SingleFFNet_slow/model_sff_slow_0703/','sff_slow_s1_0703_400')

	# neighbor_sockets = neighbor_sockets = mac.setup_connections(view, server, neighbors)
	cons_agent = ConsensusAgent(6, E_matrix, view, server, remote_views, remote_ip, diameter=longest_path(E_matrix))

	t = time.time()
	strategy_time = 0

	# indicate strategy
	strategy = 1 # start at normal
	
	# calculate processing rate.
	proc_per = []
	for i in range(test_num):
		video = Episode(view, test_name, feat_path, gt_path)
		frame_num = np.shape(video.feat)[0]
		feat_transmit = []
		idx_selected = []
		id_curr = 0
		frame_count = 0
		proc_frame = 1

		while 1 :
			

			# transmit the data every 100 frames.
			if frame_count >= T:
				frame_count = frame_count - T
				# outQueue.put(feat_transmit)
				indQueue.put(idx_selected)
				# endQueue.put(0)
				# if strategy != 3:
				# 	strategy = inQueue.get()
				if strategy != 3:
					st = time.time()
					strategy = cons_agent.compute_strategies(feat_transmit, idx_selected) + 1
					strategy_time += time.time() - st
					# print(id_curr, strategy)
				feat_transmit = []
				idx_selected = []
			if id_curr >frame_num-1 :
				break
			feat_transmit.append(video.feat[id_curr].tolist())
			idx_selected.append(id_curr)

			if strategy == 0:
				action_value = Q_fast.forward([video.feat[id_curr]])
				a_index = np.argmax(action_value)
			if strategy == 1:
				action_value = Q_normal.forward([video.feat[id_curr]])
				a_index = np.argmax(action_value)
			if strategy == 2:
				action_value = Q_slow.forward([video.feat[id_curr]])
				a_index = np.argmax(action_value)
			if strategy == 3:
				action_value = Q_normal.forward([video.feat[id_curr]])
				a_index = np.argmax(action_value)
			frame_count = frame_count + a_index +1
			id_next = id_curr + a_index+1
			if id_next >frame_num-1 :
				frame_count = frame_count - (id_next - (frame_num-1))
				# break
			proc_frame = proc_frame + 1
			id_curr = id_next
		proc_per.append(proc_frame/frame_num)

	if strategy != 3:
		st = time.time()
		strategy = cons_agent.compute_strategies(None, None) + 1
		strategy_time += time.time() - st
	
	t = time.time() - t
	with open("time_results_remote.txt", "a") as f:
		print("split", split, "view", view, "time", t, "stategy computation time", strategy_time, file=f)

	endQueue.put(1)
	# print(proc_per)

	comm_stat_p2p, com_stat_bc, com_time, com_proc_time = cons_agent.communication_statistics()

	savepath = 'proc_rate_agent_' + str(view) + '_split_' + str(split)
	sio.savemat(savepath,{'proc_rate': statistics.mean(proc_per), 'comm_stat_p2p': comm_stat_p2p/frame_num, 'com_stat_bc': com_stat_bc/frame_num, "time": t, "stategy_computation_time": strategy_time, "communication_time": com_time, "communication_process_time": com_proc_time})
	print('agent: ',view, ' end. proc_per = ', statistics.mean(proc_per))

	cons_agent.close_connection()

def longest_path(E):
	n,m = np.shape(E)
	longest = 0
	for i in range(n):
		bfs_queue = []
		bfs_record = np.zeros(n)
		bfs_queue.append(i)
		while bfs_queue:
			j = bfs_queue.pop(0)
			for k in range(m):
				if k != i and bfs_record[k] == 0 and E[j][k] > 0:
					bfs_record[k] = bfs_record[j] + 1
					bfs_queue.append(k)
		longest = max(longest, np.max(bfs_record))
	return int(longest)
if __name__ == "__main__":
	# view similarity connections.
	E = [[1, 0, 0, 0, 0, 1],
	     [0, 1, 0, 1, 0, 0],
	     [0, 0, 1, 1, 0, 1],
	     [0, 1, 1, 1, 0, 0],
	     [0, 0, 0, 0, 1, 1],
	     [1, 0, 1, 0, 1, 1]
	    ]
	print(longest_path(E))