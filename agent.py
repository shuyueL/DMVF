
# class of learning agent
import tensorflow as tf
import numpy as np
from nn import NeuralNet
import random
import math

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

class Memo(object):
	def __init__(self):

		self.state = []
		self.target = []

	def add(self, state, target):
	# state: previous state, the input of the Q net
	# target: the target output of the Q net given the state
		self.state.append(state)
		self.target.append(target)

	def get_size(self):
		return np.shape(self.state)[0]

	def reset(self):
		self.state =[]
		self.target=[]


class Agent(object):
	def __init__(self,layers,batch,explore,explore_l,explore_d,learning,decay,path):
	    # layers: architecture of Q network
		# batch: number of observations in mini-batch set
		# explore: exploration rate
		# decay: future reward decay rate
		# path: model path
		self.layers = layers
		self.batch_size = batch
		self.decay_rate = decay
		self.learning_rate = learning
		self.explore_low = explore_l
		self.explore_decay = explore_d
		self.explore_rate = explore
		self.directory = path
		self.num_action = self.layers[len(self.layers)-1].num_output
		##### build Q network
		self.Q = NeuralNet(self.layers,self.learning_rate,'mean_square', 'RMSprop')
		self.Q.initialize()
		##### data-related variables
		self.feat =[]
		self.gt=[]
		self.memory = Memo()
		self.sel_fast=[]
		self.sel_normal=[]
		self.sel_slow = []


	# initialize with data
	def data_init(self, current_eps):
		self.feat = current_eps.feat
		self.gt = current_eps.gt

	# select an action based on policy
	def policy(self, id_curr, strategy):
		single_action_num = int(self.num_action/3)
		exploration = np.random.choice(range(2),1,p=[1-self.explore_rate,self.explore_rate])
		# exploration==1: explore
		# exploration==0: exploit
		if exploration==1:          # exploration
			action_index = np.random.choice(range(single_action_num),1)[0]  # change for multi

			#print('\r')
			#print('              explore:  '+str(action_index))
			#print('\r')
			action_value = self.Q.forward([self.feat[id_curr]])
			# record average Q value
			self.Q.ave_value.append(np.mean(action_value[0]))
			# self.Q.ave_value = np.append(self.Q.ave_value,np.mean(action_value[0]))

			# print(action_value[0])
			return action_index
		else:                       # exploitation
			action_value = self.Q.forward([self.feat[id_curr]])
			# record average Q value
			self.Q.ave_value.append(np.mean(action_value[0]))
			# self.Q.ave_value = np.append(self.Q.ave_value,np.mean(action_value[0]))
			if strategy == 0:
				action_index = np.argmax(action_value[0][0:single_action_num])
			if strategy == 1:
				action_index = np.argmax(action_value[0][single_action_num:2*single_action_num])
			if strategy == 2:
				action_index = np.argmax(action_value[0][2*single_action_num:3*single_action_num])

			#print('\r')
			##print('exploit:  '+str(action_index))
			#print('\r')
			# print(action_value[0])
			return action_index

	# perform action to get next state
	def action(self, id_curr, a_index):
		id_next = id_curr + a_index+1  #action 0,1,2,...
		'''
		if a_index<=14 :
			id_next = id_curr + a_index +1
		elif a_index<= 24 :
			id_next = id_curr + 16 + (a_index-15)*2 +1
		else :
			id_next = id_curr + 40 + (a_index-25)*5 +1
		'''		
		return id_next
		#id_next = id_curr + a_index*5+1  #action 0,5,10,15....

	# compute the reward	
	# REWARD 1: fast skipping
	def reward_fast(self, id_curr, a_index, id_next):
		# normal reward
		rnormal = self.reward_normal(id_curr, a_index, id_next)
		r_fast = (1 + sigmoid(a_index)/2)*rnormal
		return r_fast

	# REWARD 2: normal skipping
	def reward_normal(self, id_curr,a_index, id_next):
		# gaussian_value = [0.0001,0.0044,0.0540,0.2420,0.3989,0.2420,0.0540,0.0044,0.0001]
		# skipping interval,missing part
		seg_gt = self.gt[0][id_curr+1:id_next]
		total = len(seg_gt)		
		n1=sum(seg_gt)  
		n0=total-n1
		miss =(0.8*n0-n1)/20  #largest action step.
		# accuracy
		acc = 0
		if id_next-4>-1:
			if self.gt[0][id_next-4]==1:
				acc = acc+0.0001
		if id_next-3>-1:
			if self.gt[0][id_next-3]==1:
				acc = acc+0.0044
		if id_next-2>-1:
			if self.gt[0][id_next-2]==1:
				acc = acc +0.0540
		if id_next-1>-1:
			if self.gt[0][id_next-1]==1:
				acc = acc+0.2420
		if self.gt[0][id_next]==1:
			acc = acc + 0.3989
		if id_next+1<len(self.gt[0]):
			if self.gt[0][id_next+1]==1:
				acc = acc+0.2420
		if id_next+2<len(self.gt[0]):
			if self.gt[0][id_next+2]==1:
				acc = acc+0.0540

		if id_next+3<len(self.gt[0]):
			if self.gt[0][id_next+3]==1:
				acc = acc+0.0044
		if id_next+4<len(self.gt[0]):
			if self.gt[0][id_next+4]==1:
				acc = acc+0.0001
		r = miss+acc
		return r

	# REWARD 3: slow skipping
	def reward_slow(self, id_curr, a_index, id_next):
		# normal reward
		rnormal = self.reward_normal(id_curr, a_index, id_next)
		r_slow = (1 - sigmoid(a_index)/2)*rnormal
		return r_slow

	# update target Q value
	def update(self, r, id_curr, id_next, a_index,strategy):
		# strategy 0,1,2: fast, normal,slow
		target = self.Q.forward([self.feat[id_curr]]) # target:[ [] ]
		qval_estimated = self.Q.forward([self.feat[id_next]])[0]

		# target[0][a_index + strategy*25] = r + self.decay_rate * max(qval_estimated[strategy*25:(strategy+1)*25])
		target[0][a_index + strategy*20] = r + self.decay_rate * max(qval_estimated[strategy*20:(strategy+1)*20])
		
		return target

	# run an episode to get case set for training
	def episode_run(self):
		frame_num = np.shape(self.feat)[0]
		# fast strategy
		self.sel_fast = np.zeros(frame_num)
		id_curr = 0
		self.sel_fast[id_curr]=1
		while id_curr < frame_num :
			a_index = self.policy(id_curr,0)
			id_next = self.action(id_curr, a_index)
			if id_next >frame_num-1 :
				break
			self.sel_fast[id_next]=1
			r = self.reward_fast(id_curr,a_index, id_next)
			target_vector = self.update(r, id_curr, id_next, a_index, 0)[0]
			input_vector = self.feat[id_curr]
			self.memorize(input_vector, target_vector)
			if self.memory.get_size() == self.batch_size:
				self.train()
			id_curr = id_next

		# normal strategy
		self.sel_normal = np.zeros(frame_num)
		id_curr = 0
		self.sel_normal[id_curr]=1
		while id_curr < frame_num :
			a_index = self.policy(id_curr,1)
			id_next = self.action(id_curr, a_index)
			if id_next >frame_num-1 :
				break
			self.sel_normal[id_next]=1
			r = self.reward_normal(id_curr,a_index, id_next)
			target_vector = self.update(r, id_curr, id_next, a_index,1)[0]
			input_vector = self.feat[id_curr]
			self.memorize(input_vector, target_vector)
			if self.memory.get_size() == self.batch_size:
				#print('training')
				self.train()
			id_curr = id_next

		# slow strategy
		self.sel_slow = np.zeros(frame_num)
		id_curr = 0
		self.sel_slow[id_curr]=1
		while id_curr < frame_num :
			a_index = self.policy(id_curr,2)
			id_next = self.action(id_curr, a_index)
			if id_next >frame_num-1 :
				break
			self.sel_slow[id_next]=1
			r = self.reward_slow(id_curr,a_index, id_next)
			target_vector = self.update(r, id_curr, id_next, a_index,2)[0]
			input_vector = self.feat[id_curr]
			self.memorize(input_vector, target_vector)
			if self.memory.get_size() == self.batch_size:
				#print('training')
				self.train()
			id_curr = id_next
		
			

	# training Q net using one batch data
	def train(self):
		self.explore_rate = max(self.explore_rate - self.explore_decay, self.explore_low)
		#print('\r')
		#print('--current epsilon: %f ---', self.explore_rate)
		#print(np.shape(self.memory.target))
		x = self.memory.state
		y = self.memory.target
		self.Q.train(x,y)
		self.memory.reset()

	# store current observation to memory
	def memorize(self,state, target):
     	# observation: new observation (s,a,r,s')
		self.memory.add(state, target)

	# reset data-related variables
	def data_reset(self):
		self.feat =[]
		self.gt=[]
		self.sel_fast=[]
		self.sel_normal=[]
		self.sel_slow=[]

	# save trained Q net
	def save_model(self,filename):
		# module backup
		path = self.directory
		self.Q.saving(path,filename)
	'''
	# restore Q net from pretrained model
	def recover_model(self):
		# module recovery
		path = self.directory
		self.Q.recover(path,'Q_net.ckpt','Q_net.npy');
	'''
