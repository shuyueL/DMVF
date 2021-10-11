

# select a video for an episode. can be modified to randomly select.


import scipy.io as sio
import numpy as np

class Episode(object):

    def __init__(self, idx, name, feat_path,gt_path):
    # idx: episode index
    # num: total number of sequences
    # name: sequence name
    # cam: camera name
    # path: sequence feature and gt path
        self.index = idx
        self.name = name[idx] 
        self.feat = []
        self.gt = []
        temp_1= sio.loadmat(feat_path+self.name+'_feat.mat')
        self.feat = temp_1['features']
        temp_2 = sio.loadmat(gt_path+self.name+'_gt.mat')
        self.gt = temp_2['gt'].T

    def get_size(self):
        num_frames = np.shape(self.feat)[0]
        return num_frames
