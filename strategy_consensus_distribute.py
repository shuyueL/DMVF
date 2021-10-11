from math import*
import numpy as np 
import time
import multi_agent_communication as mac

# similarity between 2 frames.
def l2norm_distance(x, y):
        """ return l2norm distance between two lists"""
        # print(len(x))
        # print(len(y))
        return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

def similarity(a, b):
    return np.exp(-l2norm_distance(a,b)/20)

# Compute all similarity between each pair of frames from two different views
# frames_all_views: a list contains frames from all views
class SimilarityTable:
    def __init__(self, frames_all_views = None):
        self.similarity_table = []
        self.n_view = 0
        self.n_frame_of_view = []
        if frames_all_views is not None:
            self.group_similarity_compute(frames_all_views)

    def group_similarity_compute(self, frames_all_views):
        self.similarity_table = []
        self.n_frame_of_view = []
        self.n_view = len(frames_all_views)
        for i in range(self.n_view):
            self.n_frame_of_view.append(len(frames_all_views[i]))
        for i in range(self.n_view-1):
            view_dict = {}
            for j in range(i+1, self.n_view):
                similarity_list = [[0 for k in range(self.n_frame_of_view[j])] for l in range(self.n_frame_of_view[i])]
                for l in range(self.n_frame_of_view[i]):
                    for k in range(self.n_frame_of_view[j]):
                        similarity_list[l][k] = similarity(frames_all_views[i][l], frames_all_views[j][k])
                view_dict[j] = similarity_list
            self.similarity_table.append(view_dict)
    
    def similarity(self, view1, frame1, view2, frame2):
        try:
            if view1 > view2:
                tmp = view2
                view2 = view1
                view1 = tmp
                tmp = frame2
                frame2 = frame1
                frame1 = tmp
            return self.similarity_table[view1][view2][frame1][frame2]
        except:
            print("DBG: Exception occurs when calculate similarity in Similarity Table. (This should not happen.)")
            return 0

# # match one view with all views in main.
# # match_id: the id of frames in f which are matched to mv
# # match_count: the number of matched frames in f.
# # def match_one_view_to_main(mv, f, mvi, fi, thres):
def match_one_view_to_main(mains, other, frames, indices, similarityTable):

    # n = len(f)
    n = len(frames[other])
    sim_sum = 0
    for i in range(n):
        # for j in range(len(mv)):
        #     for k in range(len(mv[j])):
        #         s = similarity(f[i], mv[j][k])
        sim_max = 0
        for j in mains:
            for k in range(len(frames[j])):
                s = similarityTable.similarity(other, i, j, k) # frames[other][i] vs frames[j][k]
                sim_max = max(sim_max, s)
        sim_sum += sim_max
    return sim_sum/n

# compute the match score of a set of main views.
# matches: match ids for each view besides main
# score: the score for this main subset.
# method 1: add up all counts with/without dividing
# def match_score(mv, f, mvi, fi, thres):
def match_score(mains, others, frames, indices, similarityTable):
    score = 0

    # Denominator
    nFrmInMain = 0
    # for i in range(len(mv)):
    #     nFrmInMain = nFrmInMain + len(mv[i])
    for i in mains:
        nFrmInMain = nFrmInMain + len(frames[i])
    
    # Numerator
    # for i in range(len(f)):
    #     match, count = match_one_view_to_main(mv, f[i], mvi, fi[i], thres)
    for i in others:
        sim_ave = match_one_view_to_main(mains, i, frames, indices, similarityTable)
        score = score + sim_ave
    # (TODO: whether to divide)
    # if nFrmInMain:
    # 	score = score/(nFrmInMain*len(others))
    # else:
    #     score = 0
    score = score/len(others)
    return score #, matches, counts

class ConsensusAgent():
    def __init__(self, n_node, e_mtx, view, server, remote_views, remote_ip, diameter=None):
        if n_node != 6:
            print("Not supported: n_node != 6. Check compute_strategies()")
        self.n_node = n_node
        self.E = e_mtx
        self.view = view
        self.server = server
        if diameter is None:
            self.diameter = self.n_node - 1
        else:
            self.diameter = diameter

        self.neighbors = self.E[self.view]
        self.B = np.diag(np.sum(self.E, axis=1))
        self.build_P()
        self.neighbor_sockets = mac.setup_connections(self.view, self.server, self.neighbors, remote_views, remote_ip)
        self.total_comm_p2p = 0
        self.total_comm_bc = 0
        self.total_comm_time = 0
        self.total_comm_process_time = 0

    def close_connection(self):
        mac.close_connections(self.neighbor_sockets)

    def build_P(self):
        D = np.sum(self.E, axis = 1) - 1
        D = np.diag(D)
        A = self.E - np.eye(self.n_node)
        self.P = A/self.n_node + np.eye(self.n_node) - D/self.n_node

    def transition(self, x_t, x_0, gamma):
        x_t_1 = np.matmul(self.P, x_t) - gamma * np.matmul(np.linalg.inv(self.B), self.E)*(x_t-x_0)
        return x_t_1


    def get_consensus_dgd(self, x_0, max_iter, stop_err):
        x =x_0
        k = 1
        for i in range(max_iter):
            gamma = 10/(i+1)
            x = self.transition(x, x_0, gamma)
            if i > 0 and np.log10(i) >= k:
                # print(x)
                k += 1
            if max(np.max(x, axis=0) - np.min(x, axis=0)) < stop_err:
                print("Consensus reached at step", i)
                break
        # print(np.std(x,axis=0))
        # print(x)
        return x

    def get_consensus(self, x_0):
        x =np.zeros_like(x_0)
        for i in range(len(x)):
            sum_score = 0
            sum_weight = 0
            for j in range(len(x[0])):
                if self.E[i][j] > 0 :
                    sum_weight += 1/(sum(self.E[j]) - 1)
                    sum_score += x_0[j][i]/(sum(self.E[j])-1)
            x[i][i] = sum_score/sum_weight
        for i in range(len(x)):
            for j in range(len(x[0])):
                x[i][j] = x[j][j]
        # print(x)
        return x

    def max_consensus(self, x0):
        for i in range(self.diameter):
            X = self.broadcast_and_recv(x0, i)
            for i in range(self.n_node):
                for j in X:
                    x0[i] = max(x0[i], X[j][i])
        return x0
    
    def weighted_average(self, X0):
        sum_score = 0
        sum_weight = 0
        for i in X0:
            sum_weight += 1/(sum(self.E[i]) - 1)
            sum_score += X0[i][self.view]/(sum(self.E[i])-1)
        x_wa = np.zeros(self.n_node)
        x_wa[self.view] = sum_score/sum_weight
        return x_wa

    def broadcast_and_recv(self, own_data, dbg = -1):
        t = time.time()
        t_proc = time.process_time()
        msg_size = 0
        for i in self.neighbor_sockets:
            s = self.neighbor_sockets[i]
            msg_size = mac.send_message(s, own_data)
            self.total_comm_p2p += msg_size
        # print("[Agent", self.view, "] broadcasted.", dbg)
        self.total_comm_bc += msg_size

        neighbor_data = {}
        for i in self.neighbor_sockets:
            s = self.neighbor_sockets[i]
            neighbor_data[i] = mac.receive_message(s)
        # print("[Agent", self.view, "] received.", dbg)
        self.total_comm_time += time.time()- t
        self.total_comm_process_time += time.process_time()-t_proc
        return neighbor_data

    def compute_strategies(self, f_own, ind_own):
        own_data = (f_own, ind_own)

        neighbor_data = self.broadcast_and_recv(own_data)
        
        done = False
        if f_own is None:
            done = True
        f = [f_own]
        ind = [ind_own]
        id_map = [self.view]
        for i in neighbor_data:
            f_i, ind_i = neighbor_data[i]
            if f_own is None:
                done = True
            f.append(f_i)
            ind.append(ind_i)
            id_map.append(i)

        x0 = np.zeros(self.n_node) # local copy of the initial scores
        if not done:
            # setup the similarity table for looking up
            # t = time.time()
            similarityTable = SimilarityTable(f)
            # print("time for tablebuild=", time.time()-t)

            # X0 = np.zeros((self.n_node, self.n_node))
            for j in range(len(id_map)):
                mains = [j]
                others = [k for k in range(len(id_map)) if k != j]
                score = match_score(mains, others, f, ind, similarityTable)
                x0[id_map[j]] = score
        
        X0 = self.broadcast_and_recv(x0)
        X0[self.view] = x0

        if not done:
            x0 = self.weighted_average(X0)
        else:
            x0 = np.zeros(self.n_node)
            x0[self.view] = 5.0
        # print("[Agent", self.view, "] weighted_average:", x0)

        xc = self.max_consensus(x0)
        
        # Xc = self.get_consensus(X0)
        # err = 1e-3
        # print(X0)
        # X_dgd = self.get_consensus_dgd(X0,2000, err)
        # print(np.max(np.abs(Xc-X_dgd)))

        print("[Agent", self.view, "] score:", xc)

        if max(xc) > 2:
            return 2

        ranks = np.argsort(np.argsort(xc))
        # ranks_dgd = np.argsort(np.argsort(X_dgd[i]))
        # print(ranks-ranks_dgd)
        rank = ranks[self.view]
        if rank < 3:
            s = -1
        elif rank < 5:
            s = 0
        else:
            s = 1
        strategy = s

        return strategy

    def communication_statistics(self):
        return self.total_comm_p2p, self.total_comm_bc, self.total_comm_time, self.total_comm_process_time