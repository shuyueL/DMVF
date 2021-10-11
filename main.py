import os
from central_controller import central_controller
from get_E_of_graph import read_E_matrix
# from strategy_consensus import Consensus
import multi_agent_communication as mac
import argparse
import time

parser = argparse.ArgumentParser(description='Remote client agent.')
parser.add_argument('view', type=int, nargs='*', help='remote view id')
args = parser.parse_args()
remote_views = args.view
print("Remote views:", remote_views)
remote_ip = {i:'192.168.0.103' for i in remote_views}

n_node = 6

E = read_E_matrix()

servers = {}
for i in range(n_node):
    if i not in remote_views:
        servers[i] = mac.setup_server(i)

time.sleep(5)
# split number
for split in range(1,6):

    # frame number
    # split 1 8474, split 2 8488, split 3: 5434, split 4 6946, split 5 6668 
    frms = [8474, 8488, 5434, 6946, 6668]
    frame_num = frms[split - 1] 

    # result folder (TODO)
    result_path = "split_" + str(split) + '/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        print("Directory " , result_path,  " Created ")
    else:    
        print("Directory " , result_path,  " already exists")

    print(result_path)

    # call central controller for a run
    central_controller(split, result_path, frame_num, servers, E, remote_views, remote_ip)

# print('average iteration = ', cons_agent.iter_total/cons_agent.iter_count)

for i in servers:
    mac.close_server(servers[i])






  