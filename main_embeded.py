import os
from central_controller import central_controller
# from strategy_consensus import Consensus
import multi_agent_communication as mac
import argparse
import time

n_node = 6

IP_TABLE = ['192.168.0.103', '192.168.0.153', '192.168.0.171',  '192.168.0.158',  '192.168.0.178', '192.168.0.191']
#            TX2             SY's workstation  SY's thin laptop  SY's thick laptop   ZL's laptop    ZL's desktop

SELF_IP_IDX = 1

parser = argparse.ArgumentParser(description='Local client agent.')
parser.add_argument('offset', type=int, help='local view id offset')
args = parser.parse_args()
local_view = (args.offset + SELF_IP_IDX) % 6
remote_views = [i for i in range(n_node) if i != local_view]
print("Remote views:", remote_views)
remote_ip = {i: IP_TABLE[(i+6-args.offset)%6] for i in remote_views}
# print(remote_ip)
# exit(0)

# physical distance
E = [[1, 1, 0, 0, 1, 1],
     [1, 1, 1, 1, 1, 0],
     [0, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 0, 0],
     [1, 1, 0, 0, 1, 0],
     [1, 0, 0, 0, 0, 1]
    ]
# view similarity connections.
E = [[1, 0, 0, 0, 0, 1],
     [0, 1, 0, 1, 0, 0],
     [0, 0, 1, 1, 0, 1],
     [0, 1, 1, 1, 0, 0],
     [0, 0, 0, 0, 1, 1],
     [1, 0, 1, 0, 1, 1]
    ]
# fully connected
E = [[1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1]
    ]
# view sim new graph
E= [[1, 0, 1, 0, 1, 1],
    [0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 1, 0],
    [1, 0, 0, 1, 1, 0],
    [1, 1, 1, 0, 0, 1]]
# cons_agent = Consensus(n_node,E)

servers = {}
for i in range(n_node):
    if i not in remote_views:
        servers[i] = mac.setup_server(i)

time.sleep(10)
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






  