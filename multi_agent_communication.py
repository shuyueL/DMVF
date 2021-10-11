import select, socket, sys
import pickle
from multiprocessing import Process, Queue 
import time
import numpy as np

HEADERSIZE = 10
N_AGENTS = 6

def test_agent_communication(view, server, neighbors):

    neighbor_sockets = setup_connections(view, server, neighbors)

    time.sleep(2)

    for i in neighbor_sockets:
        s = neighbor_sockets[i]
        feedback = "hello, Agent "+str(i)+", this is agent "+str(view)+", what's up man!"
        send_message(s, feedback)
        
    for i in neighbor_sockets:
        s = neighbor_sockets[i]
        recv_msg = receive_message(s)
        print("[Agent", view, "] received from Neighbor", i, ":", recv_msg)
    time.sleep(2)

    print("[Agent", view, "] Connections closing")
    close_connections(neighbor_sockets)

def setup_connections(view, server, neighbors, remote_views = [], remote_ips = {}):
    neighbor_sockets = {}
    for i in range(view):
        if neighbors[i] == 0:
            continue
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if i in remote_views:
            ip_address = remote_ips[i]
        else:
            ip_address = 'localhost'
        try:
            s.connect((ip_address, 25894 + i))
        except ConnectionRefusedError as e:
            print(e)
            s.close()
            return False
        neighbor_sockets[i] = s
        send_message(s, view)
        print("[Agent", view, "] connected to Neighbor", i)
    for i in range(view+1, len(neighbors)):
        if neighbors[i] == 0:
            continue
        client, client_address = server.accept()
        neighbor = receive_message(client)
        neighbor_sockets[neighbor] = client
        print("[Agent", view, "] connected to Neighbor", neighbor, "from address", client_address)
    return neighbor_sockets

def close_connections(neighbor_sockets):
    for i in neighbor_sockets:
        neighbor_sockets[i].close()

def send_message(s, message):
    msg = pickle.dumps(message)
    msg_size = len(msg)
    msg = bytes("{0:<{hdsz}}".format(len(msg), hdsz=HEADERSIZE), 'utf-8')+msg
    s.sendall(msg)
    return msg_size

def receive_message(s):
    message_table = {"len":-1, "data":b'', "ready":False}
    while not message_table["ready"]:
        if message_table["len"] < 0:
            bufferSize = HEADERSIZE - len(message_table["data"])
        else:
            bufferSize = min(1024, message_table["len"] + HEADERSIZE - len(message_table["data"]))
        try:
            data = s.recv(bufferSize)
        except ConnectionResetError as e:
            print(e)
            # s.close()
            # return False
        if data:
            message_table["data"] += data
            if message_table["len"] < 0 and len(message_table["data"]) >= HEADERSIZE:
                message_table["len"] = int(message_table["data"][:HEADERSIZE])
            if message_table["len"] == len(message_table["data"]) - HEADERSIZE:
                message_table["ready"] = True
            # if message_table["len"] < len(message_table["data"]) - HEADERSIZE:
            #     print(message_table["len"], len(message_table["data"]) - HEADERSIZE, len(data))
        else:
            print("connection aborted")
        #     raise
    return pickle.loads(message_table["data"][HEADERSIZE:])

def setup_server(view):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('', 25894 + view))
    server.listen(5)
    return server

def close_server(server):
    server.close()

if __name__ == "__main__":
    E = [[1, 1, 0, 0, 1, 1],
     [1, 1, 1, 1, 1, 0],
     [0, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 0, 0],
     [1, 1, 0, 0, 1, 0],
     [1, 0, 0, 0, 0, 1]
    ]
    servers = [setup_server(i) for i in range(N_AGENTS)]
    procs = [Process(target=test_agent_communication, args=(i, servers[i], E[i])) for i in range(N_AGENTS)]
    for p in procs:
        p.start()

    # end = 0
    # t = time.time()
    # while end == 0:
    #     f = feature_queues[0].get()
    #     idx = index_queues[0].get()
    #     end = endflag_queues[0].get()
    #     # time.sleep(1)
    #     stg = np.random.choice([0,1,2])
    #     print("server:", np.shape(f), idx[0], end, stg)
    #     strategy_queues[0].put(stg)
    # print(t - time.time())
    # time.sleep(2)
    for p in procs:
        p.join()

    for s in servers:
        s.close()
    print("Main function finish.")