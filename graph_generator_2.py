import networkx as nx
import random
import matplotlib.pyplot as plt



def random_edge(graph,):
    '''
    Create a new random edge and delete one of its current edge if del_orig is True.
    :param graph: networkx graph
    :param del_orig: bool
    :return: networkx graph
    '''
    # edges = list(graph.edges)
    # print('edge list= ', edges)
    nonedges = list(nx.non_edges(graph))
    # print('nonedges= ', nonedges)

    # random edge choice
    chosen_nonedge = random.choice(nonedges)

    # add new edge
    graph.add_edge(chosen_nonedge[0], chosen_nonedge[1])

    return graph



g = nx.Graph()
g.add_edges_from([(0, 2), (0, 5), (1, 3), (2, 4), (3, 4)])
# g.add_edges_from([(0, 2), (0, 5), (0, 3), (0, 1), (0, 4), (2, 4), (5, 3), (1, 3), (3, 4)])
# nx.draw(g, with_labels=True)
# plt.show()

for i in range(10):
    g = random_edge(g)
    E = nx.adjacency_matrix(g)
    print(g.edges)
    print(E.todense())
    print('number of edges = ', len(g.edges))
    print('diameter = ',nx.diameter(g))
    # nx.draw(g, with_labels=True)
    # plt.show()