from networkx.generators.random_graphs import erdos_renyi_graph
import networkx as nx
import numpy as np

n = 6
p = 0.3
g = erdos_renyi_graph(n,p)

E = nx.adjacency_matrix(g)

print(g.edges)
print(E.todense())
print('number of edges = ', len(g.edges))
print('diameter = ',nx.diameter(g))
print('connectivity = ',nx.is_connected(g))