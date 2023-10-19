import eflomal_network_builder
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# load eflomal undirected network
net = eflomal_network_builder.EflomalAlignmentNetwork(load_graph_from_path=True)

print(f"Number of nodes: {net.concept_net.number_of_nodes()}")
print(f"Number of edges: {net.concept_net.number_of_edges()}")

edge_weight_tuples = []
for edge in net.concept_net.edges:
    edge_weight_tuples.append((edge[0], edge[1], net.concept_net.edges[edge]['count']))
    
    
from fastnode2vec import Graph, Node2Vec

# buiding a weighted graph which is is required by fastnode2vec
graph = Graph(edge_weight_tuples, directed=False, weighted=True)


# training hyperparameters
emb_dim = 200
walk_length = 100
unormalized_return_prob = 0.5  # 1/unormalized_return_prob is the prob
unormalized_moving_prob = 2.0  # 1/unormalized_moving_prob is the prob
context_size = 10  # context_size/2 is the window size (half before the word and half after)
num_workers = 16
num_epochs = 10  # number of iterations over the corpus

n2v = Node2Vec(graph, dim=emb_dim, walk_length=walk_length, context=context_size, 
               p=unormalized_return_prob, q=unormalized_moving_prob, workers=num_workers)
n2v.train(epochs=num_epochs)


n2v.wv.save(f"./eflomal_vectors_{emb_dim}_{num_epochs}.wv")
print('Saved!')


"""
export PYTHONIOENCODING=utf8; nohup python -u eflomal_training.py > ./eflomal_training.txt 2>&1 &
server: delta
pid: 42632
"""
