import sys 
sys.path.append("..") 
import network_builder
import networkx as nx
from collections import defaultdict
import numpy as np
from networkx.algorithms import community
from fastnode2vec import Graph, Node2Vec
from gensim.models import KeyedVectors


considered_lang = 'all'
# load from disk
net = network_builder.ConceptNetwork(involved_lang=considered_lang, 
                                     load_directed_graph_from_path=True, use_updated=True,
                                     load_directed_graph_path= \
                                     '/mounts/data/proj/yihong/newhome/ConceptNetwork/network_related/stored_networks')


for num in [1, 5, 10, 20, 50, 100]:
    minimum_number_of_langs_to_consider = num
    print(f"Computing {minimum_number_of_langs_to_consider}...")
    expanded_net = net.expand_graph(minimum_number_of_langs=minimum_number_of_langs_to_consider)

    print(f"Number of nodes: {expanded_net.number_of_nodes()}")
    print(f"Number of edges: {expanded_net.number_of_edges()}")

    store_name = f"vocab_min_{minimum_number_of_langs_to_consider}_updated"
    net.store_vocab(store_name=store_name)

    edge_weight_tuples = []
    for edge in expanded_net.edges:
        edge_weight_tuples.append((edge[0], edge[1], eval(expanded_net.edges[edge]['weight'])))
    
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
    n2v.wv.save(f"./expandednet_vectors_minlang_{minimum_number_of_langs_to_consider}_{emb_dim}_{num_epochs}_updated.wv")
    print(f"{minimum_number_of_langs_to_consider} is done!")
    print()

"""
export PYTHONIOENCODING=utf8; nohup python -u train_different_min_language_embedding.py > ./training_different.txt 2>&1 &
server: delta
pid: 37465
"""