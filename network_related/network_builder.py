import networkx as nx
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# obtain all distinct language's ISO 639-3 codes
def get_langs():
    csv_path = '/mounts/Users/student/yihong/Documents/concept_align'
    pbc_info = pd.read_csv(f"{csv_path}/pbc_table.csv", converters={"language_code": str})
    langs = pbc_info['language_code'].values
    langs = sorted(list(set(langs)))
    return langs


def read_concepts(concept_path):
    concept_list = []
    with open(concept_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if len(line.strip().split(',')) > 1:
                concept_list.append(line.strip().split(',')[0])
            else:
                concept_list.append(line.strip())
    return concept_list


def draw_graph(graph, node_size=10, node_color='steelblue', edge_color='darkkhaki', edge_width=2,
               use_different_widths=False, font_family='serif', font_size=2,
               include_labels=True, maximum_edge_width_multiplier=5):
    pos = nx.spring_layout(graph)
    node_sizes = [node_size for _ in graph.nodes]
    node_colors = [node_color for _ in node_sizes]
    edge_colors = [edge_color for _ in graph.edges]

    # use different widths for edges with different number of languages / language families
    if use_different_widths:
        max_language = max([len(graph.edges[e]['lang']) for e in graph.edges()])
        # set the width of an edge with the most languages to be edge_width * 5
        # and set the minimum width to be edge_width
        edge_widths = [min(edge_width*maximum_edge_width_multiplier,
                       max([len(graph.edges[e]['lang'])*5/max_language, edge_width])) for e in graph.edges()]
    else:
        edge_widths = [edge_width for _ in graph.edges]

    nx.draw_networkx_nodes(graph, pos=pos, node_size=node_sizes, alpha=0.8, node_color=node_colors)
    nx.draw_networkx_edges(graph, pos=pos, edgelist=graph.edges, width=edge_widths, alpha=0.8, edge_color=edge_colors)
    if include_labels:
        labels = {node: graph.nodes[node]['name'].replace('$', '\$') for node in graph.nodes}
        nx.draw_networkx_labels(graph, pos=pos, labels=labels, font_family=font_family, font_size=font_size, alpha=0.8)
    plt.tight_layout(pad=0.5)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.show()


def merge_dict(dict1: dict, dict2: dict, merge_type='union'):
    """
    merge two dictionaries, the value is a set so we want to get the union of them
    :param dict1:
    :param dict2:
    :param merge_type:
    :return:
    """
    assert merge_type in ['union', 'intersection']
    for k, v in dict2.items():
        if k in dict1:
            if merge_type == 'union':
                dict1[k] = dict1[k].union(v)
            else:
                dict1[k] = dict1[k].intersection(v)
                # see if it is empty
                # we should move this key (language)
                if len(dict1[k]) == 0:
                    dict1.pop(k)
        else:
            dict1[k] = dict2[k]
    return dict1


class ConceptNetwork:
    def __init__(self, involved_lang, load_directed_graph_from_path=False, load_directed_graph_path='',
                 src_lang='eng', results_path='/mounts/data/proj/yihong/newhome/ConceptNetwork/results',
                 use_updated=False):
        """
        :param involved_lang: iso code for the language, or 'all'
        :param load_directed_graph_path: the path to load a stored directed graph
        :param src_lang: the source language
        :param results_path: the path storing the computed results
        """

        # path related
        parallel_dir = f"../required_files/parallel_data/{src_lang}" if not use_updated \
            else f"../required_files/parallel_data/{src_lang}/updated"
        concept_path = f"{parallel_dir}/concept_list"

        results_path = f"{results_path}/{src_lang}" if not use_updated else f"{results_path}/{src_lang}/updated"

        # control the path
        self.use_updated = use_updated

        # when load the data, we have to build directed graph
        self.concept_net = nx.DiGraph()

        # assign objects one the two functions are called
        self.expanded_graph = None
        self.undirected_graph = None

        # load directed graph from disk
        if load_directed_graph_from_path:
            if len(load_directed_graph_path) == 0:
                self.load_net()
            else:
                self.load_net(load_directed_graph_path)
            self.added_concepts = set([node for node in self.concept_net.nodes])
            self.added_edges = set()
            for edge in self.concept_net.edges:
                source, end = edge[0], edge[1]
                self.added_edges.add(f"{source}-{end}")
        else:
            # load concepts
            concept_list = read_concepts(concept_path)
            if involved_lang == 'all' or isinstance(involved_lang, list):
                tgt_results = dict()
                if involved_lang == 'all':
                    langs = get_langs()
                else:
                    langs = involved_lang
                for tgt_lang in langs:
                    with open(f"{results_path}/{tgt_lang}_results_lemmata.pickle", 'rb') as handle:
                        tgt_results[tgt_lang] = pickle.load(handle)
            else:
                tgt_results = dict()
                try:
                    with open(f"{results_path}/{involved_lang}_results_lemmata.pickle", 'rb') as handle:
                        tgt_results[involved_lang] = pickle.load(handle)
                except:
                    print('There is no such language / computed file!')

            # build a graph given the data
            self.added_concepts = set()
            self.added_edges = set()
            for concept in concept_list:

                # add focal concepts into the graph
                self.concept_net.add_node(concept, name=concept, ctype='focal', lang=set())
                self.added_concepts.add(concept)

                # add edges and identified concepts (not focal concepts) into the graph
                for lang in tgt_results.keys():
                    fp_results = tgt_results[lang][concept]['fp']
                    bp_results = tgt_results[lang][concept]['bp']

                    # for some languages we might be not have that concept
                    if len(fp_results) == 0:
                        continue
                    # add the current lang to the concept it we get anything in this language.
                    self.concept_net.nodes[concept]['lang'].add(lang)

                    # including the statistics obtained in FP into the network
                    tgt_lang_ngrams_freq = {ngram: v[1][0][0] for ngram, v in fp_results.items()}
                    tgt_lang_ngrams = set(tgt_lang_ngrams_freq.keys())
                    total_fp_total_freq = fp_results[list(tgt_lang_ngrams_freq.keys())[0]][1].sum(axis=0)[0]
                    tgt_lang_ngrams_freq = {ngram: f"{str(v)}/{str(total_fp_total_freq)}"
                                            for ngram, v in tgt_lang_ngrams_freq.items()}
                    if len(bp_results) == 0 or bp_results == 'NA':
                        # 1-to-1 mapping / no BP is performed
                        eng_lang_concepts_freq = {concept: f"1/1"}
                        if f"{concept}-{concept}" not in self.added_edges:
                            # the target-language ngrams identified in FP through which we identify the association

                            self.concept_net.add_edge(concept, concept,
                                                      lang={lang},
                                                      realizations={lang: tgt_lang_ngrams},
                                                      stats_fp={lang: tgt_lang_ngrams_freq},
                                                      stats_bp={lang: eng_lang_concepts_freq})
                            self.added_edges.add(f"{concept}-{concept}")
                        else:
                            self.concept_net.edges[concept, concept]['lang'].add(lang)
                            self.concept_net.edges[concept, concept]['realizations'][lang] = tgt_lang_ngrams
                            self.concept_net.edges[concept, concept]['stats_fp'][lang] = tgt_lang_ngrams_freq
                            self.concept_net.edges[concept, concept]['stats_bp'][lang] = eng_lang_concepts_freq
                    else:
                        # including the statistics obtained in BP into the network
                        eng_lang_concepts_freq = {c: v[1][0][0] for c, v in bp_results.items()}
                        total_bp_total_freq = bp_results[list(eng_lang_concepts_freq.keys())[0]][1].sum(axis=0)[0]
                        eng_lang_concepts_freq = {c: f"{str(v)}/{str(total_bp_total_freq)}"
                                                  for c, v in eng_lang_concepts_freq.items()}

                        # there are multiple concepts identified in BP
                        for k, v in bp_results.items():
                            # add the identified concepts
                            if k not in self.added_concepts:
                                # only add the the concepts that are not focal into the self.added_concepts
                                self.concept_net.add_node(k, name=k, ctype='identified', lang={lang})
                                self.added_concepts.add(k)
                            else:
                                # add the language to the current identified concept (reachable in this lang)
                                self.concept_net.nodes[k]['lang'].add(lang)

                            # add edges
                            if f"{concept}-{k}" not in self.added_edges:
                                self.concept_net.add_edge(concept, k, lang={lang},
                                                          realizations={lang: tgt_lang_ngrams},
                                                          stats_fp={lang: tgt_lang_ngrams_freq},
                                                          stats_bp={lang: eng_lang_concepts_freq})
                                self.added_edges.add(f"{concept}-{k}")
                            else:
                                # for each focal concept, there can only be one group of ngrams identified
                                # through focal concept -(FP)-> target ngrams -(BP)-> identified concepts
                                self.concept_net.edges[concept, k]['stats_bp'][lang] = eng_lang_concepts_freq
                                self.concept_net.edges[concept, k]['stats_fp'][lang] = tgt_lang_ngrams_freq
                                self.concept_net.edges[concept, k]['realizations'][lang] = tgt_lang_ngrams
                                self.concept_net.edges[concept, k]['lang'].add(lang)

    def is_considered(self, source, end, minimum_number_of_langs=1):
        """
        :param source: the source of the edge
        :param end: the end of the edge
        :param minimum_number_of_langs: the minimum number of langs for an edge to be considered
        :return: boolean value indicating whether to consider an edge between the source and end node
        """
        if minimum_number_of_langs == 1:
            return True
        else:
            # # if the edge is two-sided, we consider the edge with the largest number of lang
            # two_sided = False
            strength_of_collection = len(self.concept_net.edges[source, end]['lang'])
            if (end, source) in self.concept_net.edges:
                two_sided = True
                strength_of_collection_ = len(self.concept_net.edges[end, source]['lang'])
                strength_of_collection = max(strength_of_collection, strength_of_collection_)
            # only consider a edge from the original graph if there are multiple languages have such an associations
            # if (strength_of_collection < minimum_number_of_langs) and not two_sided:
            if strength_of_collection < minimum_number_of_langs:
                # if it is two-sided, the associations should be highly accurate, we should keep
                return False
        return True

    def to_undirected(self, aggregate_type='union', minimum_number_of_langs=1):
        """
        this function is for demonstrating the structure of
        English concepts (one edge is created between two English concepts)
        :param aggregate_type: the aggregate type for edges that share source and end nodes in the directed graph
        :param minimum_number_of_langs: the minimum number of langs for an edge to be considered
        :return:
        """
        assert isinstance(self.concept_net, nx.DiGraph)
        assert aggregate_type in ['union', 'intersection']
        # update edges
        undirected_graph = nx.Graph()

        # the nodes of the undirected_graph is the not same as the directed graph if minimum_number_of_langs > 1
        # the edges would be an integration of the original graph, especially when two-sided edges are present

        # add the nodes to the undirected graph
        undirected_graph.add_nodes_from(self.concept_net.nodes(data=True))

        # only consider nodes that are reachable in more than minimum_number_of_langs
        # so we could avoid some less accurate nodes/edges due to instability of algorithm/misalignment
        nodes_to_be_removed = [node for node in self.concept_net
                               if len(self.concept_net.nodes[node]['lang']) < minimum_number_of_langs]
        undirected_graph.remove_nodes_from(nodes_to_be_removed)

        # also consider the edges
        # add the edges to the undirected graph
        for edge in self.concept_net.edges:
            source, end = edge[0], edge[1]

            # only consider the edge in the original directed graph whose
            # source and end are both preserved after the removal
            if undirected_graph.has_node(source) and undirected_graph.has_node(end):
                pass
            else:
                continue

            # see if this edge should be considered given current criterion of minimum number of languages
            if not self.is_considered(source, end, minimum_number_of_langs=minimum_number_of_langs):
                continue

            if source == end:
                # recurrent edges connecting the focal concepts themselves
                undirected_graph.add_edge(source, end)
                undirected_graph._adj[source][end]['lang'] = self.concept_net._adj[source][end]['lang'].copy()
                undirected_graph._adj[source][end]['realizations'] = \
                    self.concept_net._adj[source][end]['realizations'].copy()
            elif (end, source) not in self.concept_net.edges:
                # one-sided edges in the directed graph
                """
                the following code does not work: 
                (because it seems the attributes of bidirected edges cannot be wisely copied to undirected edges)
                undirected_graph.add_edge(source, end)
                undirected_graph._adj[source][end] = self.concept_net._adj[source][end]
                """
                # for the undirected graph, we don't need to include the stats
                undirected_graph.add_edge(source, end,
                                          lang=self.concept_net.adj[source][end]['lang'].copy(),
                                          realizations=self.concept_net.adj[source][end]['realizations'].copy())
                # print(source, end)
                # print(undirected_graph.edges[source, end])
                # print(undirected_graph.edges[end, source])
            else:
                # two-sided edges in the directed graph between two focal concepts
                # this condition must be that both source and end are the focal concepts
                # aggregate the information of the two involved edges
                if aggregate_type == 'union':
                    lang = self.concept_net._adj[source][end]['lang'].union(self.concept_net._adj[end][source]['lang'])
                    # print(lang)
                    # e.g., fowl -> bird {'deu': {'vogel'}, 'zho': {'鸟'}} merge with
                    # bird -> fowl {'deu': {'vogel'}, 'zho': {'飞禽'}}
                    # {'deu': {'vogel'}, 'zho': {'鸟', '飞禽'}}
                    # create a new copy of the dictionary
                    realizations = merge_dict(self.concept_net._adj[source][end]['realizations'].copy(),
                                              self.concept_net._adj[end][source]['realizations'].copy(),
                                              merge_type=aggregate_type)
                    # print(realizations)
                else:
                    # e.g., fowl -> bird {'deu': {'vogel'}, 'zho': {'鸟'}} merge with
                    # bird -> fowl {'deu': {'vogel'}, 'zho': {'飞禽'}}
                    # {'deu': {'vogel'}, 'zho': {}}
                    realizations = merge_dict(self.concept_net._adj[source][end]['realizations'].copy(),
                                              self.concept_net._adj[end][source]['realizations'].copy(),
                                              merge_type=aggregate_type)
                    # lang should be updated according to the realizations
                    lang = set(realizations.keys())
                # add edge if the set of language is not none
                if len(lang) > 0:
                    undirected_graph.add_edge(source, end, lang=lang, realizations=realizations)
                # when len(lang) == 0, we will not add this edge into the undirected graph

        # remove the nodes whose degree is zero
        zero_degree_nodes = [n for n, v in dict(undirected_graph.degree()).items() if v == 0]
        undirected_graph.remove_nodes_from(zero_degree_nodes)

        self.undirected_graph = undirected_graph
        return undirected_graph

    def expand_graph(self, minimum_number_of_langs=1):
        """
        (this function is for creating a network including target-language ngrams
        the major aim is to train node2vec embeddings on this graph for each ngram)
        we keep the construction of expanded graph consistent with the undirected graph
        :param minimum_number_of_langs: the minimum number of langs for an edge to be considered
        :return:
        """
        expanded_graph = nx.Graph()
        expanded_graph.add_nodes_from(self.concept_net.nodes, lang='eng')  # the concepts are all English
        # the original graph is a directed graph

        nodes_to_be_removed = [node for node in self.concept_net
                               if len(self.concept_net.nodes[node]['lang']) < minimum_number_of_langs]
        expanded_graph.remove_nodes_from(nodes_to_be_removed)

        # adding nodes and edges to the expanded graph
        for edge in self.concept_net.edges:
            source, end = edge[0], edge[1]

            # only consider the edge in the original directed graph whose
            # source and end are both preserved after the removal
            if expanded_graph.has_node(source) and expanded_graph.has_node(end):
                pass
            else:
                continue

            # see if this edge should be considered given current criterion of minimum number of languages
            if not self.is_considered(source, end, minimum_number_of_langs=minimum_number_of_langs):
                continue

            # for each edge, we will add multiple new edges (source node to all ngrams, and all ngrams to target nodes)
            # create target-language ngrams nodes
            # based on the fp stats and bp stats
            for lang, ngrams in self.concept_net.edges[source, end]['realizations'].items():
                for ngram in ngrams:
                    # networks will automatically add new nodes when adding edges
                    # adding weight to the edge

                    # always update the weight of an edge if it is larger than the previous weight

                    # based on fp (source node -> target ngram node)
                    if expanded_graph.has_edge(source, f"{lang}:{ngram}"):
                        # print(source, f"{lang}:{ngram}", expanded_graph.edges[source, f"{lang}:{ngram}"])
                        old_weight = expanded_graph.edges[source, f"{lang}:{ngram}"]['weight']
                        new_weight = self.concept_net.edges[source, end]['stats_fp'][lang][ngram]
                        # always use larger weight
                        if eval(old_weight) < eval(new_weight):
                            expanded_graph.edges[source, f"{lang}:{ngram}"]['weight'] = new_weight
                    else:
                        expanded_graph.add_edge(source, f"{lang}:{ngram}",
                                                weight=self.concept_net.edges[source, end]['stats_fp'][lang][ngram])

                    # based on bp (target ngram node -> end node)
                    if expanded_graph.has_edge(end,  f"{lang}:{ngram}"):
                        old_weight = expanded_graph.edges[source, f"{lang}:{ngram}"]['weight']  # maybe from FP or BP
                        new_weight = self.concept_net.edges[source, end]['stats_bp'][lang][end]
                        if eval(old_weight) < eval(new_weight):
                            expanded_graph.edges[end, f"{lang}:{ngram}"]['weight'] = new_weight
                    else:
                        expanded_graph.add_edge(end, f"{lang}:{ngram}",
                                                weight=self.concept_net.edges[source, end]['stats_bp'][lang][end])
                    expanded_graph.nodes[f"{lang}:{ngram}"]['lang'] = lang

        # remove the nodes whose degree is zero
        zero_degree_nodes = [n for n, v in dict(expanded_graph.degree()).items() if v == 0]
        expanded_graph.remove_nodes_from(zero_degree_nodes)

        self.expanded_graph = expanded_graph
        return expanded_graph

    def create_subgraph_of_a_concept(self, concept, depth=5, minimum_number_of_langs=1):
        """
        :param concept: the concept (as centered) of the subgraph
        :param depth: the maximum number of neighbors to the centered concept
        :param minimum_number_of_langs: the minimum languages for each node/edge to be included in the subgraph
        :return: subgraph: a subgraph of the original graph self.concept_net when looking at a given concept
        """
        # if depth > 5:
        #     print('Depth larger than 5 is not supported')
        #     raise ValueError

        # this should be based on the undirected graph
        # maybe here define my own searches, which can be more efficient and faster
        assert self.undirected_graph is not None

        if concept not in self.added_concepts:
            raise ValueError('The concept is not in the concept network!')

        selected_nodes = list(nx.bfs_tree(self.undirected_graph, source=concept, depth_limit=depth))
        selected_nodes = [node for node in selected_nodes
                          if len(self.undirected_graph.nodes[node]['lang']) >= minimum_number_of_langs]
        subgraph = self.undirected_graph.subgraph(selected_nodes).copy()

        # remove edges that only very few languages have the associations
        edges_to_remove = []
        for edge in subgraph.edges:
            if len(subgraph.edges[edge[0], edge[1]]['lang']) < minimum_number_of_langs:
                edges_to_remove.append(edge)
        subgraph.remove_edges_from(edges_to_remove)

        # remove nodes that are not reachable from the concept
        nodes_to_remove = []
        for node in subgraph.nodes:
            if nx.has_path(subgraph, concept, node):
                pass
            else:
                nodes_to_remove.append(node)
        subgraph.remove_nodes_from(nodes_to_remove)

        selected_nodes = list(nx.bfs_tree(subgraph, source=concept, depth_limit=depth))
        subgraph = subgraph.subgraph(selected_nodes).copy()

        return subgraph

    def create_subgraph_of_a_language(self, language):
        assert self.undirected_graph is not None
        # this should also be based on undirected graph
        # get the nodes that are reachable by language
        selected_nodes = [node for node in self.undirected_graph.nodes
                          if language in self.undirected_graph.nodes[node]['lang']]
        subgraph = self.undirected_graph.subgraph(selected_nodes).copy()

        # remove edges that are not from this language
        edges_to_remove = []
        for edge in subgraph.edges:
            if language not in subgraph.edges[edge[0], edge[1]]['lang']:
                edges_to_remove.append(edge)
        subgraph.remove_edges_from(edges_to_remove)

        return subgraph

    # write a function to create a subgraph for a language family
    def create_subgraph_of_a_language_family(self, language_family_code, threshold,
                                             min_lang=5, max_lang=20, family_path=None):
        """
        :param language_family_code: a language family code
        :param threshold: a float value between [0, 1] control the minimum numebr of languages to be considered
        :return:
        """
        assert self.undirected_graph is not None
        # we only consider the following language families where each has at least 50 languages
        assert language_family_code in ['sino1245', 'otom1299', 'nucl1709', 'indo1319', 'atla1278', 'aust1307']
        import pickle5

        # load language family data
        families = {}
        if family_path is None:
            with open('./iso2family.pickle', 'rb') as handle:
                iso2family = pickle5.load(handle)
        else:
            with open(family_path, 'rb') as handle:
                iso2family = pickle5.load(handle)
        for key, item in iso2family.items():
            if item[0] not in families:
                families[item[0]] = [key]
            else:
                families[item[0]].append(key)

        subgraph = self.undirected_graph.subgraph(self.undirected_graph.nodes).copy()

        # filtering edges
        edges_to_remove = []
        for edge in subgraph.edges:
            lang_count = 0
            for lang in subgraph.edges[edge[0], edge[1]]['lang']:
                if lang in families[language_family_code]:
                    lang_count += 1
            if lang_count < max(min_lang, min(threshold * len(families[language_family_code]), max_lang)):
                edges_to_remove.append(edge)
        subgraph.remove_edges_from(edges_to_remove)

        # for ARI, the number of nodes must be the same, so we will keep the nodes with zero degree

        # zero_degree_nodes = [n for n, v in dict(subgraph.degree()).items() if v == 0]
        # subgraph.remove_nodes_from(zero_degree_nodes)

        return subgraph

    def create_subgraph_of_an_area(self, area_code, threshold, min_lang=5, max_lang=20, area_path=None):
        assert self.undirected_graph is not None

        # we only consider the following areas where each has at least 150 languages
        assert area_code in ['South America', 'North America', 'Eurasia', 'Africa', 'Papunesia']
        # this should also be based on undirected graph
        # get the nodes that are reachable by language
        import pickle5

        # load language area data
        if area_path is None:
            with open('./iso2area.pickle', 'rb') as handle:
                iso2area = pickle5.load(handle)
        else:
            with open(area_path, 'rb') as handle:
                iso2area = pickle5.load(handle)
        areas = {}
        for key, item in iso2area.items():
            if item not in areas:
                areas[item] = [key]
            else:
                areas[item].append(key)

        subgraph = self.undirected_graph.subgraph(self.undirected_graph.nodes).copy()
        # filtering edges
        edges_to_remove = []
        for edge in subgraph.edges:
            lang_count = 0
            for lang in subgraph.edges[edge[0], edge[1]]['lang']:
                if lang in areas[area_code]:
                    lang_count += 1
            if lang_count < min(min_lang, max(threshold * len(areas[area_code]), max_lang)):
                edges_to_remove.append(edge)
        subgraph.remove_edges_from(edges_to_remove)

        # for ARI, the number of nodes must be the same, so we will keep the nodes with zero degree

        # zero_degree_nodes = [n for n, v in dict(subgraph.degree()).items() if v == 0]
        # subgraph.remove_nodes_from(zero_degree_nodes)

        return subgraph

    def store_net(self, path='./stored_networks'):
        """
        the path to store the base directed graph
        :param path: the path to store
        :return:
        """
        if not os.path.exists(path):
            os.makedirs(path)

        store_name = f"{path}/directed_network.pickle" if not self.use_updated \
            else f"{path}/directed_network_updated.pickle"

        try:
            with open(store_name, 'wb') as handle:
                pickle.dump(self.concept_net, handle)
            print(f"The directed net has been stored in {store_name}!")
        except:
            return False
        else:
            return True

    def load_net(self, path='./stored_networks'):
        """
        the path to load the base directed graph
        :param path: the path to store
        :return:
        """
        store_name = f"{path}/directed_network.pickle" if not self.use_updated \
            else f"{path}/directed_network_updated.pickle"
        try:
            print(f"Loading directed net from {store_name} ...")
            with open(store_name, 'rb') as handle:
                directed_net = pickle.load(handle)
        except:
            raise ValueError(f"Cannot load {store_name}!")
        else:
            self.concept_net = directed_net

    def store_vocab(self, path='./stored_vocab', store_name=None):
        assert self.expanded_graph is not None

        if not os.path.exists(path):
            os.makedirs(path)

        # parse the selected nodes
        vocabs = defaultdict(lambda: set())
        for node in self.expanded_graph.nodes:
            if len(node.split(':')) == 1:
                # this is the english source concept npde
                vocabs['eng'].add(node)
            elif len(node.split(':')) == 2:
                # we should move skip the vocabulary from matched 'eng'
                # we only use english concepts
                if node.split(':')[0] == 'eng':
                    continue
                vocabs[node.split(':')[0]].add(node.split(':')[1])

        if store_name is None:
            store_name = f"{path}/vocab.pickle" if not self.use_updated else f"{path}/vocab_updated.pickle"
        else:
            store_name = f"{path}/{store_name}.pickle"
        try:
            with open(store_name, 'wb') as handle:
                pickle.dump(dict(vocabs), handle)
        except:
            return False
        else:
            print(f"The vocabularies net has been stored in {store_name}!")


# test store net
# considered_lang = ['deu', 'rus', 'zho', 'arb']
# considered_lang = 'all'
# net = ConceptNetwork(involved_lang=considered_lang, load_directed_graph_from_path=False, use_updated=True)
# print(net.concept_net.number_of_nodes())
# print(net.concept_net.number_of_edges())
# net.store_net()

# test load net
# considered_lang = ['deu', 'rus', 'zho', 'arb']

# considered_lang = 'all'
# net = ConceptNetwork(involved_lang=considered_lang, load_directed_graph_from_path=True, use_updated=True)
# print(net.concept_net.number_of_nodes())
# print(net.concept_net.number_of_edges())
#
# expanded_net = net.expand_graph(minimum_number_of_langs=50)
# print(expanded_net.number_of_nodes())
# print(expanded_net.number_of_edges())

# net.store_vocab()