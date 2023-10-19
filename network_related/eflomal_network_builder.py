import networkx as nx
import pickle
import os
from collections import defaultdict


def split_aligns(align: str):
    if len(align) == 0:
        return dict()
    align = align.split(' ')
    try:
        align = [(int(a.split('-')[0]), int(a.split('-')[1])) for a in align]
    except:
        return dict()
    else:
        align_dict = dict(align)
        return align_dict


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


class EflomalAlignmentNetwork:
    def __init__(self, load_graph_from_path=False, load_undirected_graph_path='',
                 alignment_path='/mounts/data/proj/yihong/newhome/AssociationNetwork/elfomal_related/'
                                'alignments/alignments.pickle',
                 parallel_content_path='/mounts/data/proj/yihong/newhome/AssociationNetwork/required_files/'
                                       'para_contents.pickle'):
        """
        :param load_undirected_graph_path: the path to load a stored directed graph
        :param alignment_path: the path storing the computed eflomal results
        """
        # for eflomal, we directly build an undirected graph
        # the nodes are English tokens AND tokens from any other languages
        # the edges show the two tokens (one from English and one from other langauge) are aligned
        self.concept_net = nx.Graph()

        # load directed graph from disk
        if load_graph_from_path:
            if len(load_undirected_graph_path) == 0:
                self.load_net()
            else:
                self.load_net(load_undirected_graph_path)
        else:
            # load parallel data
            with open(f"{parallel_content_path}", 'rb') as handle:
                para_contents = pickle.load(handle)

            # load alignments
            with open(f"{alignment_path}", 'rb') as handle:
                alignments = pickle.load(handle)

            # build a graph given the alignments
            self.added_concepts = set()
            self.added_edges = set()
            for lang_pair, _ in alignments.items():
                # lang_pair = 'eng-grc'
                print(lang_pair)
                src_lang, tgt_lang = lang_pair.split('-')
                # print(data)
                parallel_verse_ids = list(alignments[lang_pair].keys())
                src_verse = para_contents[f"eng-eng"]  # the source language is always english
                tgt_verse = para_contents[lang_pair]
                src_sentences = [src_verse[v].lower() for v in parallel_verse_ids]
                tgt_sentences = [tgt_verse[v].lower() for v in parallel_verse_ids]
                current_alignments = [alignments[lang_pair][v] for v in parallel_verse_ids]

                assert len(src_sentences) == len(tgt_sentences)
                assert len(tgt_sentences) == len(current_alignments)

                for i in range(len(src_sentences)):
                    src_token_list = src_sentences[i].split(' ')
                    tgt_token_list = tgt_sentences[i].split(' ')
                    # a dictionary key: src_token_id, value tgt_token_id
                    align_dict = split_aligns(current_alignments[i])

                    for src_id, tgt_id in align_dict.items():

                        # ignore some problems in alignment
                        if tgt_id >= len(tgt_token_list):
                            continue

                        src_token = f"{src_lang}:{src_token_list[src_id]}"
                        tgt_token = f"{tgt_lang}:{tgt_token_list[tgt_id]}"

                        # add nodes and edges based on current alignment
                        if src_token not in self.added_concepts:
                            self.added_concepts.add(src_token)
                        if tgt_token not in self.added_concepts:
                            self.added_concepts.add(tgt_token)
                        if f"{src_token}-{tgt_token}" not in self.added_edges:
                            self.added_edges.add(f"{src_token}-{tgt_token}")
                            self.concept_net.add_edge(src_token, tgt_token, count=1)
                        else:
                            self.concept_net.edges[src_token, tgt_token]['count'] += 1
            edges_to_remove = []
            for edge in self.concept_net.edges:
                # remove edges that are not good
                if self.concept_net.edges[edge]['count'] <= 1:
                    edges_to_remove.append(edge)
            self.concept_net.remove_edges_from(edges_to_remove)

            # we would also like to remove the nodes have zero degree
            # remove the nodes whose degree is zero
            zero_degree_nodes = [n for n, v in dict(self.concept_net.degree()).items() if v == 0]
            self.concept_net.remove_nodes_from(zero_degree_nodes)
            print(self.concept_net.number_of_nodes())
            print(self.concept_net.number_of_edges())

            """
            5499528
            18846094
            """
            # print(self.concept_net.edges(data=True))


    def store_net(self, path='./stored_networks'):
        """
        the path to store the base directed graph
        :param path: the path to store
        :return:
        """
        if not os.path.exists(path):
            os.makedirs(path)

        store_name = f"{path}/eflomal_undirected_network.pickle"

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
        store_name = f"{path}/eflomal_undirected_network.pickle"
        try:
            print(f"Loading directed net from {store_name} ...")
            with open(store_name, 'rb') as handle:
                directed_net = pickle.load(handle)
        except:
            raise ValueError(f"Cannot load {store_name}!")
        else:
            self.concept_net = directed_net

    def store_vocab(self, path='./stored_vocab'):

        if not os.path.exists(path):
            os.makedirs(path)

        # parse the selected nodes
        vocabs = defaultdict(lambda: set())
        for node in self.concept_net.nodes:
            lang = node[:3]
            token = node[4:]
            # print(lang, token)
            vocabs[lang].add(token)

        store_name = f"{path}/eflomal_vocab.pickle"
        try:
            with open(store_name, 'wb') as handle:
                pickle.dump(dict(vocabs), handle)
            print(f"The vocabularies net has been stored in {store_name}!")
        except:
            return False
        else:
            return True

# net = EflomalAlignmentNetwork(load_graph_from_path=False)
# net.store_net()
# net.store_vocab()