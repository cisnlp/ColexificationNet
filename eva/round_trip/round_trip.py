import pickle
from gensim.models import KeyedVectors, Word2Vec
import random

# function to load embedding
def load_embedding(emb_name):
    if emb_name == 'clique_word' or emb_name == 'nt_word':
        embedding_path = f"/mounts/data/proj/yihong/newhome/ConceptNetwork/eva/baseline_vectors" + \
                         f"/word_embeddings_{emb_name}.kv"
        loaded_n2v = KeyedVectors.load(embedding_path)
    elif emb_name == 'sentence_id':
        epochs = 50
        emb_dim = 200
        word_vec = Word2Vec.load(f"/mounts/data/proj/yihong/newhome/ConceptNetwork/eva/baseline_vectors"
                                 f"/word2vec_{epochs}_{emb_dim}.model")
        loaded_n2v = word_vec.wv
    elif emb_name == 'eflomal':
        epochs = 10
        emb_dim = 200
        loaded_n2v = KeyedVectors.load(f"/mounts/data/proj/yihong/newhome/ConceptNetwork/network_related"
                                       f"/eflomal_vectors_{emb_dim}_{epochs}.wv")
    elif emb_name == 'ExpandedEmb':
        emb_dim = 200
        num_epochs = 10
        number_of_languages = 50
        embedding_path = f"/mounts/data/proj/yihong/newhome/ConceptNetwork/network_related/" + \
                         f"expandednet_vectors_minlang_{number_of_languages}_{emb_dim}_{num_epochs}_updated.wv"

        loaded_n2v = KeyedVectors.load(embedding_path)
    return loaded_n2v

vocabs = dict()
for emb_name in ['sentence_id', 'clique_word', 'nt_word', 'eflomal', 'ExpandedEmb']:
    if emb_name == 'eflomal':
        vocab_dir = f"/mounts/data/proj/yihong/newhome/ConceptNetwork/network_related/stored_vocab/eflomal_vocab.pickle"
    elif emb_name == 'ExpandedEmb':
        number_of_languages = 50
        vocab_dir = f"/mounts/data/proj/yihong/newhome/ConceptNetwork/network_related/" \
                f"stored_vocab/vocab_min_{number_of_languages}_updated.pickle"
    else:
        vocab_dir = f"/mounts/data/proj/yihong/newhome/ConceptNetwork/eva/" + \
                f"baseline_vectors/stored_vocab/vocab_{emb_name}.pickle"

    with open(f"{vocab_dir}", 'rb') as handle:
        vocabs[emb_name] = pickle.load(handle)

            
n2v_dict = {}
candidate_common_set = {}
for i, emb_name in enumerate(['sentence_id', 'clique_word', 'nt_word', 'eflomal', 'ExpandedEmb']):
    n2v_dict[emb_name] = load_embedding(emb_name)
    if i == 0:
        candidate_common_set = vocabs[emb_name]['eng']
        continue
    new_set = vocabs[emb_name]['eng']
    if emb_name == 'ExpandedEmb':
        new_set = {word.replace('$', '') for word in vocabs[emb_name]['eng'] if word[0]=='$' and word[-1]=='$'}
    candidate_common_set = candidate_common_set.intersection(new_set)

print(len(candidate_common_set))
random.seed(114514)
selected_index = random.sample(range(0, len(candidate_common_set)), len(candidate_common_set))
candidate_common_list = list(candidate_common_set)
candidates = [candidate_common_list[i] for i in selected_index]

# function of round-trip translation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def round_trip_translation(emb_name, embeddings, start_words, sequence_of_langs, vocabs, start_lang='eng', top_k=1):
    assert len(sequence_of_langs) >= 1
    assert top_k >= 1
#     start_lang_vocab = list(vocabs[start_lang])
    start_lang_vocab = start_words
    if emb_name == 'ExpandedEmb' and start_lang == 'eng':
        start_lang_vocab = ['$' + v + '$' for v in start_lang_vocab]
        start_lang_vocab_embedding = np.array([embeddings[v] for v in start_lang_vocab])
    else:
        start_lang_vocab_embedding = np.array([embeddings[f"{start_lang}:{v}"] for v in start_lang_vocab])
    percentage = 0
    for start_word in start_words:
        if emb_name == 'ExpandedEmb' and start_lang == 'eng':
            start_word = '$' + start_word + '$'
            start_embedding = embeddings[start_word]
        else:
            start_embedding = embeddings[f"{start_lang}:{start_word}"]
        current_embedding = start_embedding
        emb_dim = current_embedding.shape[0]
        for lang in sequence_of_langs:
            current_vocab = list(vocabs[lang])
            vocab_embedding = np.array([embeddings[f"{lang}:{v}"] for v in current_vocab])
            cos_sim = cosine_similarity(current_embedding.reshape(-1, emb_dim), vocab_embedding).reshape(-1)
            idx = cos_sim.argsort()[::-1][0:top_k]
            current_embedding = embeddings[f"{lang}:{current_vocab[idx[0]]}"]
#             print(current_vocab[idx[0]])
        # finally returning to the start language
        cos_sim = cosine_similarity(current_embedding.reshape(-1, emb_dim), start_lang_vocab_embedding).reshape(-1)
        idx = cos_sim.argsort()[::-1][0:top_k]
        final_word = [start_lang_vocab[idx[i]] for i in range(0, min(len(idx), top_k))]
#         print(start_word, final_word)
        if start_word in final_word:
            percentage += 1
    return percentage/len(start_words)


import pandas as pd
# obtain all distinct language's ISO 639-3 codes
def get_langs():
    csv_path = '/mounts/Users/student/yihong/Documents/concept_align'
    pbc_info = pd.read_csv(f"{csv_path}/pbc_table.csv", converters={"language_code": str})
    langs = pbc_info['language_code'].values
    langs = sorted(list(set(langs)))
    return langs

total_langs = get_langs()
total_langs.remove('eng')

for i in range(10):
    # select random langauges
    while True:
        selectd_langs = [total_langs[i] for i in random.sample(range(0, len(total_langs)), 3)]
        flag = True
        for emb_name in ['sentence_id', 'clique_word', 'nt_word', 'eflomal', 'ExpandedEmb']:
            for lang in selectd_langs:
                if lang not in vocabs[emb_name]:
                    flag = False
                    break
        if flag:
            break
        
    print(selectd_langs)
    for top_k in [1, 5, 10]:
        for emb_name in ['sentence_id', 'clique_word', 'nt_word', 'eflomal', 'ExpandedEmb']:
            result = round_trip_translation(emb_name, n2v_dict[emb_name], start_words=candidates, 
                                            sequence_of_langs=selectd_langs, vocabs=vocabs[emb_name], top_k=top_k)
            print(f"emb_name-{emb_name}, top-{top_k}, {round(result, 3)}")
            
    """
    # export PYTHONIOENCODING=utf8; nohup python -u round_trip.py > ./result.txt 2>&1 &
    # server: pi pid: 21918
    #
    """
