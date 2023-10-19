import pickle
from gensim.models import KeyedVectors
import random

# function to load embedding
def load_embedding(number_of_languages):
    emb_dim = 200
    num_epochs = 10
    embedding_path = f"/mounts/data/proj/yihong/newhome/ConceptNetwork/network_related/" + \
                     f"expandednet_vectors_minlang_{number_of_languages}_{emb_dim}_{num_epochs}_updated.wv"

    loaded_n2v = KeyedVectors.load(embedding_path)
    return loaded_n2v

vocabs = dict()
for number_of_languages in [1, 5, 10, 20, 50, 100]:
    vocab_dir = f"/mounts/data/proj/yihong/newhome/ConceptNetwork/network_related/" \
            f"stored_vocab/vocab_min_{number_of_languages}_updated.pickle"
    with open(f"{vocab_dir}", 'rb') as handle:
            vocabs[number_of_languages] = pickle.load(handle)

            
n2v_dict = {}
candidate_common_set = {}
for i, number_of_languages in enumerate([1, 5, 10, 20, 50, 100]):
    n2v_dict[number_of_languages] = load_embedding(number_of_languages)
    if i == 0:
        candidate_common_set =\
        {word for word in vocabs[number_of_languages]['eng'] if word in n2v_dict[number_of_languages]}
        continue
    new_set = {word for word in vocabs[number_of_languages]['eng'] if word in n2v_dict[number_of_languages]}
    candidate_common_set = candidate_common_set.intersection(new_set)


random.seed(114514)
selected_index = random.sample(range(0, len(candidate_common_set)), len(candidate_common_set))
candidate_common_list = list(candidate_common_set)
candidates = [candidate_common_list[i] for i in selected_index]

# function of round-trip translation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def round_trip_translation(embeddings, start_words, sequence_of_langs, vocabs, start_lang='eng', top_k=1):
    assert len(sequence_of_langs) >= 1
    assert top_k >= 1
#     start_lang_vocab = list(vocabs[start_lang])
    start_lang_vocab = start_words
    start_lang_vocab_embedding = np.array([embeddings[v] for v in start_lang_vocab])
    percentage = 0
    for start_word in start_words:
        start_embedding = embeddings[start_word]
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
    selectd_langs = [total_langs[i] for i in random.sample(range(0, len(total_langs)), 3)]
    print(selectd_langs)
    for top_k in [1, 5, 10]:
        for number_of_languages in [1, 5, 10, 20, 50, 100]:
            result = round_trip_translation(n2v_dict[number_of_languages], start_words=candidates, 
                                            sequence_of_langs=selectd_langs, vocabs=vocabs[number_of_languages], 
                                            top_k=top_k)
            print(f"num_lang-{number_of_languages}, top-{top_k}, {round(result, 3)}")
            
    """
    # export PYTHONIOENCODING=utf8; nohup python -u round_trip.py > ./result.txt 2>&1 &
    # server: pi pid: 79967
    #
    """
