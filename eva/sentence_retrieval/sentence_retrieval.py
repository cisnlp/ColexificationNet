import os
import pandas as pd
import pickle


def get_langs():
    csv_path = '/mounts/Users/student/yihong/Documents/concept_align'
    pbc_info = pd.read_csv(f"{csv_path}/pbc_table.csv", converters={"language_code": str})
    langs = pbc_info['language_code'].values
    langs = sorted(list(set(langs)))
    return langs


def parse_file(file_path, verse_ngrams, contained_ngrams_set):
    """
    # process an individual file
    :param file_path: the path of the ids, each line is a verse ID
    :param verse_ngrams: a dictionary of the verse and its contained ngrams set
    :param contained_ngrams_set: a set of all possible ngrams in a language
    :return:
    """
    with open(file_path, 'r', encoding="utf-8") as f:
        pairs = []
        for line in f.readlines():
            verse_number = line.strip()
            representation = []
            if verse_number not in verse_ngrams:
                continue

            for ngram in verse_ngrams[verse_number]:
                # check if the ngram is in the vocabulary
                # each token in the representation should only occur once
                if ngram in contained_ngrams_set and ngram not in representation:
                    representation.append(ngram)
            pairs.append((verse_number, representation))
    return pairs


tgt_langs = get_langs()

updated_ngrams = True
number_of_languages = 100

test_ids_path = './test_ids.txt'

ngrams_dir = '/mounts/data/proj/yihong/newhome/ConceptNetwork/required_files/ngrams' if not updated_ngrams else \
    '/mounts/data/proj/yihong/newhome/ConceptNetwork/required_files/ngrams/updated'

if not updated_ngrams:
    vocab_dir = '/mounts/data/proj/yihong/newhome/ConceptNetwork/network_related/stored_vocab/vocab.pickle'
    processed_dataset_path = '/mounts/data/proj/yihong/newhome/ConceptNetwork/eva/sentence_retrieval'
else:
    vocab_dir = f"/mounts/data/proj/yihong/newhome/ConceptNetwork/network_related/" \
                f"stored_vocab/vocab_min_{number_of_languages}_updated.pickle"
    processed_dataset_path = f"/mounts/data/proj/yihong/newhome/ConceptNetwork/eva/" \
                             f"sentence_retrieval/updated/{number_of_languages}"

test_set_to_store = {}

with open(f"{vocab_dir}", 'rb') as handle:
    vocabs = pickle.load(handle)

for test_lang in tgt_langs:
    print(test_lang)
    if test_lang != 'eng':
        with open(f"{ngrams_dir}/{test_lang}.pickle", 'rb') as handle:
            _, test_lang_verse_ngrams = pickle.load(handle)
    else:
        eng_dir = f"/mounts/data/proj/yihong/newhome/ConceptNetwork/required_files/" \
                  f"parallel_data/eng/eng-metadata_spacy"
        if updated_ngrams:
            eng_dir = f"/mounts/data/proj/yihong/newhome/ConceptNetwork/required_files/" \
                      f"parallel_data/eng/updated/eng-metadata_spacy"

        with open(f"{eng_dir}.pickle", 'rb') as handle:
            _, test_lang_verse_ngrams = pickle.load(handle)

    test_lang_selected_ngrams = vocabs[test_lang]

    # process train files
    pairs = parse_file(file_path=test_ids_path, verse_ngrams=test_lang_verse_ngrams,
                       contained_ngrams_set=test_lang_selected_ngrams)

    print(f"Number of test set: {len(pairs)}")
    print()

    test_set_to_store[test_lang] = pairs


# storing test set
if os.path.exists(f"{processed_dataset_path}/test"):
    pass
else:
    os.makedirs(f"{processed_dataset_path}/test")
with open(f"{processed_dataset_path}/test/test.pickle", 'wb') as handle:
    pickle.dump(test_set_to_store, handle)
