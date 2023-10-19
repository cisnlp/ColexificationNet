import argparse
import time
import pickle
import os
from processing_concepts_eng import is_new_testament

"""
This script is to process the parallel data, given the source language
Note that in case of English, the processing is different
"""

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


# read concepts and their verses
def get_parser():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(description="concept alignment")

    # main parameters
    # the source language should always be English
    parser.add_argument("--src_lang", type=str, help="source languages", default='eng')
    parser.add_argument('--required_path', type=str, help='the path that stores required files by preprocessing para',
                        default='/mounts/data/proj/yihong/newhome/ConceptNetwork/required_files')
    parser.add_argument("--ignore_case", type=bool_flag, help="Whether ignore the uppercase when comparing the strings",
                        default=True)
    parser.add_argument('--updated_ngrams', type=bool_flag, help='whether the updated ngrams target words',
                        default=True)
    return parser


def find_concept(concept, content, ignore_case=True):
    """
    :param concept: a string that represents the focal concept of interest (multiple strings split by |||)
    :param content: a dictionary, key is the verse_ID and the ngrams (set) the verse contains
    :param ignore_case: a boolean indicating whether ignores the case
    :return: concept_IDs (a set containing all the Ids of the verses that contain the concept)
    """
    concept = concept.lower() if ignore_case else concept
    concept_split = concept.split('|||')
    concept_IDs = set()  # 'set' object does not support indexing

    for verse_ID, verse_content in content.items():
        for c in concept_split:
            if c in verse_content:
                concept_IDs.add(verse_ID)
                break
    return concept_IDs


def read_concepts(concept_path):
    concept_list = []
    with open(concept_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if len(line.strip().split(',')) > 1:
                concept_list.append(line.strip().split(',')[0])
            else:
                concept_list.append(line.strip())
    return concept_list


def main(params):
    start = time.time()
    src_lang = params.src_lang
    store_dir = params.required_path
    ignore_case = params.ignore_case
    updated_ngrams = params.updated_ngrams

    # if we restrict the ngrams to be within white-spaced tokenized things, we should use different directories
    ngrams_dir = store_dir + '/ngrams' if not updated_ngrams else store_dir + '/ngrams/updated'

    parallel_dir = f"{store_dir}/parallel_data/{src_lang}" if not updated_ngrams \
        else f"{store_dir}/parallel_data/{src_lang}/updated"

    # make sure the paths exit
    assert os.path.exists(ngrams_dir)
    assert os.path.exists(parallel_dir)

    if src_lang == 'eng':
        # for english, we should propose concepts differently (generate our own concept list)
        # running processing concepts eng script
        os.system(f"/mounts/data/proj/yihong/newhome/ENTER/envs/concept-net/bin/python "
                  f"./processing_concepts_eng.py --updated_ngrams {updated_ngrams}")
        # load the eng language ngram data
        with open(f"{parallel_dir}/eng-metadata_spacy.pickle", 'rb') as handle:
            src_lang_ngrams_data = pickle.load(handle)
            src_lang_ngrams_freq, src_lang_ngrams = src_lang_ngrams_data
    else:
        # load the source language ngram data
        with open(f"{ngrams_dir}/{src_lang}.pickle", 'rb') as handle:
            src_lang_ngrams_data = pickle.load(handle)
            src_lang_ngrams_freq, src_lang_ngrams = src_lang_ngrams_data

    concept_path = f"{parallel_dir}/concept_list"
    assert os.path.exists(concept_path)

    concept_list = read_concepts(concept_path)
    concept_verse_IDs = dict()

    for i, concept in enumerate(concept_list):
        concept_verse_IDs[concept] = find_concept(concept, src_lang_ngrams, ignore_case)

        if src_lang == 'eng':
            if i % 10 == 0:
                print(f"Processing concept {concept} ...")
                print(f"{concept}: {len(concept_verse_IDs[concept])}")
                print(f"{len([v for v in concept_verse_IDs[concept] if is_new_testament(v)])}")
                print()
        else:
            print(f"Processing concept {concept} ...")
            print(f"{concept}: {len(concept_verse_IDs[concept])}")
            print(f"{len([v for v in concept_verse_IDs[concept] if is_new_testament(v)])}")
            print()

        # the first part (the global freq of the concept) should be equal to the second part (verseIDs contains concept)
        if src_lang == 'eng':
            assert src_lang_ngrams_freq[concept] == len(concept_verse_IDs[concept])

    print("Storing to concept_verse_IDs.pickle...")
    with open(parallel_dir + '/concept_verse_IDs.pickle', 'wb') as handle:
        pickle.dump(concept_verse_IDs, handle)

    end = time.time()
    print(f"Runtime: {end - start} s")


if __name__ == '__main__':
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    main(params)