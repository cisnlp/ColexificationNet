import os
from collections import defaultdict
import pickle
import argparse
import pandas as pd


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}


# obtain all distinct language's ISO 639-3 codes
def get_langs():
    csv_path = '/mounts/Users/student/yihong/Documents/concept_align'
    pbc_info = pd.read_csv(f"{csv_path}/pbc_table.csv", converters={"language_code": str})
    langs = pbc_info['language_code'].values
    langs = sorted(list(set(langs)))
    return langs


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


def get_parser():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(description="concept alignment")

    # main parameters
    parser.add_argument('--required_path', type=str, help='the path that stores required files by preprocessing para',
                        default='/mounts/data/proj/yihong/newhome/ConceptNetwork/required_files')
    parser.add_argument("--ignore_case", type=bool_flag, help="Whether ignore the uppercase when comparing the strings",
                        default=True)
    parser.add_argument("--ngram_min_len", type=int, help='minimum length of the ngram', default=1)
    parser.add_argument("--ngram_max_len", type=int, help='maximum length of the ngram', default=8)
    parser.add_argument("--min_threshold", type=int, help='minimum frequency for a ngram', default=2)
    parser.add_argument('--updated_ngrams', type=bool_flag, help='whether the updated ngrams target words',
                        default=True)
    parser.add_argument('--number_of_languages', type=int, default=100)
    return parser


def obtain_datasets(dataset_path='/mounts/data/proj/ayyoob/Chunlan/lrs_train_dev_test2'):
    files = os.listdir(dataset_path)

    # classifying files
    trains = defaultdict(lambda: defaultdict())  # for different number of sets
    devs = {}
    tests = {}
    for filename in files:
        # TEST or VALID
        flist = filename.split('_')
        if len(flist) == 2:
            if flist[1].split('.')[0] == 'dev':
                devs[flist[0]] = filename
            elif flist[1].split('.')[0] == 'test':
                tests[flist[0]] = filename
            else:
                print(filename)
                raise ValueError('this file name has problem')
        elif len(flist) == 3:
            trains[flist[0]][flist[1]] = filename
        else:
            print(filename)
            raise ValueError('this file name has problem')

    assert len(devs) == len(tests)
    # print(trains)
    for num, train_files in trains.items():
        # print(f"{num}: {len(train_files)}")
        # print(len(devs))
        # print()
        if len(train_files) != len(devs):
            print(f"{num} has problem.")

    return trains, devs, tests


def parse_file(file_path, verse_ngrams, contained_ngrams_set):
    """
    # process an individual file
    :param file_path: the path of the file wants to be processed
    :param verse_ngrams: a dictionary of the verse and its contained ngrams set
    :param contained_ngrams_set: a set of all possible ngrams in a language
    :return:
    """
    with open(file_path, 'r', encoding="utf-8") as f:
        pairs = []
        for line in f.readlines():
            parts = line.strip().split('\t')
            verse_number = parts[0]
            verse_label = parts[1]
            representation = []
            if verse_number not in verse_ngrams:
                continue
            for ngram in verse_ngrams[verse_number]:
                if ngram in contained_ngrams_set and ngram not in representation:
                    representation.append(ngram)
            pairs.append((verse_number, representation, verse_label))
    return pairs


def main(params):
    updated_ngrams = params.updated_ngrams
    dataset_path = '/mounts/data/proj/ayyoob/Chunlan/lrs_train_dev_test2/'
    number_of_languages = params.number_of_languages

    ngrams_dir = '/mounts/data/proj/yihong/newhome/ConceptNetwork/required_files/ngrams' if not updated_ngrams else \
        '/mounts/data/proj/yihong/newhome/ConceptNetwork/required_files/ngrams/updated'

    if not updated_ngrams:
        vocab_dir = '/mounts/data/proj/yihong/newhome/ConceptNetwork/network_related/stored_vocab/vocab.pickle'
        processed_dataset_path = '/mounts/data/proj/yihong/newhome/ConceptNetwork/eva/sentence_classification'
    else:
        vocab_dir = f"/mounts/data/proj/yihong/newhome/ConceptNetwork/network_related/" \
                    f"stored_vocab/vocab_min_{number_of_languages}_updated.pickle"
        processed_dataset_path = f"/mounts/data/proj/yihong/newhome/ConceptNetwork/eva/" \
                                 f"sentence_classification/updated/{number_of_languages}"

    # with open(f"{processed_dataset_path}/train/860/train.pickle", 'rb') as handle:
    #     train_set_to_store = pickle.load(handle)
    #
    # with open(f"{processed_dataset_path}/valid/valid.pickle", 'rb') as handle:
    #     valid_set_to_store = pickle.load(handle)
    #
    # with open(f"{processed_dataset_path}/test/test.pickle", 'rb') as handle:
    #     test_set_to_store = pickle.load(handle)

    train_set_to_store = dict()
    valid_set_to_store = dict()
    test_set_to_store = dict()

    train_set, valid_set, test_set = obtain_datasets(dataset_path=dataset_path)
    tgt_langs = get_langs()
    # tgt_langs = ['eng']

    for test_lang in tgt_langs:

        # if the language is not in the dataset
        # if test_lang not in train_set['860'] or test_lang in train_set_to_store:
        #     continue
        if test_lang not in train_set['860']:
            continue

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

        with open(f"{vocab_dir}", 'rb') as handle:
            vocabs = pickle.load(handle)
        test_lang_selected_ngrams = vocabs[test_lang]

        # process train files
        pairs = parse_file(file_path='/mounts/data/proj/ayyoob/Chunlan/lrs_train_dev_test2/' + train_set['860'][test_lang],
                           verse_ngrams=test_lang_verse_ngrams, contained_ngrams_set=test_lang_selected_ngrams)
        print(f"Number of train set: {len(pairs)}")
        train_set_to_store[test_lang] = pairs

        # process valid files
        pairs = parse_file(file_path='/mounts/data/proj/ayyoob/Chunlan/lrs_train_dev_test2/' + valid_set[test_lang],
                           verse_ngrams=test_lang_verse_ngrams, contained_ngrams_set=test_lang_selected_ngrams)
        print(f"Number of valid set: {len(pairs)}")
        valid_set_to_store[test_lang] = pairs

        # process test files
        pairs = parse_file(file_path='/mounts/data/proj/ayyoob/Chunlan/lrs_train_dev_test2/' + test_set[test_lang],
                           verse_ngrams=test_lang_verse_ngrams, contained_ngrams_set=test_lang_selected_ngrams)
        print(f"Number of test set: {len(pairs)}")
        test_set_to_store[test_lang] = pairs
        print()

    # storing train set
    if os.path.exists(f"{processed_dataset_path}/train/860"):
        pass
    else:
        os.makedirs(f"{processed_dataset_path}/train/860")
    with open(f"{processed_dataset_path}/train/860/train.pickle", 'wb') as handle:
        pickle.dump(train_set_to_store, handle)

    # storing valid set
    if os.path.exists(f"{processed_dataset_path}/valid"):
        pass
    else:
        os.makedirs(f"{processed_dataset_path}/valid")
    with open(f"{processed_dataset_path}/valid/valid.pickle", 'wb') as handle:
        pickle.dump(valid_set_to_store, handle)

    # storing test set
    if os.path.exists(f"{processed_dataset_path}/test"):
        pass
    else:
        os.makedirs(f"{processed_dataset_path}/test")
    with open(f"{processed_dataset_path}/test/test.pickle", 'wb') as handle:
        pickle.dump(test_set_to_store, handle)


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    main(params)

    """
    # export PYTHONIOENCODING=utf8; nohup python -u sentence_classification.py > ./temp.txt 2>&1 &
    # server: pi pid: 31200
    #
    """

