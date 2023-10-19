import argparse
import time
import pickle
from collections import Counter
from processing_ngrams import get_langs
import os
import multiprocessing
import numpy as np

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
    parser.add_argument('--updated_ngrams', type=bool_flag, help='whether the updated ngrams target words',
                        default=True)
    return parser


def aggregate_counter(IDs, sents):
    results = Counter()
    for ID, sent in zip(IDs, sents):
        results += Counter(sent)
    return results


def main(params):
    start = time.time()
    src_lang = params.src_lang
    store_dir = params.required_path
    updated_ngrams = params.updated_ngrams

    # if we restrict the ngrams to be within white-spaced tokenized things, we should use different directories
    ngrams_dir = store_dir + '/ngrams' if not updated_ngrams else store_dir + '/ngrams/updated'

    parallel_dir = f"{store_dir}/parallel_data/{src_lang}" if not updated_ngrams \
        else f"{store_dir}/parallel_data/{src_lang}/updated"

    if os.path.exists(parallel_dir):
        pass
    else:
        os.makedirs(parallel_dir)

    print(f"Src language: {src_lang}")

    # load the ngrams data of the source language
    if src_lang == 'eng':
        eng_data_name = parallel_dir + "/eng-metadata_spacy.pickle"
        with open(eng_data_name, 'rb') as handle:
            src_lang_ngram_freq, src_lang_ngrams = pickle.load(handle)
    else:
        with open(f"{ngrams_dir}/{src_lang}.pickle", 'rb') as handle:
            src_lang_ngram_freq, src_lang_ngrams = pickle.load(handle)

    # obtain the parallel data: 1. parallel IDs, 2. updated filtered_tgt_lang_ngrams according to the parallel IDs
    parallel_ids = dict()
    tgt_lang_ngrams_freq_to_subtract = dict()
    src_lang_ngrams_freq_to_subtract = dict()
    for iteration, tgt_lang in enumerate(get_langs()):
        print(tgt_lang)
        with open(f"{ngrams_dir}/{tgt_lang}.pickle", 'rb') as handle:
            tgt_lang_ngrams_data = pickle.load(handle)
            tgt_lang_ngram_freq, tgt_lang_ngrams = tgt_lang_ngrams_data

        # obtain the parallel IDs and the IDs not included
        src_verseIDs = list(sorted(src_lang_ngrams.keys()))
        tgt_verseIDs = list(sorted(tgt_lang_ngrams.keys()))

        common_verseIDs = list(set(src_verseIDs).intersection(set(tgt_verseIDs)))  # verse in both src and tgt language
        removed_verseIDs_from_tgt = list(set(tgt_verseIDs).difference(set(src_verseIDs)))  # verse only in tgt language
        removed_verseIDs_from_src = list(set(src_verseIDs).difference(set(tgt_verseIDs)))  # verse only in src language
        # using the removed_verseIDs to make a record of ngrams to be updated for each language
        tgt_lang_ngrams_freq_to_remove = Counter()
        src_lang_ngrams_freq_to_remove = Counter()


        # parallelization
        cores_num = max(16, multiprocessing.cpu_count() // 2)
        results = []
        pool = multiprocessing.Pool(cores_num)

        # for tgt side
        IDs_chunked_tgt = np.array_split(removed_verseIDs_from_tgt, cores_num)
        for index in range(cores_num):
            IDs_parts_tgt = IDs_chunked_tgt[index].tolist()
            results.append(pool.apply_async(aggregate_counter,
                                            (IDs_parts_tgt, [tgt_lang_ngrams[ID] for ID in IDs_parts_tgt])))
        pool.close()
        pool.join()
        for r in results:
            chucked = r.get()
            tgt_lang_ngrams_freq_to_remove += chucked

        pool = multiprocessing.Pool(cores_num)
        # for src side
        IDs_chunked_src = np.array_split(removed_verseIDs_from_src, cores_num)
        for index in range(cores_num):
            IDs_parts_src = IDs_chunked_src[index].tolist()
            results.append(pool.apply_async(aggregate_counter,
                                            (IDs_parts_src, [src_lang_ngrams[ID] for ID in IDs_parts_src])))
        pool.close()
        pool.join()
        for r in results:
            chucked = r.get()
            src_lang_ngrams_freq_to_remove += chucked

        parallel_ids[tgt_lang] = common_verseIDs
        tgt_lang_ngrams_freq_to_subtract[tgt_lang] = tgt_lang_ngrams_freq_to_remove
        src_lang_ngrams_freq_to_subtract[tgt_lang] = src_lang_ngrams_freq_to_remove

        # print(len(tgt_lang_ngram_freq), len(tgt_lang_ngrams_freq_to_remove))
        # print(len(Counter(tgt_lang_ngram_freq) - tgt_lang_ngrams_freq_to_remove))
        # print(len(src_lang_ngram_freq), len(src_lang_ngrams_freq_to_remove))
        # print(len(Counter(src_lang_ngram_freq) - tgt_lang_ngrams_freq_to_remove))

    with open(f"{parallel_dir}/parallel_ids.pickle", 'wb') as handle:
        pickle.dump(parallel_ids, handle)

    with open(f"{parallel_dir}/tgt_lang_ngrams_freq_to_subtract.pickle", 'wb') as handle:
        pickle.dump(tgt_lang_ngrams_freq_to_subtract, handle)

    with open(f"{parallel_dir}/src_lang_ngrams_freq_to_subtract.pickle", 'wb') as handle:
        pickle.dump(src_lang_ngrams_freq_to_subtract, handle)

    end = time.time()
    print(f"Runtime: {end - start} s")


if __name__ == '__main__':
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    main(params)

    """
    cmn:
    
    # export PYTHONIOENCODING=utf8; nohup python -u processing_parallel.py > ./parallel_processing_log_cmn.txt 2>&1 &
    # server: delta pid: 10488
    
    
    eng:
    
    # export PYTHONIOENCODING=utf8; nohup python -u processing_parallel.py > ./parallel_processing_log_eng.txt 2>&1 &
    # server: epsilon3 pid: 1052665
    done
    
    
    eng: 
    # export PYTHONIOENCODING=utf8; nohup python -u processing_parallel.py > ./parallel_processing_log_eng_updated.txt 2>&1 &
    # server: pi pid: 66522
    # running
    """