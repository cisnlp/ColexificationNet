import numpy as np
import argparse
import time
import pickle
import pandas as pd
from collections import OrderedDict, Counter
import multiprocessing
import numba as nb
import math
from processing_concepts import read_concepts
import os


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


# read concepts and their verses
def get_parser():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(description="association finder")

    # main parameters
    # the source language should always be English
    parser.add_argument("--src_lang", type=str, help="source language", default='eng')
    parser.add_argument('--required_path', type=str, help='the path that stores required files by preprocessing para',
                        default='/mounts/data/proj/yihong/newhome/ConceptNetwork/required_files')
    parser.add_argument('--results_path', type=str, help='the path that stores required files by preprocessing para',
                        default='/mounts/data/proj/yihong/newhome/ConceptNetwork/results')
    parser.add_argument("--ignore_case", type=bool_flag, help="Whether ignore the uppercase when comparing the strings",
                        default=True)
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="the minimum coverage that the set of ngrams has")
    parser.add_argument('--do_lemmatize', type=bool_flag, help='whether lemmatize the word', default=True)
    parser.add_argument('--lemmatizer_type', type=str, help='the lemmatizer used for the source language',
                        default='spacy')
    parser.add_argument('--use_ngrams', type=bool_flag, help='whether use ngrams of lemmata', default=False)
    parser.add_argument("--use_multiprocessing", type=bool_flag, default=True,
                        help="whether use multiprocessing")
    parser.add_argument('--updated_ngrams', type=bool_flag, help='whether the updated ngrams target words',
                        default=True)
    return parser


# proposing the most associated ngram
def filter_string(candidates_dict: OrderedDict):
    for ngram1, value1 in candidates_dict.items():
        flag = True
        for ngram2, value2 in candidates_dict.items():
            if value1[0] > value2[0]:  # if the chi2 decreases, there is no need to compare
                break
            if ngram1 == ngram2:
                continue
            elif ngram1 in ngram2:
                # when the current ngram is a substring of another ngram and the chi2 score is the same
                # we always keep the longest substring with the same chi2 score
                if value1[0] == value2[0]:
                    flag = False  # current ngram1 is a substring, so we do not want it.
                    break
            else:
                pass
        if flag:
            return ngram1, value1


# check if the contingency table has any problems
@nb.jit(nopython=True, cache=True)
def check_contingency_table(contingency_table: np.array):
    if contingency_table[1][0] == 0 and contingency_table[1][1] == 0:  # filter very common ngrams
        return False
    elif contingency_table[0][1] == 0 and contingency_table[1][1] == 0:  # filter possible errors
        return False
    elif contingency_table[0][0] == 0 and contingency_table[0][1] == 0:  # filter possible errors
        return False
    elif contingency_table[0][0] == 0 and contingency_table[1][0] == 0:  # filter possible errors
        return False
    elif contingency_table[0][0] < 2 or contingency_table[0][1] >= contingency_table.sum() / 10:  # filter bad ngrams
        return False
    else:
        return True


# own implementation of chi2: faster than scipy because I use numba to accelerate
@nb.jit(nopython=True, cache=True)
def chi2_score(table, correction=False):
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    chi_squared_stat = np.sum(np.power(table - expected, 2) / expected) if not correction \
        else np.sum(np.power(np.abs(table - expected) - 0.5, 2) / expected)
    return int(chi_squared_stat)


# computing chi2 scores for a set ngrams
def compute_chi2_for_set(results, ngrams_freq, verse_containing_concept_varying: int,
                         verses_no_concept: int, ngrams_verses_to_be_filtered_currently):
    """
    :param results: a dictionary of ngrams and their frequencies TP
    :param ngrams_freq: a dictionary of ngrams and their global frequencies
    :param verse_containing_concept_varying: see function below
    :param verses_no_concept: the number of verses that do not contain the concept
    :param ngrams_verses_to_be_filtered_currently: see function below
    :return: all_candidates: a dictionary of ngrams and the chi2 and contingency table
    """
    all_candidates = dict()
    for ngram, frequencies in results.items():
        contingency_table = np.zeros((2, 2), dtype=int)
        # the number of verses where both the concept and the ngram occurs
        contingency_table[0][0] = frequencies  # TP
        # the number of verses where only the ngram occur but the concept does not occur

        c01 = (ngrams_freq[ngram] - (0 if ngram not in ngrams_verses_to_be_filtered_currently
                                     else ngrams_verses_to_be_filtered_currently[ngram])) - frequencies
        # print(ngram)
        # if ngram in ngrams_verses_to_be_filtered_currently:
        #     print(ngrams_verses_to_be_filtered_currently[ngram])
        # assert c01 >= 0
        contingency_table[0][1] = c01

        # the number of verses where only the concept occur but the ngram does not occur
        contingency_table[1][0] = verse_containing_concept_varying - frequencies
        # the number of verses where neither the concept nor the ngram occur
        contingency_table[1][1] = verses_no_concept - contingency_table[0][1]

        if not check_contingency_table(contingency_table):
            continue
        # making the computing of chi2_contingency into the numba
        all_candidates[ngram] = (chi2_score(contingency_table, correction=False), contingency_table)
    return all_candidates


# function for both forward pass and backward pass
def search_associated_ngrams(concept_verse_set: set, ngrams_freq: dict, tgt_verse_ngrams: dict,
                             threshold=0.9, backward=False, maximum_iteration_num=3, concept=None):
    # maybe also return the verses containing the selected ngram
    """
    :param concept_verse_set: a set verse (IDs) containing the concept
    :param ngrams_freq: a dict of ngrams and their global frequencies in the parallel verses
    :param tgt_verse_ngrams: a dict of verse IDs and the ngrams it contains
    :param threshold: a hyper-parameter indicating the minimum coverage of the set of proposed ngrams
    :param backward: indicator whether is the backward pass
    :param maximum_iteration_num: number of maximum iterations
    :param concept: only for backward pass, the concept used to compute in the FP
    :return: selected_ngrams_dict: a dict containing the proposed ngram as well as its statistics including chi2 and con
    """
    if backward:
        assert concept is not None
        # special case, if the concept consists of multiple words
        concept_list = concept.split('|||')

    # first we have to get the real concept_verse_set according to the parallel verses
    concept_verse_set = concept_verse_set.intersection(set(tgt_verse_ngrams.keys()))

    # for computing coverage
    verse_containing_concept_initial = len(concept_verse_set)

    # vary in each iteration (the number of verses containing the concept but not matched yet)
    verse_containing_concept_varying = len(concept_verse_set)

    # the number of verses that do not contain the concept (this should remain constant)
    verses_no_concept = len(tgt_verse_ngrams) - verse_containing_concept_initial

    # the dict to store the selected associated ngrams
    selected_ngrams_dict = dict()
    proposed_ngram_set = set()

    # coverage: the number of verses covered by the current set of selected_ngrams_dict
    coverage = 0
    last_coverage = 0  # the coverage of the last proposed ngram
    iteration_num = 0
    # for BP, we search one more time (because the first time will always be the focal concept)
    maximum_iteration_num = maximum_iteration_num + 2 if backward else maximum_iteration_num

    # break_flag
    break_flag = False

    # there are no verses containing the concepts, we should directly return
    if verse_containing_concept_initial == 0:
        return selected_ngrams_dict, coverage

    while coverage / verse_containing_concept_initial < threshold:
        # stop searching if number of iterations exceeds the maximum iteration num
        if iteration_num >= maximum_iteration_num:
            break
        # store the ngram candidates as well as the number of verses in which they occur
        results = Counter()
        # a dictionary to store the ngrams, how many verses do they occur in those covered verse
        # for correcting the number recorded in the contingency table
        ngrams_verses_to_be_filtered_currently = Counter()

        # if it is in the backward pass for the first time,
        # we first consider the focal concept
        if backward and iteration_num == 0:
            results[concept] = 0

        # absorb each target verse if it contains the concept
        for verseID in concept_verse_set:
            # in the first iteration of backward pass, we consider the focal concept always
            if backward and iteration_num == 0:
                for c in concept_list:
                    if c in tgt_verse_ngrams[verseID]:
                        results[concept] += 1
                        break
            else:
                flag = True
                # see one of the selected ngram is in this verse (if it is, it means it has been matched, otherwise not)
                for ngram in proposed_ngram_set:
                    if ngram in tgt_verse_ngrams[verseID]:
                        flag = False
                        # we store the ngrams which occur in those matched verses for correcting the contingency table
                        ngrams_verses_to_be_filtered_currently += Counter(tgt_verse_ngrams[verseID])

                        # if ngram is one string of concept list, we should create / add 1 to the statistics
                        if backward:
                            # this to ensure that we will not add additional count if the string only contains one word
                            if ngram in concept_list and len(concept_list) > 1:
                                if ngram not in ngrams_verses_to_be_filtered_currently:
                                    ngrams_verses_to_be_filtered_currently[concept] = 1
                                else:
                                    ngrams_verses_to_be_filtered_currently[concept] += 1
                        break
                # only consider the ngrams in the verses not covered yet by the set
                if flag:
                    results += Counter(tgt_verse_ngrams[verseID])
        # print(len(results))
        # ngrams_verses_to_be_filtered_currently = dict(ngrams_verses_to_be_filtered_currently)
        # in very rare cases, results is empty, we directly stop the iterations:
        if len(results) == 0:
            break

        # filter the ngrams whose frequencies are below certain values
        # filter more ngrams when there are so many verses
        # 100 -> 3, 1000 -> 30
        min_coverage_of_a_ngram = min(max(2, len(concept_verse_set) // 33), 10)
        results = {k: v for k, v in results.items() if v >= min_coverage_of_a_ngram}

        # for debugging
        # print(f"backward: {backward}, {len(concept_verse_set)}, {iteration_num}, {len(results)}")
        all_candidates = compute_chi2_for_set(results, ngrams_freq, verse_containing_concept_varying,
                                              verses_no_concept, ngrams_verses_to_be_filtered_currently)
        # if all chi2 are bad, we break
        if len(all_candidates) == 0:
            break
        # select at most first 100 ngrams
        # sorting criterion :
        # 1. chi2 value
        # 2. number of verses covered by this ngram
        # 3. how many words does this ngram include (fewer is better)
        # 4. length of ngrams (longer is better)
        items = sorted(all_candidates.items(),
                       key=lambda item: (item[1][0],
                                         item[1][1][0][0],
                                         -len([k for k in item[0].split('$') if k]),
                                         len(item[0])), reverse=True)[:100]

        selected_ngram, selected_ngram_stats = filter_string(OrderedDict(items))
        # print(items)
        # early stop
        # if the current coverage is 10 times smaller than the coverage of the last ngram, we could stop
        if selected_ngram_stats[1][0][0] < 10:
            if selected_ngram_stats[1][0][0] <= math.ceil(last_coverage / 10):
                break
            # if the current ngram's coverage is small or is 50 times smaller than the verses where this ngram occurs
            # it is bad and we should also terminate the search
            elif selected_ngram_stats[1][0][0] < math.ceil(selected_ngram_stats[1][0][1] / 50):
                break
        # here we might not need to limit the maximum number
        # this number should be slighter larger than the maximum frequencies hyperparam of the concepts
        if not backward and selected_ngram_stats[1][0][1] > 3000:
            # we want to break when the identified ngram is so common (except for the first search)
            # only in the forward search we do this (to filter less likely target ngrams)
            if iteration_num == 0:
                break_flag = True
            else:
                break

        # update coverage and the selected_ngrams_dict
        coverage += selected_ngram_stats[1][0][0]
        last_coverage = selected_ngram_stats[1][0][0]

        # if it is the backward search, we should add all concepts to the selected_ngrams_dict,
        # each with the same statistics

        if backward and iteration_num == 0:
            for c in concept_list:
                proposed_ngram_set.add(c)

        proposed_ngram_set.add(selected_ngram)
        selected_ngrams_dict[selected_ngram] = selected_ngram_stats

        # the number of verses left to be covered
        verse_containing_concept_varying = verse_containing_concept_initial - coverage
        iteration_num += 1

        # see if we need to break
        if break_flag:
            break
    return selected_ngrams_dict, coverage


def construct_new_concept(identified_ngrams: set, tgt_verse_ngrams: dict):
    """
    :param identified_ngrams: a list of identified target strings
    :param tgt_verse_ngrams: a dict of verse IDs and the ngrams it contains
    :return: a set of verses that contains the identified_ngrams
    """
    concept_verse_set = set()
    add = concept_verse_set.add
    for verseID, verses_ngram in tgt_verse_ngrams.items():
        for ngram in identified_ngrams:
            if ngram in verses_ngram:
                add(verseID)
                break
    return concept_verse_set


# using process_a_concept does not provide efficiency
# the primary reason might be for each concept, some numba codes has to be compiled
# therefore we could directly deal with multiple concepts for a process
# this is the function to perform both forward and backward pass for a given focal concept
def process_a_set_of_concept(concept_set, concept_verse_IDs, ngrams_freq_dict,
                             tgt_verse_ngrams, eng_concept_freq_dict, eng_verse_concepts, threshold, use_ngrams=False):
    return_results = dict()  # to store the set of the concepts and their fp and bp results
    for i, concept in enumerate(concept_set):
        # forward pass
        # print('FP')
        tgt_results_fp_bp = dict()
        identified_ngrams_dict, fp_coverage = search_associated_ngrams(concept_verse_IDs[concept], ngrams_freq_dict,
                                                                       tgt_verse_ngrams, threshold=threshold)
        if len(identified_ngrams_dict) == 0:
            # if we did not get associated string for this concept in this language, we do not have to proceed BP
            # store the data to tgt_results_fp_fp
            tgt_results_fp_bp['fp'] = {}
            tgt_results_fp_bp['bp'] = {}
            return_results[concept] = tgt_results_fp_bp
            continue
        else:
            tgt_results_fp_bp['fp'] = identified_ngrams_dict

        identified_ngrams = set(ngram for ngram in identified_ngrams_dict.keys())

        # constructing a set of verses containing identified tgt strings (maybe directly returned by above, and merge)
        identified_ngrams_set = construct_new_concept(identified_ngrams, tgt_verse_ngrams)

        # if we find the number of verses for backward search is very large, we should not perform backward pass
        # because this means the forward search can diverge too much
        # print('BP')
        if (len(identified_ngrams_set) > 1000 and use_ngrams) or (len(identified_ngrams_set) > 4000):
            tgt_results_fp_bp['bp'] = 'NA'
            return_results[concept] = tgt_results_fp_bp
            continue

        # we do not have to perform bp if we have very few false positives
        # (less than 10% of the coverage of identified ngrams in fp)
        if fp_coverage / len(identified_ngrams_set) >= threshold:
            tgt_results_fp_bp['bp'] = {}
            return_results[concept] = tgt_results_fp_bp
            continue
        else:
            # backward pass
            # in BP, the verse set to be covered is the verses matched in FP

            # add the artificial concept to the src freq dict
            if concept not in eng_concept_freq_dict:
                eng_concept_freq_dict[concept] = \
                    len(concept_verse_IDs[concept].intersection(set(tgt_verse_ngrams.keys())))

            identified_concepts_dict, bp_coverage = search_associated_ngrams(identified_ngrams_set,
                                                                             eng_concept_freq_dict,
                                                                             eng_verse_concepts, threshold=threshold,
                                                                             backward=True, concept=concept)
            tgt_results_fp_bp['bp'] = identified_concepts_dict
            return_results[concept] = tgt_results_fp_bp
        # if i % 5 == 0:
        # print(i, concept)
        # print(return_results[concept])
        # print()
    return return_results


def main(params):
    src_lang = params.src_lang
    store_dir = params.required_path
    threshold = params.threshold
    results_path = params.results_path  # the path to store the results
    use_multiprocessing = params.use_multiprocessing
    do_lemmatize = params.do_lemmatize
    lemmatizer_type = params.lemmatizer_type
    use_ngrams = params.use_ngrams
    updated_ngrams = params.updated_ngrams

    # TODO
    # if updated_ngrams is true, we should load ngrams from another directory

    ngrams_dir = store_dir + '/ngrams' if not updated_ngrams else store_dir + '/ngrams/updated'

    parallel_dir = f"{store_dir}/parallel_data/{src_lang}" if not updated_ngrams \
        else f"{store_dir}/parallel_data/{src_lang}/updated"

    concept_path = parallel_dir + '/concept_list'

    results_path = f"{results_path}/{src_lang}" if not updated_ngrams else f"{results_path}/{src_lang}/updated"

    if not os.path.exists(f"{results_path}"):
        os.makedirs(f"{results_path}")

    with open(parallel_dir + '/concept_verse_IDs.pickle', 'rb') as handle:
        concept_verse_IDs = pickle.load(handle)
    tgt_langs = get_langs()
    # tgt_langs = ['rus']
    # test
    # tgt_langs = ['aai', 'aak', 'cmn', 'deu', 'fra', 'zho']
    # tgt_langs = ['cmn']
    # we use lemmata data when the source language is English

    print(f"Src language: {src_lang}")

    if src_lang == 'eng':
        eng_data_name = parallel_dir + '/eng-metadata' if not do_lemmatize \
            else parallel_dir + f"/eng-metadata_{lemmatizer_type}"

        eng_data_name = eng_data_name + '.pickle' if not use_ngrams \
            else eng_data_name + '_new.pickle'

        print(eng_data_name)
        with open(eng_data_name, 'rb') as handle:
            src_data = pickle.load(handle)
            # this should be all the concepts considered by our work and the verses where they appear
            # e.g., 'bird': {#1043, #1044}
            original_src_concept_freq_dict, original_src_verse_concepts = src_data
            # src_concept_freq_dict, src_verse_concepts = src_data
            # original_src_concept_freq_dict should be similar to ngrams_freq_dict (concept: global frequencies)
            # original_src_verse_concepts should be a dict of verse id and the concept it contains
    else:
        # if the source language is not english, we should directly use the ngrams data of that language
        with open(f"{ngrams_dir}/{src_lang}.pickle", 'rb') as handle:
            src_data = pickle.load(handle)
            original_src_concept_freq_dict, original_src_verse_concepts = src_data
            # src_concept_freq_dict, src_verse_concepts = src_data

    # the concepts needs to be computed
    # concepts_list = list(concept_verse_IDs.keys())
    concepts_list = read_concepts(concept_path)
    # print(concepts_list)
    # concepts_list = ['$hand$']
    print(len(concepts_list))

    # test
    # concepts_list = concepts_list[:100]
    # concepts_list = ['$bird$']

    with open(f"{parallel_dir}/parallel_ids.pickle", 'rb') as handle:
        parallel_ids = pickle.load(handle)

    with open(f"{parallel_dir}/tgt_lang_ngrams_freq_to_subtract.pickle", 'rb') as handle:
        tgt_lang_ngrams_freq_to_subtract = pickle.load(handle)

    with open(f"{parallel_dir}/src_lang_ngrams_freq_to_subtract.pickle", 'rb') as handle:
        src_lang_ngrams_freq_to_subtract = pickle.load(handle)

    start = time.time()

    for tgt_lang in tgt_langs:

        # if for this target language, the results are available, we skip
        if use_ngrams:
            if os.path.exists(f"{results_path}/{src_lang}/{tgt_lang}_results_lemmata_ngrams.pickle"):
                pass
                # continue
        else:
            if os.path.exists(f"{results_path}/{src_lang}/{tgt_lang}_results_lemmata.pickle"):
                pass
                # continue

        # for each language, we store the data:
        tgt_results = dict()
        print(tgt_lang)
        with open(f"{ngrams_dir}/{tgt_lang}.pickle", 'rb') as handle:
            tgt_lang_data = pickle.load(handle)
            # tgt_lang_data contains [all_ngrams_freq_dict, filtered_tgt_lang_ngrams]
            # all_ngrams_freq_dict: dict of ngram and its global frequencies
            # filtered_tgt_lang_ngrams: dict of verse id and ngrams it contains
            ngrams_freq_dict, tgt_verse_ngrams = tgt_lang_data

        # updated version of ngrams_freq_dict should be used (real parallel data)
        # update the statistics according to the parallel verses
        common_verseIDs = parallel_ids[tgt_lang]
        tgt_verse_ngrams = {ID: tgt_verse_ngrams[ID] for ID in common_verseIDs}
        src_verse_concepts = {ID: original_src_verse_concepts[ID] for ID in common_verseIDs}

        # update the corresponding global frequencies according to what should subtracted for this tgt language
        ngrams_freq_dict = Counter(ngrams_freq_dict) - tgt_lang_ngrams_freq_to_subtract[tgt_lang]
        # print(ngrams_freq_dict)
        src_concept_freq_dict = Counter(original_src_concept_freq_dict) - src_lang_ngrams_freq_to_subtract[tgt_lang]

        # we set the maximum number of processes to be the cpu count or the number of focal concepts
        maximum_parallel_process_number = 4
        # print(f"number of cores: {multiprocessing.cpu_count()}")
        if len(concepts_list) < 100:
            use_multiprocessing = False
        elif len(concepts_list) < 500:
            maximum_parallel_process_number = min(multiprocessing.cpu_count() - 1, 4)
        elif len(concepts_list) < 1000:
            maximum_parallel_process_number = min(multiprocessing.cpu_count() - 1, 8)
        elif len(concepts_list) < 4000:
            maximum_parallel_process_number = min(multiprocessing.cpu_count() - 1, 16)
        else:
            maximum_parallel_process_number = min(multiprocessing.cpu_count() - 1, 32)

        # print(f"use_multiprocessing: {use_multiprocessing}")

        if use_multiprocessing:
            # print(f"maximum_parallel_process_number: {maximum_parallel_process_number}")
            processes_num = maximum_parallel_process_number
            pool = multiprocessing.Pool(processes_num)
            chunked_results = []
            concepts_chunked = np.array_split(concepts_list, processes_num)
            for i in range(processes_num):
                concepts_part = concepts_chunked[i].tolist()
                chunked_results.append(pool.apply_async(process_a_set_of_concept,
                                                        (concepts_part, concept_verse_IDs, ngrams_freq_dict,
                                                         tgt_verse_ngrams, src_concept_freq_dict, src_verse_concepts,
                                                         threshold, use_ngrams)))
            pool.close()
            pool.join()
            for r in chunked_results:
                tgt_results.update(r.get())
        else:
            tgt_results = process_a_set_of_concept(concepts_list, concept_verse_IDs,
                                                   ngrams_freq_dict, tgt_verse_ngrams,
                                                   src_concept_freq_dict, src_verse_concepts,
                                                   threshold=threshold, use_ngrams=use_ngrams)
        # for k, v in tgt_results.items():
        #     print(k, v)

        # for each language, we store a results
        if use_ngrams:
            with open(f"{results_path}/{tgt_lang}_results_lemmata_ngrams.pickle", 'wb') as handle:
                pickle.dump(tgt_results, handle)
            # break
        else:
            with open(f"{results_path}/{tgt_lang}_results_lemmata.pickle", 'wb') as handle:
                pickle.dump(tgt_results, handle)

    end = time.time()
    print(f"Runtime: {end-start} s")


if __name__ == '__main__':
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    main(params)

"""

eng:

export PYTHONIOENCODING=utf8; nohup python -u association_finder.py > ./find_association_log.txt 2>&1 &
server: sigma pid: 19199
done
Runtime: 90465.46177053452 s

eng:
# export PYTHONIOENCODING=utf8; nohup python -u association_finder.py > ./find_association_log_eng_updated.txt 2>&1 &
server: delta pid:18877
runing from 20:26
"""
