import argparse
import os
import time
import pickle
import pandas as pd
import collections
import multiprocessing
import numpy as np
from botok import WordTokenizer
import re
from khmernltk import word_tokenize


"""
This file is used to process the target languages editions and obtain the ngrams representations of each verse
and the global statistics of each ngram. The ngrams data is stored at 
'/mounts/data/proj/yihong/newhome/ConceptNetwork/required_files/ngrams'
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


def get_parser():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(description="ngrams processing")

    # main parameters
    parser.add_argument('--required_path', type=str, help='the path that stores required files by preprocessing para',
                        default='/mounts/data/proj/yihong/newhome/ConceptNetwork/required_files')
    parser.add_argument("--ignore_case", type=bool_flag, help="Whether ignore the uppercase when comparing the strings",
                        default=True)
    parser.add_argument("--ngram_min_len", type=int, help='minimum length of the ngram', default=1)
    # ngram_max_len only works when updated_ngrams is set false
    parser.add_argument("--ngram_max_len", type=int, help='maximum length of the ngram', default=8)
    parser.add_argument("--min_threshold", type=int, help='minimum frequency for a ngram', default=2)
    parser.add_argument('--updated_ngrams', type=bool_flag, help='whether the updated ngrams target words',
                        default=True)
    return parser


# obtain all distinct language's ISO 639-3 codes
def get_langs():
    csv_path = '/mounts/Users/student/yihong/Documents/concept_align'
    pbc_info = pd.read_csv(f"{csv_path}/pbc_table.csv", converters={"language_code": str})
    langs = pbc_info['language_code'].values
    langs = sorted(list(set(langs)))
    return langs


def read_pbc_script_data():
    csv_path = '/mounts/Users/student/yihong/Documents/concept_align'
    pbc_info = pd.read_csv(f"{csv_path}/pbc_table.csv", converters={"language_code": str})
    dict_script = dict(zip(pbc_info['file_name'], pbc_info['script_code']))
    return dict_script


# for finding the verse that has the most verses
# find the bible version for a given language
def find_version_most(language_code):
    if language_code == 'eng':
        return '/nfs/datc/pbc/eng-x-bible-literal.txt', 'eng-x-bible-literal.txt'
    versions = []
    for filename in os.listdir('/nfs/datc/pbc/'):
        fp = os.path.join('/nfs/datc/pbc/', filename)
        if os.path.isfile(fp) and filename[:3] == language_code:
            versions.append(fp)
    if len(versions) == 0:
        raise ValueError
    elif len(versions) == 1:
        parts = versions[0].split('/')
        return versions[0], parts[-1]
    else:
        # check if there is a newworld version for this language:
        newworld = []
        candidate = ('', 0)
        for v in versions:
            if 'newworld' in v:
                newworld.append(v)
            f = open(v, 'r', encoding="utf-8")
            length_f = len(f.readlines())
            f.close()
            if length_f > candidate[1]:
                candidate = (v, length_f)
        if len(newworld) == 0:
            parts = candidate[0].split('/')
            return candidate[0], parts[-1]
        elif len(newworld) == 1:
            return newworld[0], newworld[0].split('/')[-1]
        else:
            # multiple new world is available
            for v in newworld:
                f = open(v, 'r', encoding="utf-8")
                length_f = len(f.readlines())
                f.close()
                if length_f > candidate[1]:
                    candidate = (v, length_f)
            return candidate[0], candidate[0].split('/')[-1]


# read the contents given the path
def read_verses(path):
    contents = []
    verseIDs = []
    with open(path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            if line[0] == "#":
                continue
            parts = line.strip().split('\t')
            if len(parts) == 2:
                verseIDs.append(parts[0])
                contents.append(parts[1])
    # creating a dictionary for efficiency
    contents_dict = dict(zip(verseIDs, contents))
    return verseIDs, contents_dict


def check_ngram_validity(key, beyond_word=True):
    if '!' in key:
        return False
    # this is useful in some languages
    # elif '-' in key:
    #     return False
    elif '?' in key:
        return False
    elif '.' in key:
        return False
    elif ',' in key:
        return False
    elif '[' in key:
        return False
    elif ']' in key:
        return False
    elif '(' in key:
        return False
    elif ')' in key:
        return False
    elif ';' in key:
        return False
    elif ':' in key:
        return False
    elif '"' in key:
        return False
    elif '$$' in key:
        return False
    elif '„' in key:
        return False
    elif '“' in key:
        return False
    elif '«' in key:
        return False
    elif '»' in key:
        return False
    # chinese
    elif '，' in key:
        return False
    elif '；' in key:
        return False
    elif '。' in key:
        return False
    elif '、' in key:
        return False
    elif '：' in key:
        return False
    elif '‘' in key:
        return False
    elif '’' in key:
        return False
    elif '“' in key:
        return False
    elif '”' in key:
        return False
    elif '！' in key:
        return False
    elif '？' in key:
        return False
    elif '（' in key:
        return False
    elif '）' in key:
        return False
    # if allow the ngrams across word boundaries
    if beyond_word is False:
        if '$' in key[1:-1]:
            return False

    return True


# obtain ngrams for a given sentence
def obtain_ngram(sent, ngram_min_len, ngram_max_len, ignore_case=True, beyond_word=True):
    # when beyond_word is false, we should consider the ngrams as long as possible within a word
    ngrams2count = collections.defaultdict(int)
    sent = ('$' + sent + '$').replace(' ', '$').lower() if ignore_case else ('$' + sent + '$').replace(' ', '$')
    for i in range(len(sent)):
        if beyond_word is True:
            for j in range(i + ngram_min_len, i + ngram_max_len + 1):
                if j > len(sent) - 1:
                    break
                if sent[i:j] == '$':
                    continue
                if len(sent[i:j].replace('$', '').replace('-', '')) == 0:  # prevent the condition only - occurs
                    continue
                if sent[i:j] not in ngrams2count and check_ngram_validity(sent[i:j], beyond_word):
                    ngrams2count[sent[i:j]] = 1
        else:
            # TODO
            # check (some verses there might be too many candidates)
            # consider all ngrams with a word
            for j in range(i + ngram_min_len, len(sent) - 1):
                if sent[i:j] == '$':
                    continue
                if len(sent[i:j].replace('$', '').replace('-', '')) == 0:  # prevent the condition only - occurs
                    continue
                if not check_ngram_validity(sent[i:j], beyond_word):
                    break
                if sent[i:j] not in ngrams2count:
                    ngrams2count[sent[i:j]] = 1
    return dict(ngrams2count)


def verses_obtain_ngram(verse_IDs, sents, ngram_min_len, ngram_max_len, ignore_case=True, beyond_word=True):
    results = {}
    ngrams_freq_dict = collections.Counter()
    for ID, sent in zip(verse_IDs, sents):
        ngrams2count = obtain_ngram(sent, ngram_min_len, ngram_max_len,
                                    ignore_case=ignore_case,
                                    beyond_word=beyond_word)
        results[ID] = ngrams2count
        ngrams_freq_dict += collections.Counter(ngrams2count)
    return results, ngrams_freq_dict


def remove_tokenization(contents: dict):
    """
    this function remove all the spaces in the contents, mainly for chinese texts
    :param contents: a dictionary indexed by the verse ID, the value is the string representation of the verse
    :return:
    """
    for ID, sent in contents.items():
        contents[ID] = sent.replace(' ', '')
    return contents


def tokenize_tibetan(contents: dict, wt: WordTokenizer):
    for ID, sent in contents.items():
        tokens = wt.tokenize(sent, split_affixes=False)
        contents[ID] = re.sub(r" +", ' ', ' '.join([tokens[i]['text'] for i in range(len(tokens))]))
    return contents


def tokenize_khmer(contents: dict):
    for ID, sent in contents.items():
        tokens = word_tokenize(sent, return_tokens=True)
        contents[ID] = re.sub(r" +", ' ', ' '.join(tokens))
    return contents


def main(params):
    start = time.time()
    store_dir = params.required_path
    ignore_case = params.ignore_case
    ngram_min_len = params.ngram_min_len
    ngram_max_len = params.ngram_max_len
    min_threshold = params.min_threshold

    updated_ngrams = params.updated_ngrams

    # TODO
    # if updated_ngrams is true, we should load ngrams from another directory
    pbc_script_dict = read_pbc_script_data()  # script dictionary that stores each file's script

    ngrams_dir = store_dir + '/ngrams' if not updated_ngrams else store_dir + '/ngrams/updated'
    if os.path.exists(ngrams_dir):
        pass
    else:
        os.makedirs(ngrams_dir)

    # store the ngrams data for each language
    tgt_langs = get_langs()
    # tgt_langs = ['cmn', 'zho', 'lzh', 'jpn']
    # tgt_langs = ['fra']

    for i, tgt_lang in enumerate(tgt_langs):

        if os.path.exists(f"{ngrams_dir}/{tgt_lang}.pickle"):
            print(tgt_lang)
            # with open(f"{ngrams_dir}/{tgt_lang}.pickle", 'rb') as handle:
            #     tgt_stored_data = pickle.load(handle)
                # print(tgt_stored_data[0])
            continue

        tgt_path, tgt_file_name = find_version_most(tgt_lang)
        tgt_verseIDs, tgt_contents = read_verses(tgt_path)

        # TODO
        # according to different languages and scripts, use different strategies of saving ngrams
        # if it belongs to those scripts, the maximum length of ngrams might be good to set 5
        if pbc_script_dict[tgt_file_name] in ['Jpan', 'Hani']:  # korean tokenized naturally
            tgt_contents = remove_tokenization(tgt_contents)
        elif pbc_script_dict[tgt_file_name] == 'Tibt' and tgt_lang == 'bod':
            # when dealing with tibetan, we have to care about tokenization
            wt = WordTokenizer()
            tgt_contents = tokenize_tibetan(tgt_contents, wt)
        elif pbc_script_dict[tgt_file_name] == 'Khmr':
            # when dealing with khmer, we use different tokenizer
            tgt_contents = tokenize_khmer(tgt_contents)

        print(tgt_lang)
        # print(tgt_contents)
        tgt_lang_ngrams = dict()
        all_ngrams_freq_dict = collections.Counter()

        # parallelization
        cores_num = max(10, multiprocessing.cpu_count() // 2)
        results = []
        pool = multiprocessing.Pool(cores_num)
        IDs_chunked = np.array_split(tgt_verseIDs, cores_num)

        for i in range(cores_num):
            IDs_parts = IDs_chunked[i].tolist()
            results.append(pool.apply_async(verses_obtain_ngram,
                                            (IDs_parts,
                                             [tgt_contents[ID] for ID in IDs_parts],
                                             ngram_min_len,
                                             ngram_max_len,
                                             ignore_case,
                                             not updated_ngrams)))
            # if not updated_ngrams is True -> this allows ngrams beyond word boundaries
        pool.close()
        pool.join()
        for r in results:
            chucked_1, chunked_2 = r.get()
            tgt_lang_ngrams.update(chucked_1)
            all_ngrams_freq_dict += chunked_2

        # obtain all possible ngrams for a given language (filtering out ngrams with too small ngrams)
        all_ngrams_freq_dict = {x: count for x, count in all_ngrams_freq_dict.items()
                                if min_threshold <= count}

        # only store the ngrams that in the filtered all_ngrams_freq_dict
        filtered_tgt_lang_ngrams = {}
        for verse_ID, ngrams2count in tgt_lang_ngrams.items():
            filtered_tgt_lang_ngrams[verse_ID] = {x: count for x, count in ngrams2count.items()
                                                  if x in all_ngrams_freq_dict}
        # store for each language separately
        tgt_stored_data = [all_ngrams_freq_dict, filtered_tgt_lang_ngrams]
        # print(filtered_tgt_lang_ngrams)
        # print(f"average ngrams per verse "
              # f"{sum([len(v) for k, v in filtered_tgt_lang_ngrams.items()]) / len(filtered_tgt_lang_ngrams)}")
        with open(f"{ngrams_dir}/{tgt_lang}.pickle", 'wb') as handle:
            pickle.dump(tgt_stored_data, handle)

    end = time.time()
    print(f"Runtime: {end-start} s")


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    main(params)

    """
    # export PYTHONIOENCODING=utf8; nohup python -u processing_ngrams.py > ./ngrams_processing_log_updated.txt 2>&1 &
    # server: pi pid: 19705
    # done
    """
