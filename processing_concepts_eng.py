import argparse
import pickle
from processing_ngrams import find_version_most, read_verses
from collections import Counter
from nltk import pos_tag, word_tokenize

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
    parser.add_argument('--do_lemmatize', type=bool_flag, help='whether lemmatize the word', default=True)
    parser.add_argument('--lemmatizer_type', type=str, help='the lemmatizer used for the source language',
                        default='spacy')
    parser.add_argument("--ignore_case", type=bool_flag, help="Whether ignore the uppercase when comparing the strings",
                        default=True)
    parser.add_argument('--required_path', type=str, help='the path that stores required files by preprocessing para',
                        default='/mounts/data/proj/yihong/newhome/ConceptNetwork/required_files')
    parser.add_argument('--transform_lemmata', type=str, help='whether transform the lemmata', default=True)
    parser.add_argument('--load_additional_ngrams', type=str, help='whether load additional lemmata sequences',
                        default=True)

    parser.add_argument('--min_freq', type=int, help='minimum frequency of the concept', default=5)
    parser.add_argument('--max_freq', type=int, help='maximum frequency of the concept', default=2000)

    parser.add_argument('--updated_ngrams', type=bool_flag, help='whether the updated ngrams target words',
                        default=True)

    return parser


def check_string_validity(key):
    if key == '$':
        return False
    if key == '$-$':
        return False
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
    elif '\'' in key:
        return False
    return True


def lemmata_transformer(verse_content, lemmata_list, substring_set: set):
    """
    :param verse_content: a string that represent a verse, e.g., $i$am$standing$here
    :param lemmata_list: a list of all lemmata in this verse, e.g., [$i$, $am$, ...]
    :param substring_set: a dict of substring and lemmata, e.g., {'$adulter': [$adultery, $adulteress]}
    :return: a list
    """
    for substring in substring_set:
        if substring in verse_content:
            for i in range(len(lemmata_list)):
                if substring in lemmata_list[i]:
                    lemmata_list[i] = substring  # change to correct lemmata substring

    return list(set(lemmata_list))


def is_new_testament(verse_ID):
    if int(verse_ID) < 40001001:
        return False
    return True


def lemmatize(lemmatizer, sent, lemmatizer_type='nltk', return_difference=False):
    differences = []
    if lemmatizer_type == 'nltk':
        tokenized = word_tokenize(sent)
        after_pos_tagged = pos_tag(tokenized)
        after_lemmatized = []
        for w, p in after_pos_tagged:
            if p[0] in {'N', 'V', 'J', 'R'}:
                tag = p[0].lower() if p[0] != 'J' else 'a'
                lemma = lemmatizer.lemmatize(w, tag)
                after_lemmatized.append(lemma)
                if return_difference and w != lemma:
                    differences.append((w, lemma))
            else:
                after_lemmatized.append(w)
        if return_difference:
            return ' '.join(after_lemmatized), differences
        else:
            return ' '.join(after_lemmatized)
    elif lemmatizer_type == 'spacy':
        doc = lemmatizer(sent)
        if return_difference:
            for token in doc:
                if str(token) != token.lemma_:
                    differences.append((str(token), token.lemma_))

        if return_difference:
            return ' '.join([token.lemma_ for token in doc]), differences
        else:
            return ' '.join([token.lemma_ for token in doc])
    else:
        raise ValueError


def write_concepts(concepts, concept_path, nt_frequencies=None, total_frequencies=None):
    if nt_frequencies is not None:
        assert total_frequencies is not None
    else:
        assert total_frequencies is None

    with open(concept_path, 'w', encoding='utf-8') as f:
        for concept in concepts:
            if nt_frequencies is None:
                f.write(concept + '\n')
            else:
                if total_frequencies[concept] > 1000:
                    print(f"{concept}, {nt_frequencies[concept]}, {total_frequencies[concept]}")
                    pass
                f.write(f"{concept}, {nt_frequencies[concept]}, {total_frequencies[concept]}\n")


def main(params):
    do_lemmatize = params.do_lemmatize
    lemmatizer_type = params.lemmatizer_type
    ignore_case = params.ignore_case
    store_dir = params.required_path
    updated_ngrams = params.updated_ngrams

    min_freq = params.min_freq
    max_freq = params.max_freq

    # if we restrict the ngrams to be within white-spaced tokenized things, we should use different directories
    english_dir = f"{store_dir}/parallel_data/eng" if not updated_ngrams else f"{store_dir}/parallel_data/eng/updated"
    concept_path = english_dir + '/' + 'concept_list'

    transform_lemmata = params.transform_lemmata
    load_additional_ngrams = params.load_additional_ngrams

    if transform_lemmata:
        with open(english_dir + '/substring_set.pickle', 'rb') as handle:
            substring_set = pickle.load(handle)
    else:
        substring_set = dict()

    if load_additional_ngrams:
        with open(english_dir + '/filtered_eng_lemmata_ngrams2-4.pickle', 'rb') as handle:
            additional_verse_ngrams = pickle.load(handle)

    if do_lemmatize:
        if lemmatizer_type == 'nltk':
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
        elif lemmatizer_type == 'spacy':
            import spacy
            lemmatizer = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        else:
            raise ValueError

    src_path, src_file_name = find_version_most('eng')
    src_verseIDs, src_contents = read_verses(src_path)
    # src_contents is a dictionary of IDs and the contents

    # record the strings and their frequencies
    all_strings_freq_dict = Counter()  # global counter
    nt_strings_freq_dict = Counter()  # new testaments counter
    src_lang_strings = {}
    lemmatized_contents = {}  # to store the lemmatized english contents (this is necessary because we want to use)
    lemmatized_differences = []  # to store the differences in the lemmatization
    for i, (verse_ID, verse_content) in enumerate(src_contents.items()):
        # lemmtize the contents
        if do_lemmatize:
            verse_content, diff = lemmatize(lemmatizer, verse_content,
                                            lemmatizer_type=lemmatizer_type, return_difference=True)
            lemmatized_differences += diff
            lemmatized_contents[verse_ID] = verse_content
        verse_content = verse_content.lower() if ignore_case else verse_content
        # record the verse and the strings in a dictionary

        # 0. add lemmata to the set
        # 1. transform some lemmata to its ngrams.
        # 2. add two to four grams of lemmata sequences to the verse strings
        lemmata_list = ['$' + lemma + '$' for lemma in list(set(verse_content.split(' ')))]
        verse_content = ('$' + verse_content + '$').replace(' ', '$')
        # tranform lemmata to substring
        transformed_lemmata = lemmata_transformer(verse_content, lemmata_list, substring_set)
        # transofm the lemmata if any lemma is in the lemmata_substring_dict
        # lemmata_substring_dict: {substring: [lemma1, lemma2, lemma3, ...]}
        verse_strings = dict(Counter(transformed_lemmata))

        # add phrases
        if load_additional_ngrams:
            verse_strings.update(additional_verse_ngrams[verse_ID])

        src_lang_strings[verse_ID] = verse_strings
        if is_new_testament(verse_ID):
            nt_strings_freq_dict += Counter(verse_strings)
        all_strings_freq_dict += Counter(verse_strings)

    # for english, we might want to store the dictionary of all strings' frequencies
    # TODO
    # maybe we should clean (remove) the words that we don't want to include
    src_stored_data = [all_strings_freq_dict, src_lang_strings]

    eng_data_name = english_dir + '/eng-metadata.pickle' if not do_lemmatize \
        else english_dir + f"/eng-metadata_{lemmatizer_type}.pickle"

    print(eng_data_name)

    # store the lemmatized english data (global frequencies and the dict of verseID and concepts)
    with open(eng_data_name, 'wb') as handle:
        pickle.dump(src_stored_data, handle)

    filtered_concept_list = [k for k, v in nt_strings_freq_dict.items()
                             if min_freq <= v <= max_freq and all_strings_freq_dict[k] <= max_freq and
                             check_string_validity(k) is True and len(k) > 3]
    print(len(filtered_concept_list))
    filtered_concept_list = sorted(filtered_concept_list, key=lambda x: (x[1:] if x[0] == '$' else x, len(x)))

    write_concepts(filtered_concept_list, concept_path=concept_path,
                   nt_frequencies=nt_strings_freq_dict, total_frequencies=all_strings_freq_dict)

    # store the lemmatized data for later use (this is necessary because we want to use)
    if do_lemmatize:
        eng_content_name = english_dir + f"/eng-content-data_{lemmatizer_type}.pickle"
        print(eng_content_name)
        with open(eng_content_name, 'wb') as handle:
            pickle.dump(lemmatized_contents, handle)


if __name__ == '__main__':
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    main(params)