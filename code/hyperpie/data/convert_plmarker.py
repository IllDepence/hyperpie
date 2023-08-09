""" Convert preprocessed data into PL-Marker format.
"""

import json
import os
import re
import sys
from collections import OrderedDict, defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize


def _replace_brackets(text):
    """ Replace brackets with replacement tokens.
    """

    brackets = {
        "(": "-LRB-",
        ")": "-RRB-",
        "[": "-LSB-",
        "]": "-RSB-",
        "{": "-LCB-",
        "}": "-RCB-"
    }

    # TODO

    brackets

    return None


def _convert_single_para(para):
    """ Convert single paragraph to PL-Marker format.
    """

    conv_para = OrderedDict()
    conv_para['doc_key'] = '{}-{}'.format(
        para['document_id'],
        para['paragraph_index']
    )

    conv_para['sentences'] = []
    # tokenize text into sentences and words
    para_sentences = sent_tokenize(para['text'])
    # dertermine start and end offset of sentences in text
    para_sentence_offsets = {}
    para_sent_word_offsets = {}
    para_word_offsets = {}
    global_word_idx = 0
    for sent_idx, sent in enumerate(para_sentences):
        # generate conversion output
        conv_para['sentences'].append([])  # list of words
        # get offset of sentence in para
        sent_start = para['text'].find(sent)
        sent_end = sent_start + len(sent)
        para_sentence_offsets[sent_idx] = (sent_start, sent_end)
        # tokenize sentence into words
        sent_words = word_tokenize(sent)
        para_sent_word_offsets[sent_idx] = {}
        # keep track of how often each potentially reoccurring
        # word has already been seen
        word_seen_counts = defaultdict(int)
        for word_idx, word, in enumerate(sent_words):
            # generate conversion output
            conv_para['sentences'][sent_idx].append(word)
            assert conv_para['sentences'][sent_idx][word_idx] == word
            # get offsets of sentence words in para
            word_matches = [
                m for m in re.finditer(re.escape(word), sent)
            ]
            # determine start and end offset of word in sentence, taking
            # into account that the same word may occur multiple times
            curr_match = word_matches[word_seen_counts[word]]
            para_sent_word_offsets[sent_idx][word_idx] = (
                curr_match.start(),
                curr_match.end()
            )
            para_word_offsets[global_word_idx] = (
                sent_start + curr_match.start(),
                sent_start + curr_match.end()
            )
            # increment counters
            word_seen_counts[word] += 1
            global_word_idx += 1

    with open('/tmp/conv_para.json', 'w') as f:
        json.dump(conv_para, f, indent=2)
    # TODO
    # - tokenize paragraph into sentences while keeping track of offsets
    # - iterate over entities
    #   - iterate over surface forms
    #     - create "ner" dict entries
    #     - create "clusters" dict entries
    # - iterate over relations
    #   - create "relations" dict
    #     - open question: how to handle
    #       1. relations accross sentence boundaries
    #       2. relations between entities with multiple surface forms
    conv_para['ner'] = []

    conv_para['relations'] = []

    return conv_para


def convert(annots_path):
    # load and pre-process annotated text segments
    save_path = '../data/'
    annots_fn = os.path.basename(annots_path)
    annots_fn_base, ext = os.path.splitext(annots_fn)
    annots_processed_fn = f'{annots_fn_base}_plmarker{ext}'

    # load annotations
    with open(annots_path, 'r') as f:
        annots = json.load(f)
    # process annotations
    annots_processed = []
    for para in annots:
        para_proc = _convert_single_para(para)
        annots_processed.append(para_proc)

    # save converted annotations
    with open(os.path.join(save_path, annots_processed_fn), 'w') as f:
        json.dump(annots_processed, f)


if __name__ == '__main__':
    # check command line arguments
    if len(sys.argv) != 2:
        print(
            'Usage: python hyperpie/data/convert_plmarker.py '
            '/path/to/annots.json'
        )
        sys.exit(1)
    annots_path = sys.argv[1]
    convert(annots_path)
