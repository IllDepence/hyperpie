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
    conv_para['ner'] = []
    conv_para['relations'] = []

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
        conv_para['ner'].append([])  # list of entity offsets+classes
        conv_para['relations'].append([])  # list of rel.offs.+classes
        # get offset of sentence in para
        sent_start = para['text'].find(sent)
        sent_end = sent_start + len(sent)
        para_sentence_offsets[sent_idx] = (sent_start, sent_end)
        # tokenize sentence into words
        sent_words = word_tokenize(sent)
        # get offset of all words in sentence
        para_sent_word_offsets[sent_idx] = {}
        # create regex for tokenised words of sentence with optional
        # whitespace in between, example:
        # 'This is (maybe) a test.'
        # -> r'(This)\s*(is)\s*(\()\s*(maybe)\s*(\))\s*(a)\s*(test)\s*(\.)
        words_patt = re.compile(
            r'(' +
            r')\s*('.join([re.escape(wrd) for wrd in sent_words]) +
            r')'
        )
        words_match = words_patt.search(sent)
        assert words_match is not None
        word_matches = words_match.groups()
        word_matches = [
            (
                words_match.start(i+1),
                words_match.end(i+1),
                words_match.group(i+1)
            )
            for i in
            range(len(words_match.groups()))
        ]
        # keep track of how often each potentially reoccurring
        # word has already been seen
        for word_idx, word, in enumerate(sent_words):
            # generate conversion output
            conv_para['sentences'][sent_idx].append(word)
            assert conv_para['sentences'][sent_idx][word_idx] == word
            # get offsets of sentence words in para
            word_match = word_matches[word_idx]
            assert word_match[2] == word
            # determine start and end offset of word in sentence, taking
            # into account that the same word may occur multiple times
            para_sent_word_offsets[sent_idx][word_idx] = (
                word_match[0],
                word_match[1]
            )
            para_word_offsets[global_word_idx] = (
                sent_start + word_match[0],
                sent_start + word_match[1]
            )
            # increment counters
            global_word_idx += 1

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
    surf_id_to_global_word_offset = {}
    for e_id, entity in para['annotation']['entities'].items():
        e_type = entity['type']
        for surf_form in entity['surface_forms']:
            # determine global word offset of surface form
            output_offset_start = None
            output_offset_end = None
            for global_word_idx, word_offset in para_word_offsets.items():
                if (
                    surf_form['start'] >= word_offset[0] and
                    surf_form['start'] <= word_offset[1]
                ):
                    # surface form start within word, mark as start
                    output_offset_start = global_word_idx
                if (
                    surf_form['end'] >= word_offset[0] and
                    surf_form['end'] <= word_offset[1]
                ):
                    # surface form end within word, mark as end
                    output_offset_end = global_word_idx
                if (
                    output_offset_start is not None and
                    output_offset_end is not None
                ):
                    # both offsets found, stop searching
                    break
            try:
                assert output_offset_start is not None
            except:
                # initialize debugger to step through code
                import pdb
                pdb.set_trace()
                import sys
                print('bye')
                sys.exit()
            assert output_offset_end is not None
            surf_id_to_global_word_offset[surf_form['id']] = (
                output_offset_start,
                output_offset_end
            )
            # determine sentence in which surf form appears
            output_sent_idx = None
            for sent_idx, sent_offsets in para_sentence_offsets.items():
                if (
                    sent_offsets[0] <= surf_form['start'] and
                    sent_offsets[1] >= surf_form['end']
                ):
                    # surf form within sentence, stop searching
                    output_sent_idx = sent_idx
                    break
            assert output_sent_idx is not None
            # save output
            conv_para['ner'][output_sent_idx].append(
                (output_offset_start, output_offset_end, e_type)
            )

    with open('/tmp/conv_para.json', 'w') as f:
        json.dump(conv_para, f, indent=2)

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
