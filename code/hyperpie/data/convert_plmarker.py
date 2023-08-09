""" Convert preprocessed data into PL-Marker format.
"""

import json
import os
import random
import re
import sys
from collections import OrderedDict
from nltk.tokenize import sent_tokenize, word_tokenize


def _replace_brackets(sentence):
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

    new_sentence = []
    for word in sentence:
        if word in brackets:
            new_sentence.append(brackets[word])
        else:
            new_sentence.append(word)

    return new_sentence


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
        # prepare list of words in sentence that is safe for later
        # regex matching (needed b/c nltk.work_tokenize converts
        # quotation marks to `` and '')
        # (see: https://stackoverflow.com/a/32197336)
        nltk_regex_safe_words = []
        for word in sent_words:
            if word in ['``', "''"]:
                safe_word = r'``|\'\'|\"'
            else:
                safe_word = re.escape(word)
            nltk_regex_safe_words.append(safe_word)
        # create regex for tokenised words of sentence with optional
        # whitespace in between, example:
        # 'This is (maybe) a test.'
        # -> r'(This)\s*(is)\s*(\()\s*(maybe)\s*(\))\s*(a)\s*(test)\s*(\.)
        words_patt = re.compile(
            r'(' +
            r')\s*('.join(nltk_regex_safe_words) +
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
            try:
                assert word_match[2] == word
            except AssertionError:
                if word in ['``', "''", "\""]:
                    assert word_match[2] in ['``', "''", "\""]
                    # b/c nltk.work_tokenize converts
                    # quotation marks to `` and ''
                    # (see: https://stackoverflow.com/a/32197336)
                else:
                    raise AssertionError
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

    # convert NER annotations
    surf_id_to_global_word_offset = {}
    surf_id_to_send_idx = {}
    for e_id, entity in para['annotation']['entities'].items():
        e_type = entity['type']
        for surf_form in entity['surface_forms']:
            if surf_form['id'] == '#c1840f52-7e45-4dea-9fbf-47a5607675e3':
                # NOTE: known edge with the data where the conversion
                # method gets confused b/c a sentnce match is not unique
                continue
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
            assert output_offset_start is not None
            assert output_offset_end is not None
            # NOTE: if "clusters" key turns out to be needed in
            # output, create here if start and end differ
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
            surf_id_to_send_idx[surf_form['id']] = output_sent_idx
            # save output
            conv_para['ner'][output_sent_idx].append(
                (output_offset_start, output_offset_end, e_type)
            )

    # convert relation annotations
    for rel_id, relation in para['annotation']['relations'].items():
        for evidence in relation['evidences']:
            source_surf_id = evidence['source_surface_form']
            target_surf_id = evidence['target_surface_form']
            source_offset = surf_id_to_global_word_offset[source_surf_id]
            target_offset = surf_id_to_global_word_offset[target_surf_id]
            source_sent_idx = surf_id_to_send_idx[source_surf_id]
            target_sent_idx = surf_id_to_send_idx[target_surf_id]
            if source_sent_idx != target_sent_idx:
                # not compatible with PL-Marker output format
                continue
            # save output
            conv_para['relations'][source_sent_idx].append((
                source_offset[0],
                source_offset[1],
                target_offset[0],
                target_offset[1],
                'USED-FOR'
            ))

    # replace brackets in sentences with replacement tokens
    for sent_idx, sent in enumerate(conv_para['sentences']):
        conv_para['sentences'][sent_idx] = _replace_brackets(sent)

    return conv_para


def convert(annots_path, generate_splits=False):
    # load and pre-process annotated text segments
    save_path = '../data/'
    annots_fn = os.path.basename(annots_path)
    annots_fn_base, ext = os.path.splitext(annots_fn)
    annots_processed_fn = f'{annots_fn_base}_plmarker.jsonl'

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
        for annot in annots_processed:
            f.write(json.dumps(annot) + '\n')

    if not generate_splits:
        return
    # safe train/dev/test splits for 10-fold cross-validation
    random.seed(42)
    random.shuffle(annots_processed)
    n = len(annots_processed)
    num_folds = 10
    # split data into <num_folds> folds
    fold_size = n // num_folds
    folds = []
    for i in range(num_folds):
        if i == num_folds - 1:
            # last fold, add remaining samples
            folds.append(annots_processed[i * fold_size:])
        else:
            # not last fold, add <fold_size> samples
            folds.append(annots_processed[i * fold_size:(i + 1) * fold_size])
    for i in range(num_folds):
        # distribute samples to train/dev/test splits
        # (train and dev eaach one fold, test the remaining folds)
        train = []
        dev = []
        test = []
        for j in range(num_folds):
            if j == i:
                # current fold, add to dev
                dev.extend(folds[j])
            elif j == (i + 1) % num_folds:
                # next fold, add to test
                test.extend(folds[j])
            else:
                # other folds, add to train
                train.extend(folds[j])
        # save train/dev/test splits
        fold_name = f'fold_{i}'
        fold_save_path = os.path.join(save_path, fold_name)
        os.makedirs(fold_save_path, exist_ok=True)
        with open(os.path.join(fold_save_path, 'train.jsonl'), 'w') as f:
            for annot in train:
                f.write(json.dumps(annot) + '\n')
        with open(os.path.join(fold_save_path, 'dev.jsonl'), 'w') as f:
            for annot in dev:
                f.write(json.dumps(annot) + '\n')
        with open(os.path.join(fold_save_path, 'test.jsonl'), 'w') as f:
            for annot in test:
                f.write(json.dumps(annot) + '\n')


if __name__ == '__main__':
    # check command line arguments
    if len(sys.argv) not in [2, 3]:
        print(
            'Usage: python hyperpie/data/convert_plmarker.py '
            '/path/to/annots.jsonl [generate_splits]'
        )
        sys.exit(1)
    annots_path = sys.argv[1]
    if len(sys.argv) == 3:
        generate_splits = bool(sys.argv[2])
    convert(annots_path, generate_splits)
