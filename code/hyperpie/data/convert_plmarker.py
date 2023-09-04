""" Convert preprocessed data into PL-Marker format.
"""

import json
import os
import random
import re
import sys
import numpy as np
from collections import OrderedDict, defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
from uuid import uuid4


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


def _convert_single_para(para, loose_matching=False, cap_num_words=False):
    """ Convert single paragraph to PL-Marker format.
    """

    conv_para = OrderedDict()
    if para['paragraph_index'] is not None:
        sub_id = para['paragraph_index']
    else:
        sub_id = str(uuid4())
    conv_para['doc_key'] = '{}-{}'.format(
        para['document_id'],
        sub_id
    )
    conv_para['sentences'] = []
    conv_para['ner'] = []
    conv_para['relations'] = []

    # stats
    tkn_ext_match = 0  # annotation boundaries match token boundaries
    tkn_mid_match = 0  # annotation boundaries lie within token boundaries

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
            if type(cap_num_words) == int and word_idx > cap_num_words:
                # cap sentence length to ensure compatibility with
                # PL-Marker model
                break
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
                # NOTE: known edge case with the data where the conversion
                # method gets confused b/c a sentnce match is not unique
                continue
            # determine global word offset of surface form
            output_offset_start = None
            output_offset_end = None
            for global_word_idx, word_offset in para_word_offsets.items():
                # match character offsets to tokenized words
                # (this can lead to
                #  1. a loss in precision (e.g. if the surface form
                #     starts in the middle of a token)
                #  2. overlapping surface forms (e.g. if the surface
                #     of one entity ends in the middle of a token and
                #     the surface of another entity starts in the same
                #     token — example: 40-fold w/ 40 as v and fold as p)
                if (
                    surf_form['start'] >= word_offset[0] and
                    surf_form['start'] <= word_offset[1]
                ):
                    # surface form start within word, mark as start
                    output_offset_start = global_word_idx
                    if surf_form['start'] == word_offset[0]:
                        tkn_ext_match += 1
                    else:
                        tkn_mid_match += 1
                if (
                    surf_form['end'] >= word_offset[0] and
                    surf_form['end'] <= word_offset[1]
                ):
                    # surface form end within word, mark as end
                    output_offset_end = global_word_idx
                    if surf_form['end'] == word_offset[1]:
                        tkn_ext_match += 1
                    else:
                        tkn_mid_match += 1
                if (
                    output_offset_start is not None and
                    output_offset_end is not None
                ):
                    # both offsets found, stop searching
                    break
            if loose_matching and (
                output_offset_start is None or
                output_offset_end is None
            ):
                continue
            assert output_offset_start is not None
            assert output_offset_end is not None
            # NOTE: if “clusters” key turns out to be needed in
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
            if loose_matching and output_sent_idx is None:
                continue
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

    return conv_para, tkn_ext_match, tkn_mid_match


def convert(
    annots_path,
    generate_splits=False,
    loose_matching=False,
    cap_num_words=False,
    rnd=0
):
    # load and pre-process annotated text segments
    save_path = '../data/'
    annots_fn = os.path.basename(annots_path)
    annots_fn_base, ext = os.path.splitext(annots_fn)
    annots_processed_fn = f'{annots_fn_base}_plmarker.jsonl'

    # load annotations
    with open(annots_path, 'r') as f:
        annots = json.load(f)

    if type(cap_num_words) == int:
        print(f'WARNING: setting word cap to {cap_num_words}')

    # process annotations
    random.seed(rnd)
    annots_processed = []
    annot_boundary_exact_matches = 0
    annot_boundary_mid_token = 0
    for para in annots:
        para_proc, xt_mtch, md_mtch = _convert_single_para(
            para, loose_matching, cap_num_words
        )
        annot_boundary_exact_matches += xt_mtch
        annot_boundary_mid_token += md_mtch
        annots_processed.append(para_proc)

    tkn_boundary_matches = (
        annot_boundary_exact_matches + annot_boundary_mid_token
    )
    print(
        f'Exact token matches: {annot_boundary_exact_matches}'
        f' ({annot_boundary_exact_matches / tkn_boundary_matches:.2f})'
    )
    print(
        f'Mid token matches: {annot_boundary_mid_token}'
        f' ({annot_boundary_mid_token / tkn_boundary_matches:.2f})'
    )

    # save converted annotations
    with open(os.path.join(save_path, annots_processed_fn), 'w') as f:
        for annot in annots_processed:
            f.write(json.dumps(annot) + '\n')

    if not generate_splits:
        print('(not generating splits)')
        return
    # save train/dev/test splits for 10-fold cross-validation

    grouped_folds = []
    if generate_splits == 'doc':
        # group samples by document (paragraph “doc ID” is
        # <paper ID>-<para ID/uuid>)
        print('generating splits by document')
        ppr_groups_dict = defaultdict(list)
        for annot in annots_processed:
            ppr_id = annot['doc_key'].split('-', maxsplit=1)[0]
            ppr_groups_dict[ppr_id].append(annot)
        ppr_groups = list(ppr_groups_dict.values())
        # shuffle groups
        random.shuffle(ppr_groups)
        n = len(ppr_groups)
        num_folds = 10
        # split data into <num_folds> folds (keep ppr groups for now)
        fold_size = n // num_folds
        for i in range(num_folds):
            if i == num_folds - 1:
                # last fold, add remaining samples
                grouped_folds.append(
                    ppr_groups[i * fold_size:]
                )
            else:
                # not last fold, add <fold_size> samples
                grouped_folds.append(
                    ppr_groups[i * fold_size:(i + 1) * fold_size]
                )
    elif generate_splits == 'cls':
        # group samples to optimize for class balance
        # (i.e. number of relations per sample)
        print('generating splits by class')
        # shuffle samples
        num_folds = 5
        random.shuffle(annots_processed)
        num_smpl_rels = []
        smpl2numrels = {}
        for i, annot in enumerate(annots_processed):
            num_rels = 0
            for sentence_rels in annot['relations']:
                num_rels += len(sentence_rels)
            num_smpl_rels.append(num_rels)
            smpl2numrels[i] = num_rels
        # split data into folds
        for fld_i in range(num_folds):
            grouped_folds.append([])  # filled later
        fold_goal_rels = sum(num_smpl_rels) / num_folds
        fold_curr_rels = defaultdict(int)
        fold_total_rels = defaultdict(int)
        for i, smpl in enumerate(annots_processed):
            # determine in which fold to put sample
            max_goal_dist = -1
            alloc_fold_idx = None
            if smpl2numrels[i] == 0:
                # no relations, put in round-robin
                alloc_fold_idx = i % num_folds
            else:
                for fld_i in range(num_folds):
                    # dd based on number of relations in fold
                    goal_dist = abs(fold_goal_rels - fold_curr_rels[fld_i])
                    if goal_dist >= max_goal_dist:
                        alloc_fold_idx = fld_i
                        max_goal_dist = goal_dist
            # add sample to fold
            grouped_folds[alloc_fold_idx].append([smpl])
            fold_curr_rels[alloc_fold_idx] += num_smpl_rels[i]
            fold_total_rels[alloc_fold_idx] += num_smpl_rels[i]
        # for i, fold in enumerate(grouped_folds):
        #     print(
        #         f'fold {i}: {len(fold)} samples, '
        #         f'{fold_total_rels[i]} rels'
        #     )
    else:
        raise ValueError(
            f'Invalid value for generate_splits: {generate_splits}'
        )

    for i in range(num_folds):
        # distribute samples to train/dev/test splits
        # (train and dev eaach one fold, test the remaining folds)
        # (when assigning samples to folds, unpack ppr groups)
        train = []
        dev = []
        test = []
        for j in range(num_folds):
            if j == i:
                # current fold, add to dev
                for ppr_paras in grouped_folds[j]:
                    dev.extend(ppr_paras)
            elif j == (i + 1) % num_folds:
                # next fold, add to test
                for ppr_paras in grouped_folds[j]:
                    test.extend(ppr_paras)
            else:
                # other folds, add to train
                for ppr_paras in grouped_folds[j]:
                    train.extend(ppr_paras)
        # print split stats
        print(f'= = = = = fold {i} = = = = =')
        print(f'train: {len(train)} samples')
        print(f'dev: {len(dev)} samples')
        print(f'test: {len(test)} samples')
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
    # set word limit (optional)
    cap_num_words = False
    # check command line arguments
    if len(sys.argv) not in [2, 3, 4]:
        print(
            'Usage: python hyperpie/data/convert_plmarker.py '
            '/path/to/annots.jsonl '
            '[generate_splits_doc|generate_splits_cls|loose_matching]'
        )
        sys.exit(1)
    annots_path = sys.argv[1]
    generate_splits = False
    loose_matching = False
    if len(sys.argv) >= 3:
        if 'generate_splits_doc' in sys.argv[2:]:
            generate_splits = 'doc'
        if 'generate_splits_cls' in sys.argv[2:]:
            generate_splits = 'cls'
        if 'loose_matching' in sys.argv[2:]:
            loose_matching = True
    convert(annots_path, generate_splits, loose_matching, cap_num_words)
