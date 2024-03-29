""" Simple relation extraction bsed on entity types and relative distance.
"""

import json
import os
import pickle
import sys
import numpy as np
import torch
from functools import lru_cache
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from transformers import AutoTokenizer, AutoModel


def emb_mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return sum_embeddings / sum_mask


@lru_cache(maxsize=1000)
def get_token_embeddings(tokenizer, model, tokens):
    text = ' '.join(tokens)
    encoded_input = tokenizer(text, return_tensors='pt')
    model_output = model(**encoded_input)
    mean_emb = emb_mean_pooling(model_output, encoded_input['attention_mask'])

    return mean_emb


def prep_para_data(
    doc_key, sents, ner, rel, tokenizer, emb_map, model, smpl_offset=0
):
    """ Given a paragrapgh’s sentences w/ NER and REL info,
        generate X and y for training as well as a mapping
        from sample to entity offset pair.
    """

    X = []
    y = []
    pairs = []
    have_rel = []
    s2e_map = {}

    # collect entity pairs
    sent_offset = 0
    max_sent_len = 0
    for sent_idx in range(len(sents)):
        # get tokens, entities, and relations for this sentence
        tkns = sents[sent_idx]
        ents = ner[sent_idx]
        rels = rel[sent_idx]
        # get all entity candidate pairs
        for ei, ent_from_raw in enumerate(ents):
            for ej, ent_to_raw in enumerate(ents):
                if ei == ej:
                    # no self-relations
                    continue
                # only collect raw information here, do determine
                # derivative features such as relative distance later
                ent_from = {
                    'tokens': tkns[
                        ent_from_raw[0]-sent_offset:
                        ent_from_raw[1]+1-sent_offset
                    ],
                    'start': ent_from_raw[0],
                    'end': ent_from_raw[1],
                    'type': ent_from_raw[2],
                }
                ent_to = {
                    'tokens': tkns[
                        ent_to_raw[0]-sent_offset:
                        ent_to_raw[1]+1-sent_offset
                    ],
                    'start': ent_to_raw[0],
                    'end': ent_to_raw[1],
                    'type': ent_to_raw[2],
                }
                pairs.append((ent_from, ent_to))
                # check if there is a relation between these two entities
                rel_check = [
                    ent_from_raw[0],
                    ent_from_raw[1],
                    ent_to_raw[0],
                    ent_to_raw[1],
                    'USED-FOR'
                ]
                have_rel.append(rel_check in rels)
                # save mapping from sample index to entity offset pair
                # (to be able to create prediction output when used on
                #  unlabeled data)
                curr_sample_idx = len(pairs)-1+smpl_offset
                s2e_map[curr_sample_idx] = {
                    'doc_key': doc_key,
                    'sent_idx': sent_idx,
                    'rel': rel_check,
                }
        sent_offset += len(tkns)
        max_sent_len = max(max_sent_len, len(tkns))

    # process entity pair features
    ent_dist_range = [-max_sent_len, max_sent_len]
    ent_type_one_hot = {
        'a': [1, 0, 0, 0],
        'p': [0, 1, 0, 0],
        'v': [0, 0, 1, 0],
        'c': [0, 0, 0, 1],
    }
    for smpl_idx in range(len(pairs)):
        # prepare entity features
        ent_from, ent_to = pairs[smpl_idx]
        from_type_hot = ent_type_one_hot[ent_from['type']]
        to_type_hot = ent_type_one_hot[ent_to['type']]
        # map entity distance to [0, 1]
        rel_dist = ent_to['start'] - ent_from['end']
        rel_dist_norm = np.interp(rel_dist, ent_dist_range, [0, 1])
        # get token embeddings for entity pair
        tup = tuple(ent_from['tokens'] + ent_to['tokens'])
        if tup in emb_map:
            pair_emb = emb_map[tup]
        else:
            pair_emb = get_token_embeddings(
                tokenizer, model, tup
            )
            emb_map[tup] = pair_emb
        # flatten and de-emphasize
        pair_emb_flat = pair_emb.detach().numpy().flatten() * 0.01
        sample = from_type_hot + to_type_hot + [rel_dist_norm] \
            + pair_emb_flat.tolist()
        # map prediction label from True/False to 1/0
        label = int(have_rel[smpl_idx])
        X.append(sample)
        y.append(label)

    return X, y, s2e_map


@lru_cache(maxsize=1)
def _get_saved_embeddings_map():
    fn = 'ffnre_saved_embs.pkl'
    emb_map = {}
    if os.path.isfile(fn):
        with open(fn, 'rb') as f:
            emb_map = pickle.load(f)
    return emb_map


def _save_embeddings_map(emb_map):
    fn = 'ffnre_saved_embs.pkl'
    with open(fn, 'wb') as f:
        pickle.dump(emb_map, f)
    _get_saved_embeddings_map.cache_clear()


def _find_tokens(tokens, sentence):
    matches = []
    for i in range(len(sentence)):
        if sentence[i] == tokens[0] and sentence[i:i+len(tokens)] == tokens:
            matches.append([i, i+len(tokens)])
    return matches


def dist_supervision_ext(paras):
    # collect entity mentions
    artifacts = set()
    params = set()
    values = set()
    contexts = set()
    # print('searching for entities')
    for para in paras:
        para_delta = 0
        # iterate over sentences
        for sent_idx, sent in enumerate(para['sentences']):
            ner_true = para['ner'][sent_idx]
            for (start, end, label) in ner_true:
                tkns = sent[start-para_delta:end-para_delta+1]
                if label == 'a':
                    artifacts.add(tuple(tkns))
                elif label == 'p':
                    params.add(tuple(tkns))
                elif label == 'v':
                    values.add(tuple(tkns))
                elif label == 'c':
                    contexts.add(tuple(tkns))
            para_delta += len(sent)
    # print(f'found artifacts: {artifacts}')
    # input()
    # apply distance supervision labels
    new_paras = []
    for para in paras:
        # print('new para w/ #pred_ners:')
        # print(sum([
        #     len(s) for s in para['predicted_ner']
        # ]))
        # input()
        para_delta = 0
        # iterate over sentences
        for sent_idx, sent in enumerate(para['sentences']):
            # print(f'looking in sentence: {sent}')
            # input()

            # - - - arrf - - -

            # find known artifacts
            artf_matches = []
            for artf_mention_tup in artifacts:
                artf_mention = list(artf_mention_tup)
                # print(f'looking for {artf_mention}')
                artf_matches.extend(
                    _find_tokens(artf_mention, sent)
                )
            # print(f'found {artf_matches}')
            # input()
            # add artifacts to entity list
            ner_pred = para['predicted_ner'][sent_idx]
            for artf_offset in artf_matches:
                ner_entry = artf_offset + ['a']
                if ner_entry not in ner_pred:
                    # print(f'{ner_entry} is new, adding')
                    ner_pred.append(ner_entry)

            # - - - param - - -

            # find known params
            para_matches = []
            for para_mention_tup in params:
                para_mention = list(para_mention_tup)
                # print(f'looking for {artf_mention}')
                para_matches.extend(
                    _find_tokens(para_mention, sent)
                )
            # print(f'found {artf_matches}')
            # input()
            # add artifacts to entity list
            ner_pred = para['predicted_ner'][sent_idx]
            for para_offset in para_matches:
                par_entry = para_offset + ['a']
                if par_entry not in ner_pred:
                    # print(f'{ner_entry} is new, adding')
                    ner_pred.append(par_entry)

        new_paras.append(para)
        # print('updated para w/ #pred_ners:')
        # print(sum([
        #     len(s) for s in para['predicted_ner']
        # ]))
    # input()
    return new_paras


def eval_model(
        train_fp, test_fp, output_fp, base_dir=None,
        verbose=False
):
    if verbose:
        print('loading data')
    with open(train_fp, 'r') as f:
        train_paras = [json.loads(line) for line in f.readlines()]
    with open(test_fp, 'r') as f:
        test_paras = [json.loads(line) for line in f.readlines()]

    # if verbose:
    #     print('distant supervision ...')
    # test_paras = dist_supervision_ext(test_paras)
    # if verbose:
    #     print('done')

    # device = torch.device('cuda')

    tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-uncased',
    )
    emb_map = _get_saved_embeddings_map()
    model = AutoModel.from_pretrained(
        'bert-base-uncased',
    )

    # model.to(device)

    X_train = []
    y_train = []
    for para in tqdm(train_paras, desc='preparing train data'):
        X, y, s2e_map = prep_para_data(
            para['doc_key'],
            para['sentences'],
            para['ner'],
            para['relations'],
            tokenizer,
            emb_map,
            model
        )
        X_train += X
        y_train += y
    if verbose:
        print(f'loaded {len(X_train)} training samples')

    X_test = []
    y_test = []
    sample_idx_to_entity_offset_pair = {}
    for para in tqdm(test_paras, desc='preparing test data'):
        global_smpl_offset = len(y_test)
        X, y, s2e_map = prep_para_data(
            para['doc_key'],
            para['sentences'],
            para['predicted_ner'],
            para['relations'],
            tokenizer,
            emb_map,
            model,
            global_smpl_offset
        )
        X_test += X
        y_test += y
        sample_idx_to_entity_offset_pair.update(s2e_map)
    if verbose:
        print(f'loaded {len(X_test)} test samples')
        print(f'noted {len(sample_idx_to_entity_offset_pair)} sample offsets')

    _save_embeddings_map(emb_map)

    # if not loaded_train_data:
    #     print(f'saving tokenized training data')
    #     with open(X_train_tkn_fp, 'wb') as f:
    #         pickle.dump(X_train, f)
    #     with open(y_train_tkn_fp, 'wb') as f:
    #         pickle.dump(y_train, f)
    # if not loaded_test_data:
    #     print(f'saving tokenized test data')
    #     with open(X_test_tkn_fp, 'wb') as f:
    #         pickle.dump(X_test, f)
    #     with open(y_test_tkn_fp, 'wb') as f:
    #         pickle.dump(y_test, f)
    #     with open(smpl2off_fp, 'w') as f:
    #         json.dump(sample_idx_to_entity_offset_pair, f)

    # train model

    # dimensions
    # type_from (one-hot): 4
    # type_to (one-hot): 4
    # rel_dist_norm: 1
    # pair_emb: 768
    clf = MLPClassifier(
        hidden_layer_sizes=(300, 100, 25, 2),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=90,
        random_state=1,
        shuffle=True,
        verbose=verbose
    )
    # clf = MLPClassifier(
    #     hidden_layer_sizes=(300, 100, 25, 2),
    #     activation='relu',
    #     solver='adam',
    #     learning_rate_init=0.001,
    #     max_iter=90,
    #     random_state=1,
    #     shuffle=True,
    #     verbose=verbose
    # )
    if verbose:
        print('training model...')
    clf.fit(X_train, y_train)

    # evaluate model
    if verbose:
        print('evaluating model...')
    y_pred = clf.predict(X_test)
    # # bit mask filter
    # y_pred = [
    #     int(
    #         bool(y) and
    #         (
    #             X_test[i][:8] in [
    #                 [0, 1, 0, 0, 1, 0, 0, 0],
    #                 [0, 0, 1, 0, 0, 1, 0, 0],
    #                 [0, 0, 0, 1, 0, 0, 1, 0]
    #             ]
    #         )
    #     )
    #     for i, y in enumerate(y_pred)
    # ]
    print(classification_report(y_test, y_pred, zero_division=0.0))
    res = classification_report(
        y_test, y_pred, zero_division=0.0, output_dict=True
    )
    with open(output_fp, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'wrote results to {output_fp}')

    if base_dir is not None:
        print(
            f'writing sample_idx_to_entity_offset_pair.json'
            f' and y_pred.json into {base_dir}'
        )
        smpl2idx_fp = os.path.join(
            base_dir,
            'sample_idx_to_entity_offset_pair.json'
        )
        with open(smpl2idx_fp, 'w') as f:
            json.dump(sample_idx_to_entity_offset_pair, f)
        pred_out_fp = os.path.join(
            base_dir,
            'y_pred.json'
        )
        with open(pred_out_fp, 'w') as f:
            json.dump(y_pred.tolist(), f)


if __name__ == '__main__':
    if len(sys.argv) not in [2, 4]:
        print(
            'Usage: python ffnn_re.py <train.jsonl> <test.jsonl>'
            ' <output.jsonl>'
        )
        sys.exit(1)
    if len(sys.argv) == 2:
        base_dir = sys.argv[1]
        train_fp = os.path.join(base_dir, 'train.jsonl')
        test_fp = os.path.join(base_dir, 'merged_preds.jsonl')
        output_fp = os.path.join(base_dir, 'ffnn_re_results.jsonl')
    elif len(sys.argv) == 4:
        train_fp = sys.argv[1]
        test_fp = sys.argv[2]
        output_fp = sys.argv[3]
        base_dir = None
    eval_model(train_fp, test_fp, output_fp, base_dir=base_dir, verbose=True)
