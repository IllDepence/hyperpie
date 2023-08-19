""" Simple relation extraction bsed on entity types and relative distance.
"""

import json
import sys
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier


def prep_para_data(doc_key, sents, ner, rel):
    """ Given a paragrapgh’s sentences w/ NER and REL info,
        generate X and y for training as well as a mapping
        from sample to entity offset pair.
    """

    X = []
    y = []
    pairs = []
    have_rel = []
    sample_idx_to_entity_offset_pair = {}

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
                curr_sample_idx = len(pairs)-1
                sample_idx_to_entity_offset_pair[curr_sample_idx] = {
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
        rel_dist_abs = ent_to['start'] - ent_from['end']
        rel_dist_norm = np.interp(rel_dist_abs, ent_dist_range, [0, 1])
        # map prediction label from True/False to 1/0
        label = int(have_rel[smpl_idx])
        # NOTE: consider adding token embeddings or other features
        sample = from_type_hot + to_type_hot + [rel_dist_norm]
        X.append(sample)
        y.append(label)

    return X, y, sample_idx_to_entity_offset_pair


def eval_model(train_fp, test_fp, verbose=False):
    with open(train_fp, 'r') as f:
        train_paras = [json.loads(line) for line in f.readlines()]
    with open(test_fp, 'r') as f:
        test_paras = [json.loads(line) for line in f.readlines()]

    X_train = []
    y_train = []
    sample_idx_to_entity_offset_pair = {}
    for para in train_paras:
        X, y, s2e_map = prep_para_data(
            para['doc_key'],
            para['sentences'],
            para['ner'],
            para['relations']
        )
        X_train += X
        y_train += y
        sample_idx_to_entity_offset_pair.update(s2e_map)
    if verbose:
        print(f'loaded {len(X_train)} training samples')

    X_test = []
    y_test = []
    for para in test_paras:
        X, y, s2e_map = prep_para_data(
            para['doc_key'],
            para['sentences'],
            para['ner'],
            para['relations']
        )
        X_test += X
        y_test += y
        sample_idx_to_entity_offset_pair.update(s2e_map)
    if verbose:
        print(f'loaded {len(X_test)} test samples')

    # train model
    clf = MLPClassifier(
        hidden_layer_sizes=(18, 4, 2),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=10000,
        random_state=1,
        shuffle=True,
        verbose=verbose
    )
    if verbose:
        print('training model...')
    clf.fit(X_train, y_train)

    # evaluate model
    if verbose:
        print('evaluating model...')
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python simple_re.py <train.jsonl> <test.jsonl>')
        sys.exit(1)
    train_fp = sys.argv[1]
    test_fp = sys.argv[2]
    eval_model(train_fp, test_fp, verbose=True)
