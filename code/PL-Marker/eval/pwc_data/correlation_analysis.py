import argparse
import json
import os
from collections import defaultdict


def load_pediction_data(root_dir):
    """ Get arxiv ids of papers for which we extracted
        hyperparameter info.
    """

    pred_pprs = defaultdict(list)

    nerre_pred_fn = 'merged_preds.jsonl'
    with open(os.path.join(root_dir, nerre_pred_fn), 'r') as f:
        for line in f.readlines():
            para = json.loads(line)
            doc_key = para['doc_key']
            arxiv_id, para_uuid = doc_key.split('-', maxsplit=1)
            pred_pprs[arxiv_id].append(para)

    return pred_pprs


def _get_number_of_entities(pred_para):
    n = 0
    for ner_sent in pred_para['predicted_ner']:
        n += len(ner_sent)
    return n


def _get_number_of_relations(pred_para):
    n = 0
    for rel_sent in pred_para['predicted_relations']:
        n += len(rel_sent)
    return n


def _get_number_of_ent_type(etyp, pred_para):
    n = 0
    for ner_sent in pred_para['predicted_ner']:
        for pred in ner_sent:
            if pred[2] == etyp:
                n += 1
    return n


def _get_number_of_tups(pred_para):
    vp_rels = dict()
    pa_rels = dict()

    # iterate over sentences
    para_delta = 0
    for sent_idx, sent in enumerate(pred_para['sentences']):
        ner_pred = pred_para['predicted_ner'][sent_idx]
        ner_gold = pred_para['ner'][sent_idx]
        offs_to_label = {}
        # predicted NER
        for (start, end, label) in ner_pred:
            offs_to_label[(start, end)] = label
        # "gold" (dist sup) NER
        for (start, end, label) in ner_gold:
            offs_to_label[(start, end)] = label
        if 'predicted_relations' in pred_para:
            re_pred = pred_para['predicted_relations'][sent_idx]
            for (
                start_from, end_from, start_to, end_to, label
            ) in re_pred:
                label_from = offs_to_label.get((start_from, end_from))
                label_to = offs_to_label.get((start_to, end_to))
                if label_from == 'v' and label_to == 'p':
                    vp_rels[(start_from, end_from)] = (start_to, end_to)
                elif label_from == 'p' and label_to == 'a':
                    pa_rels[(start_from, end_from)] = (start_to, end_to)
        para_delta += len(sent)

    vpa_trips = []
    for from_v, to_p in vp_rels.items():
        for from_p, to_a in pa_rels.items():
            if to_p == from_p:
                vpa_trips.append((from_v, to_p, to_a))

    num_vp_rels = len(vp_rels)
    num_pa_rels = len(pa_rels)
    num_vpa_trips = len(vpa_trips)

    return num_vp_rels, num_pa_rels, num_vpa_trips


def analyse(ppr_preds, ppr_data):
    for arxiv_id, ppr_pred in ppr_preds.items():
        print(f'paper: {arxiv_id}')
        print(f'num paras: {len(ppr_pred)}')
        n_ents = 0
        n_rels = 0
        n_v = 0
        n_p = 0
        n_a = 0
        n_vp = 0
        n_pa = 0
        n_vpa = 0
        for pred_para in ppr_pred:
            n_ents += _get_number_of_entities(pred_para)
            n_rels += _get_number_of_relations(pred_para)
            n_v += _get_number_of_ent_type('v', pred_para)
            n_p += _get_number_of_ent_type('p', pred_para)
            n_a += _get_number_of_ent_type('a', pred_para)
            vp, pa, vpa = _get_number_of_tups(pred_para)
            n_vp += vp
            n_pa += pa
            n_vpa += vpa
        print(
            f'num ents: {n_ents}\n'
            f'num rels: {n_rels}\n'
            f'num v: {n_v}\n'
            f'num p: {n_p}\n'
            f'num a: {n_a}\n'
            f'num vp: {n_vp}\n'
            f'num pa: {n_pa}\n'
            f'num vpa: {n_vpa}\n'
        )
        input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_data_root_dir', type=str)
    args = parser.parse_args()

    # load prediction data
    ppr_preds = load_pediction_data(args.prediction_data_root_dir)

    # load external paper metadata
    with open(f'merged_data.json', 'r') as f:
        ppr_data = json.load(f)

    analyse(ppr_preds, ppr_data)
