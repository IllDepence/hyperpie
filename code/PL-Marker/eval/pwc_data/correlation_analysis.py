import argparse
import json
import os
import sys
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


def _get_ppr_pred_hyperparam_digest(ppr_pred):
    digest = defaultdict(int)
    for pred_para in ppr_pred:
        digest['n_ents'] += _get_number_of_entities(pred_para)
        digest['n_rels'] += _get_number_of_relations(pred_para)
        digest['n_v'] += _get_number_of_ent_type('v', pred_para)
        digest['n_p'] += _get_number_of_ent_type('p', pred_para)
        digest['n_a'] += _get_number_of_ent_type('a', pred_para)
        vp, pa, vpa = _get_number_of_tups(pred_para)
        digest['n_vp'] += vp
        digest['n_pa'] += pa
        digest['n_vpa'] += vpa
    return digest


def _get_ppr_data_digest(ppr_data):
    gh_meta = ppr_data.get('github_metadata')
    digest = {
        'has_repo_url': ppr_data['repo_url'] is not None,
        'mentioned_in_ppaer': ppr_data['mentioned_in_paper'],
        'mentioned_in_github': ppr_data['mentioned_in_github'],
        'has_repo_data':  gh_meta is not None,
        'repo_has_hoomepage': False,
        'repo_size': 0,
        'repo_stars': 0,
        'repo_watchers': 0,
        'repo_has_issues': False,
        'repo_has_projects': False,
        'repo_has_downloads': False,
        'repo_has_wiki': False,
        'repo_has_pages': False,
        'repo_forks': 0,
        'repo_archived': False,
        'repo_disabled': False,
        'repo_open_issues': 0,
        'repo_has_license': False
    }
    if gh_meta is not None:
        digest['repo_has_homepage'] = gh_meta['homepage'] is not None
        digest['repo_size'] = gh_meta['size']
        digest['repo_stars'] = gh_meta['stargazers_count']
        digest['repo_watchers'] = gh_meta['watchers_count']
        digest['repo_has_issues'] = gh_meta['has_issues']
        digest['repo_has_projects'] = gh_meta['has_projects']
        digest['repo_has_downloads'] = gh_meta['has_downloads']
        digest['repo_has_wiki'] = gh_meta['has_wiki']
        digest['repo_has_pages'] = gh_meta['has_pages']
        digest['repo_forks'] = gh_meta['forks_count']
        digest['repo_archived'] = gh_meta['archived']
        digest['repo_disabled'] = gh_meta['disabled']
        digest['repo_open_issues'] = gh_meta['open_issues_count']
        digest['repo_has_license'] = gh_meta['license'] is not None
    return digest


def analyse(ppr_preds, ppr_datas):
    for arxiv_id in ppr_preds.keys():
        print(f'paper: {arxiv_id}')
        ppr_pred = ppr_preds[arxiv_id]
        ppr_data = ppr_datas.get(arxiv_id)
        if ppr_data is None:
            print(f'paper data not found')
            x = input('press enter to skip, "q" to quit')
            if x == 'q':
                sys.exit()
            continue
        print(f'num paras: {len(ppr_pred)}')
        hyper_digest = _get_ppr_pred_hyperparam_digest(ppr_pred)
        repro_digest = _get_ppr_data_digest(ppr_data)
        print(
            f'- - - hyperr - - -\n'
            f'{json.dumps(hyper_digest, indent=2)}\n'
            f'- - - repro- - -\n'
            f'{json.dumps(repro_digest, indent=2)}\n'
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
