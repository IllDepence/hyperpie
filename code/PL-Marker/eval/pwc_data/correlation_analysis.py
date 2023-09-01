import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import pearsonr


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
    digest = {
        'mentioned_in_paper': -1,
        'mentioned_in_github': -1,
        'has_repo_data': -1,
        'repo_has_homepage': -1,
        'repo_size': -1,
        'repo_stars': -1,
        'repo_watchers': -1,
        'repo_has_issues': -1,
        'repo_has_projects': -1,
        'repo_has_downloads': -1,
        'repo_has_wiki': -1,
        'repo_has_pages': -1,
        'repo_forks': -1,
        'repo_archived': -1,
        'repo_open_issues': -1,
        'repo_has_license': -1
    }
    if ppr_data is None:
        return digest
    gh_meta = ppr_data.get('github_metadata')
    digest['mentioned_in_paper'] = int(ppr_data['mentioned_in_paper'])
    digest['mentioned_in_github'] = int(ppr_data['mentioned_in_github'])
    digest['has_repo_data'] = int(gh_meta is not None)
    if gh_meta is not None:
        digest['repo_has_homepage'] = int(gh_meta['homepage'] is not None)
        digest['repo_size'] = gh_meta['size']
        digest['repo_stars'] = gh_meta['stargazers_count']
        digest['repo_watchers'] = gh_meta['watchers_count']
        digest['repo_has_issues'] = int(gh_meta['has_issues'])
        digest['repo_has_projects'] = int(gh_meta['has_projects'])
        digest['repo_has_downloads'] = int(gh_meta['has_downloads'])
        digest['repo_has_wiki'] = int(gh_meta['has_wiki'])
        digest['repo_has_pages'] = int(gh_meta['has_pages'])
        digest['repo_forks'] = gh_meta['forks_count']
        digest['repo_archived'] = int(gh_meta['archived'])
        digest['repo_open_issues'] = gh_meta['open_issues_count']
        digest['repo_has_license'] = int(gh_meta['license'] is not None)
    return digest


def analyse(ppr_preds, ppr_datas):
    # prep lists for overview corr matrix (all vars)
    hyper_repro_data = []
    hyper_repro_keys = []

    # prep lists for selected eval
    # (n_vpa <-> repo_stars / repo_watchers / repo_open_issues)
    n_vpa_vals = []
    repo_stars_vals = []
    log_repo_stars_vals = []
    repo_forks_vals = []
    repo_open_issues_vals = []

    star_threshold = 50

    for arxiv_id in ppr_preds.keys():
        # print(f'paper: {arxiv_id}')
        ppr_pred = ppr_preds[arxiv_id]
        ppr_data = ppr_datas.get(arxiv_id)
        if ppr_data is None:
            # print(f'paper data not found')
            # x = input('press enter to skip, "q" to quit')
            # if x == 'q':
            #     sys.exit()
            continue
        # print(f'num paras: {len(ppr_pred)}')
        hyper_digest = _get_ppr_pred_hyperparam_digest(ppr_pred)
        repro_digest = _get_ppr_data_digest(ppr_data)
        # print(
        #     f'- - - hyperr - - -\n'
        #     f'{json.dumps(hyper_digest, indent=2)}\n'
        #     f'- - - repro- - -\n'
        #     f'{json.dumps(repro_digest, indent=2)}\n'
        # )
        # input()

        # note general vals
        hyper_repro_data.append(
            list(hyper_digest.values()) +
            list(repro_digest.values())
        )
        hyper_repro_keys = list(hyper_digest.keys()) \
            + list(repro_digest.keys())

        if repro_digest['repo_stars'] < star_threshold:
            continue

        # note selected vals
        repo_stars_vals.append(repro_digest['repo_stars'])
        gt0log = lambda x: np.log2(x) if x > 0 else x  # noqa: E731
        log_repo_stars_vals.append(gt0log(repro_digest['repo_watchers']))
        repo_forks_vals.append(repro_digest['repo_forks'])
        repo_open_issues_vals.append(repro_digest['repo_open_issues'])
        n_vpa_vals.append(
            hyper_digest['n_vpa'] +
            hyper_digest['n_vp'] +
            hyper_digest['n_pa']
        )

    # print(len(hyper_data))
    # print(len(repro_data))
    # repro_d_lens = set()
    # for i, rd in enumerate(repro_data):
    #     if len(rd) != 19:
    #         print(i, rd)
    #     repro_d_lens.add(len(rd))
    # print(repro_d_lens)

    # plot correlation matrix between hyper_data and repro_data
    hyper_repro_data = np.array(hyper_repro_data)
    print(hyper_repro_data.shape)
    corr_mat = np.corrcoef(hyper_repro_data, rowvar=False)
    # # make it log scale
    # corr_mat = np.log(corr_mat)
    print(corr_mat.shape)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.matshow(corr_mat)
    ax.set_xticks(np.arange(len(hyper_repro_data[0])))
    ax.set_yticks(np.arange(len(hyper_repro_data[0])))
    ax.set_xticklabels(hyper_repro_keys)
    ax.set_yticklabels(hyper_repro_keys)
    # tilting the x axis labels
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="left",
        rotation_mode="anchor"
    )
    plt.tight_layout()
    plt.show()

    # get Pearson correlation coefficient and p-value for
    # selected eval

    print(f'---===[ star threshold: {star_threshold} ]===---')

    # n_vpa <-> repo_stars
    cc, p = pearsonr(n_vpa_vals, repo_stars_vals)

    # n_vpa <-> repo_stars, rounded to 3 decimal places without leading 0
    print(f'n_vpa <-> repo_stars')
    print(f'{cc:.3f} ({p:.3f})')

    # print(
    #     f'n_vpa <-> repo_stars\n'
    #     f'Pearson correlation coefficient: {cc}\n'
    #     f'p-value: {p}'
    # )
    # n_vpa <-> log(repo_stars)
    cc, p = pearsonr(n_vpa_vals, log_repo_stars_vals)
    # print(
    #     f'n_vpa <-> log(repo_stars)\n'
    #     f'Pearson correlation coefficient: {cc}\n'
    #     f'p-value: {p}'
    # )
    # n_vpa <-> repo_open_issues
    cc, p = pearsonr(n_vpa_vals, repo_open_issues_vals)
    # print(
    #     f'n_vpa <-> repo_open_issues\n'
    #     f'Pearson correlation coefficient: {cc}\n'
    #     f'p-value: {p}'
    # )
    # n_vpa <-> repo_forks
    cc, p = pearsonr(n_vpa_vals, repo_forks_vals)

    # n_vpa <-> repo_forks, rounded to 3 decimal places without leading 0
    print(f'n_vpa <-> repo_forks')
    print(f'{cc:.3f} ({p:.3f})')

    # print(
    #     f'n_vpa <-> repo_forks\n'
    #     f'Pearson correlation coefficient: {cc}\n'
    #     f'p-value: {p}'
    # )

    # scatter plot n_vpa_vals and repo_stars_vals
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        n_vpa_vals, repo_stars_vals,
        alpha=0.5
    )
    ax.set_xlabel('n_vpa')
    ax.set_ylabel('repo_stars')
    plt.show()


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
