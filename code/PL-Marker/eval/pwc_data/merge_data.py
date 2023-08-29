""" Merge
    - prediction data
    - Papers with Code data
    - Github repo data
"""

import argparse
import json
import os
import requests
import time
import sys


def get_pediction_arxiv_ids(root_dir):
    """ Get arxiv ids of papers for which we extracted
        hyperparameter info.
    """

    arxiv_ids = set()

    ner_pred_fp = os.path.join(
        'all',
        'ent_pred_test.json'
    )
    with open(os.path.join(root_dir, ner_pred_fp), 'r') as f:
        for line in f.readlines():
            para = json.loads(line)
            doc_key = para['doc_key']
            arxiv_id, para_uuid = doc_key.split('-', maxsplit=1)
            assert len(para_uuid) == 36
            arxiv_ids.add(arxiv_id)

    return arxiv_ids


def get_pwc_data(pred_arxiv_ids):
    """ Get repo URLs from PwC data.
    """

    pwc_data_fw = 'links-between-papers-and-code.json'
    pwc_data = {}
    pwc_data_pred = {}

    with open(pwc_data_fw, 'r') as f:
        pprs = json.load(f)
        for ppr in pprs:
            arxiv_id = ppr['paper_arxiv_id']
            if arxiv_id is not None:
                pwc_data[arxiv_id] = ppr

    for arxiv_id in pred_arxiv_ids:
        pwc_data_pred[arxiv_id] = None
        if arxiv_id in pwc_data:
            pwc_data_pred[arxiv_id] = pwc_data[arxiv_id]

    return pwc_data_pred


def get_github_medatada(pwc_data_pred):
    """ Get repo data from Github API.
    """

    ext_data = {}

    ext_data_tmp_fp = 'merged_data.json'
    if os.path.exists(ext_data_tmp_fp):
        with open(ext_data_tmp_fp, 'r') as f:
            ext_data_prev = json.load(f)
        print(f'loaded {len(ext_data_prev)} already crawled entries')

    for i, (arxiv_id, ppr) in enumerate(pwc_data_pred.items()):
        print(f'{i+1}/{len(pwc_data_pred)}')
        ext_data[arxiv_id] = None
        if ppr is None:
            continue
        if arxiv_id in ext_data_prev:
            print(f'already have data for {arxiv_id}, skipping')
            ext_data[arxiv_id] = ext_data_prev[arxiv_id]
            continue
        ext_data[arxiv_id] = ppr
        ext_data[arxiv_id]['github_metadata'] = None
        repo_url = ppr['repo_url']
        if repo_url is None:
            print(f'no repo url for {arxiv_id}, skipping')
            continue
        assert repo_url.startswith('https://github.com/')
        repo_api_url = repo_url.replace(
            'https://github.com/',
            'https://api.github.com/repos/'
        )
        resp = requests.get(
            repo_api_url,
            headers={
                'Accept': 'application/json',
                'X-GitHub-Api-Version': '2022-11-28'
            }
        )
        if resp.status_code == 404:
            print(f'got 404 for {arxiv_id}, skipping')
            continue
        if resp.status_code != 200:
            print(f'response status code {resp.status_code} for {arxiv_id}')
            print(f'exiting...')
            sys.exit()
        else:
            repo_data = resp.json()
            ext_data[arxiv_id]['github_metadata'] = repo_data

            with open(ext_data_tmp_fp, 'w') as f:
                json.dump(ext_data, f)

        print(f'waiting 30 seconds...')
        time.sleep(30)

    return ext_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_data_root_dir', type=str)
    args = parser.parse_args()

    arxiv_ids = get_pediction_arxiv_ids(args.prediction_data_root_dir)
    print(f'loaded {len(arxiv_ids)} arxiv ids')
    pwc_data_pred = get_pwc_data(arxiv_ids)
    print(
        f'found PwC data for '
        f'{len([d for d in pwc_data_pred.values() if d is not None])} '
        f'papers'
    )
    ext_data = get_github_medatada(pwc_data_pred)

    with open(f'merged_data_final.json', 'w') as f:
        json.dump(ext_data, f)
