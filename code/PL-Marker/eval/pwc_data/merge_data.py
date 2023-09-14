""" Merge
    - prediction data
    - Papers with Code data
    - Github repo data
"""

import argparse
import json
import os
import time
import sys
from functools import lru_cache
from github import Github, Auth, GithubException


def get_pediction_arxiv_ids(pred_fp):
    """ Get arxiv ids of papers for which we extracted
        hyperparameter info.
    """

    arxiv_ids = set()

    with open(pred_fp, 'r') as f:
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


@lru_cache(maxsize=1)
def _get_cached_github_metadata():
    if not os.path.exists('gh_cache.json'):
        return dict()
    with open('gh_cache.json', 'r') as f:
        gh_cache = json.load(f)
    return gh_cache


def _save_cached_github_metadata(gh_cache):
    with open('gh_cache.json', 'w') as f:
        json.dump(gh_cache, f)
    _get_cached_github_metadata.cache_clear()


def _get_github_medatada(repo_url, gh_api):

    assert repo_url.startswith('https://github.com/')
    repo_name = repo_url.replace(
        'https://github.com/',
        ''
    )

    # return from cache if available
    cache = _get_cached_github_metadata()
    if repo_name in cache:
        print('using cached Github metadata for {}'.format(repo_name))
        return cache[repo_name]

    # get from Github API
    try:
        repo = gh_api.get_repo(repo_name)
        # get all metadata that is publicly available
        # through the Github API’s repo object
        repo_metadata = {
            'name': repo.name,
            'full_name': repo.full_name,
            'description': repo.description,
            'created_at': repo.created_at.timestamp(),
            'updated_at': repo.updated_at.timestamp(),
            'pushed_at': repo.pushed_at.timestamp(),
            'homepage': repo.homepage,
            'size': repo.size,
            'stargazers_count': repo.stargazers_count,
            'watchers_count': repo.watchers_count,
            'language': repo.language,
            'forks_count': repo.forks_count,
            'open_issues_count': repo.open_issues_count,
            'default_branch': repo.default_branch,
            'network_count': repo.network_count,
            'subscribers_count': repo.subscribers_count,
            'archived': repo.archived,
            'forks': repo.forks,
            'open_issues': repo.open_issues,
            'has_issues': repo.has_issues,
            'has_projects': repo.has_projects,
            'has_downloads': repo.has_downloads,
            'has_wiki': repo.has_wiki,
            'has_pages': repo.has_pages,
        }
        # additionally get
        # - number closed of issues
        repo_metadata[
            'closed_issues_count'
        ] = repo.get_issues(state='closed').totalCount
        # - total number of pull requests
        repo_metadata['pulls_count'] = repo.get_pulls().totalCount
        # - total number of commits
        repo_metadata['commits_count'] = repo.get_commits().totalCount
        # - total number of contributors
        repo_metadata[
            'contributors_count'
        ] = repo.get_contributors().totalCount
        # - total number of releases
        repo_metadata['releases_count'] = repo.get_releases().totalCount
        # - text of the README file
        repo_metadata[
            'readme'
        ] = repo.get_readme().decoded_content.decode('utf-8')
        repo_metadata['readme_len'] = len(repo_metadata['readme'])
    except GithubException as e:
        if e.status == 404:
            print('GithubException: 404 (not found), skipping')
            return None
        elif e.status == 409:
            print('GithubException: 409 (repo empty), skipping')
            return None
        else:
            print(f'GithubException: {e}')
            sys.exit()

    # save cache with new entry
    cache[repo_name] = repo_metadata
    _save_cached_github_metadata(cache)

    print(f'waiting 4 seconds...')
    time.sleep(4)

    return repo_metadata


def add_github_medatada(pwc_data_pred, access_token):
    """ Get repo data from Github API.
    """

    gh_auth = Auth.Token(access_token)
    gh_api = Github(auth=gh_auth)

    ext_data = {}

    for i, (arxiv_id, ppr) in enumerate(pwc_data_pred.items()):
        print(f'{i+1}/{len(pwc_data_pred)}')
        ext_data[arxiv_id] = None
        if ppr is None:
            continue
        ext_data[arxiv_id] = ppr
        ext_data[arxiv_id]['github_metadata'] = None
        repo_url = ppr['repo_url']
        if repo_url is None:
            print(f'no repo url for {arxiv_id}, skipping')
            continue
        if not repo_url.startswith('https://github.com/'):
            print(f'invalid repo url {repo_url} for {arxiv_id}, skipping')
            continue

        gh_metadata = _get_github_medatada(repo_url, gh_api)

        ext_data[arxiv_id]['github_metadata'] = gh_metadata

    return ext_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_json_fp', type=str)
    args = parser.parse_args()

    arxiv_ids = get_pediction_arxiv_ids(args.prediction_json_fp)
    print(f'loaded {len(arxiv_ids)} arxiv ids')
    pwc_data_pred = get_pwc_data(arxiv_ids)
    print(
        f'found PwC data for '
        f'{len([d for d in pwc_data_pred.values() if d is not None])} '
        f'papers'
    )

    with open('gh_token', 'r') as f:
        access_token = f.read().strip()
    ext_data = add_github_medatada(pwc_data_pred, access_token)

    with open(f'merged_data_final.json', 'w') as f:
        json.dump(ext_data, f)
