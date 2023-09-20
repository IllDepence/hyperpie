""" Cache LLm completions

    {
        "md5_hash_of_param_str": {
            "md5_hash_of_prompt": {
                <completion>
            },
            ...
        },
        ...
    }

    Cache is keyed by the MD5 hash of a string representation of
    the parameters used to generate the completion.

    Prompts are keyed by the MD5 hash of the prompt string

    Each completion is a dictionary with the following keys:
        - timestamp: datetime string
        - params: dictionary of parameters used to generate the completion
        - completion: the completion object returned by the OpenAI API

"""

import os
import json
import re
from hashlib import md5
from hyperpie import settings
from functools import lru_cache


def _param_str(param_dict):
    """ Return a string representation of a dictionary of parameters.
    """

    # create list of <key>=<value> strings
    # (ensures consistent ordering)
    sorted_param_strs = sorted(
        [
            f'{k}={v}'
            for k, v in
            param_dict.items()
        ]
    )

    # join to a single string
    joint_param_str = '|'.join(sorted_param_strs)

    return joint_param_str


def llm_completion_cache_save(
    params, prompt, completion
):
    """ Save LLM completions to cache and
        clear cached version in memory.
    """

    model_key = md5(
        _param_str(params).encode('utf-8')
    ).hexdigest()
    prompt_key = md5(
        prompt.encode('utf-8')
    ).hexdigest()

    # get existing cache
    cache = _load_cache()

    # create entry for params if not exists
    if model_key not in cache:
        cache[model_key] = {}

    # create entry for prompt (if exists, overwrite)
    cache[model_key][prompt_key] = completion

    # save cache
    with open(settings.llm_cache_fp, 'w') as f:
        json.dump(cache, f)

    # clear cache in memory
    _load_cache.cache_clear()


@lru_cache(maxsize=1)
def _load_cache():
    """ Load LLM cache and keep in memory until new data
        is saved.
    """

    cache = {}

    # get existing cache if exists
    if os.path.exists(settings.llm_cache_fp):
        with open(settings.llm_cache_fp) as f:
            cache = json.load(f)

    return cache


def llm_completion_cache_load(
    params, prompt
):
    """ Load cached LLM completions if exists.
    """

    model_key = md5(
        _param_str(params).encode('utf-8')
    ).hexdigest()
    prompt_key = md5(
        prompt.encode('utf-8')
    ).hexdigest()

    # get existing cache if exists
    cache = _load_cache()

    # return completion if exists
    if model_key in cache and prompt_key in cache[model_key]:
        return cache[model_key][prompt_key]

    # otherwise return None
    return None


def load_external(subdir, doc_id, para_idx):
    """ Load external LLM completion from file.

        (Used to evaluation models where completions
         were generated  using code outside of this
         package.)
    """

    fp = os.path.join(
        settings.ext_cache_dir,
        subdir,
        f'{doc_id}-{para_idx}.txt'
    )
    with open(fp) as f:
        completion = f.read()

    # excape backslashes
    completion = completion.replace('\\', '\\\\')

    # fix entity existence key
    if '_json' in subdir:
        wrong_key_patt = re.compile(
            r'"has_entities":\s+(true|false),$',
            re.M
        )
        completion = wrong_key_patt.sub(
            r'"text_contains_entities": \1,',
            completion
        )
    elif '_yaml' in subdir:
        wrong_key_patt = re.compile(
            r'has_entities:\s+(true|false)$',
            re.M
        )
        completion = wrong_key_patt.sub(
            r'text_contains_entities: \1',
            completion
        )

    return completion
