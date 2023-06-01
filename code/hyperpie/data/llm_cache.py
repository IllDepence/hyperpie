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
from hashlib import md5
from hyperpie import settings


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
    """ Save LLM completions to cache.
    """

    cache = {}
    model_key = md5(
        _param_str(params).encode('utf-8')
    ).hexdigest()
    prompt_key = md5(
        prompt.encode('utf-8')
    ).hexdigest()

    # get existing cache if exists
    if os.path.exists(settings.llm_cache_fp):
        with open(settings.llm_cache_fp) as f:
            cache = json.load(f)

    # create entry for params if not exists
    if model_key not in cache:
        cache[model_key] = {}

    # create entry for prompt (if exists, overwrite)
    cache[model_key][prompt_key] = completion

    # save cache
    with open(settings.llm_cache_fp, 'w') as f:
        json.dump(cache, f)


def llm_completion_cache_load(
    params, prompt
):
    """ Load cached LLM completions if exists.
    """

    cache = {}
    model_key = md5(
        _param_str(params).encode('utf-8')
    ).hexdigest()
    prompt_key = md5(
        prompt.encode('utf-8')
    ).hexdigest()

    # get existing cache if exists
    if os.path.exists(settings.llm_cache_fp):
        with open(settings.llm_cache_fp) as f:
            cache = json.load(f)

    # return completion if exists
    if model_key in cache and prompt_key in cache[model_key]:
        return cache[model_key][prompt_key]

    # otherwise return None
    return None
