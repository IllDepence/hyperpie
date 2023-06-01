import datetime
import openai
from collections import OrderedDict
from hyperpie import settings
from hyperpie.data import llm_cache


def openai_api(para, prompt, params=None, verbose=False):
    """ Get LLM completion for `prompt` using OpenAI API.

        If the prompt has been used before, the completion will be
        loaded from cache.
    """

    if params is None:
        params = settings.gpt_default_params

    # check cache
    cached_completion = llm_cache.llm_completion_cache_load(
        params, prompt
    )
    if cached_completion is not None:
        if verbose:
            print('loading completion from cache')
        return cached_completion
    if verbose:
        print('completion not in cache')

    # not in cache, get from API
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    completion = openai.Completion.create(prompt=prompt, **params)

    completion_dict = OrderedDict({
        "timestamp": timestamp,
        "params": params,
        "paragraph": para,
        "prompt": prompt,
        "completion": completion
    })

    # save to cache
    llm_cache.llm_completion_cache_save(
        params, prompt, completion_dict
    )
    if verbose:
        print('saving completion to cache')

    return completion_dict
