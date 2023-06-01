import datetime
import openai
from collections import OrderedDict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from hyperpie import settings
from hyperpie.data import llm_cache


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(9))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def openai_api(para, prompt, params=None, verbose=False):
    """ Get LLM completion for `prompt` using OpenAI API.

        If the prompt has been used before, the completion will be
        loaded from cache.

        Returns:
        <completion_dict>, <from_cache>
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
        return cached_completion, True
    if verbose:
        print('completion not in cache')

    # not in cache, get from API
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    completion = completion_with_backoff(prompt=prompt, **params)

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

    return completion_dict, False


def get_completion_text(completion):
    """ Get completion text from completion dict.
    """

    return completion['completion']['choices'][0]['text']
