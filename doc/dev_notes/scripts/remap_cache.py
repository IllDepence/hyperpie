""" Remap cache keys
"""

import json
import sys
import hyperpie as hp
from hashlib import md5


def keyswap(
    cache, prompt_old, prompt_fixed, new_cache, para_fixed_text
):

    model_key = md5(
        hp.data.llm_cache._param_str(
            hp.settings.gpt_default_params
        ).encode('utf-8')
    ).hexdigest()
    prompt_key_old = md5(
        prompt_old.encode('utf-8')
    ).hexdigest()
    prompt_key_fixed = md5(
        prompt_fixed.encode('utf-8')
    ).hexdigest()

    cached_completion = cache[model_key][prompt_key_old]

    # set new para text and prompt in completion
    cached_completion['paragraph']['text'] = para_fixed_text
    cached_completion['prompt'] = prompt_fixed

    # create entry for params if not exists
    if model_key not in new_cache:
        new_cache[model_key] = {}

    # create entry for prompt (if exists, overwrite)
    new_cache[model_key][prompt_key_fixed] = cached_completion

    return cached_completion


paras_fixed = hp.data.load.load_annotated()  # post line-break fix
paras_old_fp = sys.argv[1]
with open(paras_old_fp, 'r') as f:
    paras_old = json.load(f)  # pre line-break fix

with open(hp.settings.llm_cache_fp) as f:
    cache = json.load(f)

new_cache = {}

for i, para_fixed in enumerate(paras_fixed):
    print(f'{i}/{len(paras_fixed)}')

    para_old = paras_old[i]

    assert para_fixed['document_id'] == para_old['document_id']
    assert para_fixed['paragraph_index'] == para_old['paragraph_index']

    # signle step
    prompt0_old = hp.llm.prompt_templates.text_e2e.format(
        text=para_old['text']
    )
    prompt0_fixed = hp.llm.prompt_templates.text_e2e.format(
        text=para_fixed['text']
    )
    _ = keyswap(
        cache, prompt0_old, prompt0_fixed, new_cache,
        para_fixed['text']
    )

    # prompt (stage 1)
    prompt1_old = hp.llm.prompt_templates.text_e2e_fillin_twostep_1.format(
        text=para_old['text']
    )
    prompt1_fixed = hp.llm.prompt_templates.text_e2e_fillin_twostep_1.format(
        text=para_fixed['text']
    )
    completion_dict1 = keyswap(
        cache, prompt1_old, prompt1_fixed, new_cache,
        para_fixed['text']
    )

    # prompt (stage 2)
    prompt2_fixed = hp.llm.prompt_templates.text_e2e_fillin_twostep_2.format(
        yaml=completion_dict1['completion']['choices'][0]['text'],
        text=para_fixed['text']
    )
    prompt2_old = hp.llm.prompt_templates.text_e2e_fillin_twostep_2.format(
        yaml=completion_dict1['completion']['choices'][0]['text'],
        text=para_old['text']
    )
    _ = keyswap(
        cache, prompt2_old, prompt2_fixed, new_cache,
        para_fixed['text']
    )

# save new cache
new_cache_fp = '/tmp/llm_cache_fixed.json'
with open(new_cache_fp, 'w') as f:
    json.dump(new_cache, f)
