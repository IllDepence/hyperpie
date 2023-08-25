""" Evaluate zero-shot setting for GALACTICA with JOSN format.
"""

import re
import json
import random
import hyperpie as hp

assert hp.settings.use_openai_api is False

d = hp.data.load.load_annotated()
paras_true = d

# shuffle
random.Random(42).shuffle(paras_true)

paras_true, filter_stats = hp.data.filter_annots.require_parent(
    d
)
paras_pred = []

from_idx = 0
to_idx = len(paras_true)
params_512 = hp.settings.gpt_default_params.copy()
params_512['max_tokens'] = 512
# # FIXME: ↓ temporary for cache access w/o API endpoint ↓
# params_512['model'] = 'facebook/galactica-120b'
# # FIXME: ↑ temporary for cache access w/o API endpoint ↑
stats_dicts = []
for i, para in enumerate(paras_true[from_idx:to_idx]):
    print(f'{i}/{len(paras_true)}')

    prompt = hp.llm.prompt_templates.text_e2e_fillin_twostep_1_galacitca_json.format(  # noqa
        text=para['text']
    )

    print('TEXT')
    print(para['text'][0:300], '...')

    completion_dict, from_cache = hp.llm.predict.openai_api(
        para, prompt, params=params_512
    )

    print('COMPLETION')
    print(completion_dict['completion']['choices'][0]['text']) # [:300], '...') # noqa
    print('\n\n')

    # para_pred, stats_dict = hp.llm.convert.llm_output2eval_input(
    #     completion_dict,
    #     llm_annotated_text='foo',
    #     matched_surface_forms=True,
    #     preprocessor=hp.llm.convert.galaxy_json_extract,
    #     output_format='json'
    # )
    # stats_dicts.append(stats_dict)

    # # filter
    # filtered_para, num_full_triples_para = \
    #     hp.data.filter_annots.require_parent_single(para_pred)
    # paras_pred.append(filtered_para)

import sys
sys.exit()

aggregate_stats = hp.llm.convert.aggregate_format_stats(stats_dicts)

mode_fn_save = re.sub(
    r'[^\w]',
    '_',
    params_512['model']
    )

with open(f'format_eval_{mode_fn_save}_json.json', 'w') as f:
    print(f'Saving format eval to {f.name}')
    json.dump(aggregate_stats, f, indent=2)

# evaluate
eval_res = hp.evaluation.full(
    paras_true[from_idx:to_idx],
    paras_pred,
    # verbose=True
)

with open(f'eval_{mode_fn_save}_json.json', 'w') as f:
    print(f'Saving eval to {f.name}')
    json.dump(eval_res, f, indent=2)
