""" Evaluate few-shot setting for Vicuna JSON.
"""

import re
import json
import random
import hyperpie as hp

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
params_ext = hp.settings.gpt_default_params.copy()
params_ext['model'] = 'lmsys/vicuna-13b-v1.5-16k'
stats_dicts = []
for i, para in enumerate(paras_true[from_idx:to_idx]):
    print(f'{i}/{len(paras_true)}')

    completion_str = hp.data.llm_cache.load_external(
        'vicuna_5shot_json',
        para['document_id'],
        para['paragraph_index']
    )

    # reconstruct completion_dict
    completion_dict = {
        'paragraph': para,
        'completion': {
            'choices': [
                {
                    'text': completion_str
                }
            ]
        }
    }

    para_pred, stats_dict = hp.llm.convert.llm_output2eval_input(
        completion_dict,
        llm_annotated_text='foo',
        matched_surface_forms=True,
        output_format='json'
    )
    stats_dicts.append(stats_dict)

    # filter
    filtered_para, num_full_triples_para = \
        hp.data.filter_annots.require_parent_single(para_pred)
    paras_pred.append(filtered_para)

    # print('TEXT')
    # print(para['text'][0:300], '...')
    # print('COMPLETION')
    # print('>>{}<<'.format(completion_dict['completion']['choices'][0]['text']))
    # print('\n\n')
    # print('>>{}<<'.format(json.dumps(para_pred, indent=2)))
    # print('\n\n')
    # input()

aggregate_stats = hp.llm.convert.aggregate_format_stats(stats_dicts)

mode_fn_save = re.sub(
    r'[^\w]',
    '_',
    params_ext['model']
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
