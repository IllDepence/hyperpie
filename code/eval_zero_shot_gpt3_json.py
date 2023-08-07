""" Evaluate zero-shot setting for GPT-3.
"""

import json
import re
import hyperpie as hp

assert hp.settings.use_openai_api is True  # make sure we’re using GPT3

# Use data filtered for “full info sets” (a<p<v[<c])
# because GPT-3 only predicts those
d = hp.data.load.load_annotated()
# paras_true = d
paras_true, filter_stats = hp.data.filter_annots.require_parent(
    d
)
paras_pred = []

# get predictions
# from_idx = 269, 279
from_idx = 0
to_idx = len(paras_true)
stats_dicts = []
for i, para in enumerate(paras_true[from_idx:to_idx]):
    print(f'{i}/{len(paras_true)}')

    prompt1 = hp.llm.prompt_templates.text_e2e_fillin_twostep_1_json.format(
        text=para['text']
    )
    completion_dict1, from_cache = hp.llm.predict.openai_api(
        para, prompt1, params=hp.settings.gpt_default_params
    )

    # print('TEXT')
    # print(para['text'][0:300], '...')
    # print('COMPLETION')
    # print(completion_dict1['completion']['choices'][0]['text'])  # [:300], '...') # noqa
    # print('\n\n')

    para_pred, stats_dict = hp.llm.convert.llm_output2eval_input(
        completion_dict1,
        llm_annotated_text='',
        matched_surface_forms=True,
        preprocessor=hp.llm.convert.gpt3_json_extract,
        output_format='json'
    )

    stats_dicts.append(stats_dict)

    # # filter
    # filtered_para, num_full_triples_para = \
    #     hp.data.filter_annots.require_parent_single(para_pred)
    # paras_pred.append(filtered_para)

aggregate_stats = hp.llm.convert.aggregate_format_stats(stats_dicts)

mode_fn_save = re.sub(
    r'[^\w]',
    '_',
    hp.settings.gpt_default_params['model']
    )

with open(f'format_eval_{mode_fn_save}_json.json', 'w') as f:
    print(f'Saving format eval to {f.name}')
    json.dump(aggregate_stats, f, indent=2)

# # evaluate
# relext_f1 = hp.evaluation.full(
#     paras_true[from_idx:to_idx],
#     paras_pred,
#     # verbose=True
# )
