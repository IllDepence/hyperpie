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
    # prompt (stage 1)
    print(f'{i+0.5}/{len(paras_true)}')

    # # print number of entities and relations
    # print(f'  {len(para["annotation"]["entities"])} entities')
    # print(f'  {len(para["annotation"]["relations"])} relations')
    # x = input()
    # if x == 'q':
    #     import sys
    #     sys.exit()
    # elif x == 'p':
    #     print(para['text'])
    # continue

    # # - - - - - - - - - - - - - - - - - - - - -
    # # - - - - - - single step eval - - - - - -
    # # - - - - - - - - - - - - - - - - - - - - -
    # prompt = hp.llm.prompt_templates.text_e2e.format(
    #     text=para['text']
    # )
    # completion_dict, from_cache = hp.llm.predict.openai_api(
    #     para, prompt, params=hp.settings.gpt_default_params
    # )
    # # convert
    # para_pred, status = hp.llm.convert.llm_output2eval_input(
    #     completion_dict,
    # )

    # # - - - - - - - - - - - - - - - - - - -
    # # - - - - - - two step eval - - - - - -
    # # - - - - - - - - - - - - - - - - - - -
    prompt1 = hp.llm.prompt_templates.text_e2e_fillin_twostep_1.format(
        text=para['text']
    )
    completion_dict1, from_cache = hp.llm.predict.openai_api(
        para, prompt1, params=hp.settings.gpt_default_params
    )
    # # prompt (stage 2)
    # print(f'{i+1}/{len(paras_true)}')
    # prompt2 = hp.llm.prompt_templates.text_e2e_fillin_twostep_2.format(
    #     yaml=completion_dict1['completion']['choices'][0]['text'],
    #     text=para['text']
    # )
    # completion_dict2, from_cache = hp.llm.predict.openai_api(
    #     para, prompt2, params=hp.settings.gpt_default_params
    # )
    # # convert
    # para_pred, stats_dict= hp.llm.convert.llm_output2eval_input(
    #     completion_dict1,
    #     completion_dict2['completion']['choices'][0]['text']
    # )
    # “1.5 stage” version
    para_pred, stats_dict = hp.llm.convert.llm_output2eval_input(
        completion_dict1,
        llm_annotated_text='',
        matched_surface_forms=True,
        skip_nonmatching=True
    )
    # # “broken” “1.5 stage” version leading to higher F1 score
    # # b/c of fewer false positives
    # para_pred, stats_dict = hp.llm.convert.llm_output2eval_input(
    #     completion_dict1,
    # )

    stats_dicts.append(stats_dict)

    # filter
    filtered_para, num_full_triples_para = \
        hp.data.filter_annots.require_parent_single(para_pred)
    paras_pred.append(filtered_para)

aggregate_stats = hp.llm.convert.aggregate_format_stats(stats_dicts)

mode_fn_save = re.sub(
    r'[^\w]',
    '_',
    hp.settings.gpt_default_params['model']
    )

with open(f'format_eval_{mode_fn_save}.json', 'w') as f:
    print(f'Saving format eval to {f.name}')
    json.dump(aggregate_stats, f, indent=2)

# eval_name = 'zero_shot_gpt3_twostep'
# # save paras_pred in JSON file in /tmp/
# fp = f'/tmp/{eval_name}_pred.json'
# with open(fp, 'w') as f:
#     json.dump(paras_pred, f, indent=2)
# # save paras_true in JSON file in /tmp/
# fp = f'/tmp/{eval_name}_true.json'
# with open(fp, 'w') as f:
#     json.dump(paras_true[from_idx:to_idx], f, indent=2)

# evaluate
eval_res = hp.evaluation.full(
    paras_true[from_idx:to_idx],
    paras_pred,
    # verbose=True
)

with open(f'eval_{mode_fn_save}.json', 'w') as f:
    print(f'Saving eval to {f.name}')
    json.dump(eval_res, f, indent=2)
