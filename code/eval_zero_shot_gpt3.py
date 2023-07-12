""" Evaluate zero-shot setting for GPT-3.
"""

import json
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

    # - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - single step eval - - - - - -
    # - - - - - - - - - - - - - - - - - - - - -
    prompt = hp.llm.prompt_templates.text_e2e.format(
        text=para['text']
    )
    completion_dict, from_cache = hp.llm.predict.openai_api(
        para, prompt, params=hp.settings.gpt_default_params
    )
    # convert
    para_pred = hp.llm.convert.llm_output2eval_input(
        completion_dict,
    )

    # # - - - - - - - - - - - - - - - - - - -
    # # - - - - - - two step eval - - - - - -
    # # - - - - - - - - - - - - - - - - - - -
    # prompt1 = hp.llm.prompt_templates.text_e2e_fillin_twostep_1.format(
    #     text=para['text']
    # )
    # completion_dict1, from_cache = hp.llm.predict.openai_api(
    #     para, prompt1, params=hp.settings.gpt_default_params
    # )
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
    # para_pred = hp.llm.convert.llm_output2eval_input(
    #     completion_dict1,
    #     completion_dict2['completion']['choices'][0]['text']
    # )
    # # # “1.5 stage” version
    # # para_pred = hp.llm.convert.llm_output2eval_input(
    # #     completion_dict1,
    # #     # completion_dict2['completion']['choices'][0]['text']
    # #     '',
    # #     matched_surface_forms=True
    # # )
    # # # “broken” “1.5 stage” version leading to higher F1 score
    # # # b/c of fewer false positives
    # # para_pred = hp.llm.convert.llm_output2eval_input(
    # #     completion_dict1,
    # # )

    # filter
    filtered_para, num_full_triples_para = \
        hp.data.filter_annots.require_parent_single(para_pred)
    paras_pred.append(filtered_para)

eval_name = 'zero_shot_gpt3_twostep'
# save paras_pred in JSON file in /tmp/
fp = f'/tmp/{eval_name}_pred.json'
with open(fp, 'w') as f:
    json.dump(paras_pred, f, indent=2)
# save paras_true in JSON file in /tmp/
fp = f'/tmp/{eval_name}_true.json'
with open(fp, 'w') as f:
    json.dump(paras_true[from_idx:to_idx], f, indent=2)

# evaluate
relext_f1 = hp.evaluation.full(
    paras_true[from_idx:to_idx],
    paras_pred,
    # verbose=True
)

# for eval_idx in range(80, 81):

#     print(f'eval_idx: {eval_idx}')

#     # get predictions
#     for i, para in enumerate(paras_true[eval_idx:eval_idx+1]):
#         print(f'{i+1}/{len(paras_true)}')
#         prompt = hp.llm.prompt_templates.text_e2e.format(
#             text=para['text']
#         )
#         completion_dict, from_cache = hp.llm.predict.openai_api(
#             para, prompt, params=hp.settings.gpt_default_params
#         )
#         para_pred = hp.llm.convert.yaml2json(completion_dict)
#         filtered_para, num_full_triples_para = \
#             hp.data.filter_annots.require_parent_single(para_pred)
#         paras_pred.append(filtered_para)

#     # evaluate
#     relext_f1 = hp.evaluation.full(
#         paras_true[eval_idx:eval_idx+1],
#         [filtered_para]
#     )

#     if relext_f1 < 1:
#         print(f'para text: {paras_true[eval_idx:eval_idx+1][0]["text"]}')
#         print()
#         print(f'{completion_dict["completion"]["choices"][0]["text"]}')
#         print()
#         with open('/tmp/paras_true.json', 'w') as f:
#             json.dump(paras_true[eval_idx:eval_idx+1], f, indent=4)
#         with open('/tmp/paras_pred.json', 'w') as f:
#             json.dump([filtered_para], f, indent=4)
#         input()
