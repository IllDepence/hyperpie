""" Evaluate zero-shot setting for GALACTICA.
"""

import json
import random
import hyperpie as hp

assert hp.settings.use_openai_api is False

d = hp.data.load.load_annotated()
paras_true = d

# shuffle
random.Random(42).shuffle(paras_true)

# paras_true, filter_stats = hp.data.filter_annots.require_parent(
#     d
# )
paras_pred = []

from_idx = 0
to_idx = len(paras_true)
params_512 = hp.settings.gpt_default_params.copy()
params_512['max_tokens'] = 512
for i, para in enumerate(paras_true[from_idx:to_idx]):
    print(f'{i}/{len(paras_true)}')

    # - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - single step eval - - - - - -
    # - - - - - - - - - - - - - - - - - - - - -
    prompt = hp.llm.prompt_templates.text_e2e_fillin_twostep_1_alpaca_style_nointro.format(  # noqa: E501
        text=para['text']
    )
    completion_dict, from_cache = hp.llm.predict.openai_api(
        para, prompt, params=params_512
    )

    print('TEXT')
    print(para['text'][0:300], '...')
    print('COMPLETION')
    print(completion_dict['completion']['choices'][0]['text'][:300], '...')
    print('\n\n')
    # # convert
    # para_pred = hp.llm.convert.llm_output2eval_input(
    #     completion_dict,
    # )

    # # filter
    # filtered_para, num_full_triples_para = \
    #     hp.data.filter_annots.require_parent_single(para_pred)
    # paras_pred.append(filtered_para)

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
