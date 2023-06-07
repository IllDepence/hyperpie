""" Evaluate zero-shot setting for GPT-3.
"""

import json
import hyperpie as hp

# Use data filtered for “full info sets” (a<p<v[<c])
# because GPT-3 only predicts those
d = hp.data.load.load_annotated()
paras_true = d
# paras_true, filter_stats = hp.data.filter_annots.require_parent(
#     d
# )
paras_pred = []

# get predictions
for i, para in enumerate(paras_true):
    print(f'{i+1}/{len(paras_true)}')
    prompt = hp.llm.prompt_templates.text_e2e.format(
        text=para['text']
    )
    completion_dict, from_cache = hp.llm.predict.openai_api(
        para, prompt, params=hp.settings.gpt_default_params
    )
    # print(completion_dict['completion']['choices'][0]['text'])
    # x = input()
    # if x == 'i':
    #     import IPython
    #     IPython.embed()
    # elif x == 'q':
    #     exit(1)
    # continue
    para_pred = hp.llm.convert.llm_output2eval_input(completion_dict)
    filtered_para, num_full_triples_para = \
        hp.data.filter_annots.require_parent_single(para_pred)
    paras_pred.append(filtered_para)

# evaluate
relext_f1 = hp.evaluation.full(
    paras_true,
    paras_pred
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
