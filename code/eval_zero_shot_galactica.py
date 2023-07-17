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


# # evaluate
# relext_f1 = hp.evaluation.full(
#     paras_true[from_idx:to_idx],
#     paras_pred,
#     # verbose=True
# )
