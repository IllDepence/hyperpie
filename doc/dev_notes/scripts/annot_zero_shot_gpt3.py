""" Annotate pre-filtered paragraphs in zero-shot setting with GPT-3.
"""

import json
import hyperpie as hp

assert hp.settings.use_openai_api is True  # make sure we’re using GPT3

paras_unannot = hp.data.load.load_filtered_unannotated()
paras_pred = []

for i, para in enumerate(paras_unannot):
    print(f'{i}/{len(paras_unannot)}')
    if i == 1201:
        break

    prompt = hp.llm.prompt_templates.text_e2e_fillin_twostep_1.format(
        text=para['text']
    )

    completion_dict, from_cache = hp.llm.predict.openai_api(
        para, prompt, params=hp.settings.gpt_default_params
    )
    # print(completion_dict['completion']['choices'][0]['text'])
    # import sys
    # sys.exit()

    # convert
    para_pred = hp.llm.convert.llm_output2eval_input(
        completion_dict,
        '',
        matched_surface_forms=True
    )
    filtered_para, num_full_triples_para = \
        hp.data.filter_annots.require_parent_single(para_pred)
    paras_pred.append(filtered_para)


with open('/tmp/llm_pred.json', 'w') as f:
    json.dump(paras_pred, f, indent=4)
