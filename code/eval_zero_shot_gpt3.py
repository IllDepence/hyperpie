""" Evaluate zero-shot setting for GPT-3.
"""

import hyperpie as hp

# Use data filtered for “full info sets” (a<p<v[<c])
# because GPT-3 only predicts those
paras_true = hp.data.load.load_annotated(filtered=True)
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
    para_pred = hp.llm.convert.yaml2json(completion_dict)
    paras_pred.append(para_pred)

# evaluate
hp.evaluation.full(paras_true, paras_pred)
