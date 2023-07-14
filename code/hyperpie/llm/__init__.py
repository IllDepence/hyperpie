import os
import openai
from . import (
    convert, predict, prompt_templates, util
)
from hyperpie import settings


if settings.use_openai_api:
    print(f'WARNING: using OpenAI API for LLM completion')
    print(f'WARNING: this will cost money')
    if os.path.exists(settings.openai_org_fp):
        with open(settings.openai_org_fp) as f:
            openai.organization = f.read().strip()
    else:
        openai.organization = None
    if os.path.exists(settings.openai_key_fp):
        with open(settings.openai_key_fp) as f:
            openai.api_key = f.read().strip()
    else:
        openai.api_key = None
else:
    print(f'INFO: using Basaran for LLM completion')
    print(f'INFO: change settings.use_openai_api to True to use OpenAI API\n')
    openai.api_key = 'xxx'
    openai.api_base = settings.basaran_api_base
    completion = openai.Completion.create(
        model='xxx', prompt='Han shot ', max_tokens=2, echo=True
    )
    model_in_use = completion['model']
    settings.gpt_default_params['model'] = model_in_use
    print(f'INFO: using model {model_in_use} for LLM completion')
    print(f'INFO: the model says "{completion["choices"][0]["text"]}"')
