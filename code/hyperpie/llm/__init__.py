import openai
from . import (
    convert, predict, prompt_templates, util
)


with open('hyperpie/llm/api_organization') as f:
    openai.organization = f.read().strip()
with open('hyperpie/llm/api_key') as f:
    openai.api_key = f.read().strip()


default_params = {
    "model": "text-davinci-003",  # "default" for other models
    "max_tokens": 512,
    "temperature": 0.0,           # 0 - 2
    "top_p": 1,                   # default 1, change only w/ detault temp
    "n": 1,                       # num completions to generate
    "logprobs": 0,                # return log probs of n tokens (max 5)
    "echo": False,
}
