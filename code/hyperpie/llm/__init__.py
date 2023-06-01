import openai
from . import (
    convert, predict, prompt_templates, util
)
from hyperpie import settings


with open(settings.openai_org_fp) as f:
    openai.organization = f.read().strip()
with open(settings.openai_key_fp) as f:
    openai.api_key = f.read().strip()
