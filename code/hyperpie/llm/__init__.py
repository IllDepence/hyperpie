import os
import openai
from . import (
    convert, predict, prompt_templates, util
)
from hyperpie import settings


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
