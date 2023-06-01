""" Overall package settings.
"""

# Data paths
data_base_path = '/home/tarek/proj/hyperparam_paper/xiao_collab_repo/data/'
annot_raw_fp = data_base_path + 'tsa.json'
annot_prep_fp = data_base_path + 'tsa_processed.json'
annot_filtered_fp = data_base_path + 'tsa_processed_onlyfull.json'
llm_cache_fp = 'hyperpie/llm/completion_cache.json'

# LLM
# # GPT
openai_org_fp = 'hyperpie/llm/api_organization'
openai_key_fp = 'hyperpie/llm/api_key'
gpt_default_params = {
    "model": "text-davinci-003",  # "default" for other models
    "max_tokens": 512,
    "temperature": 0.0,           # 0 - 2
    "top_p": 1,                   # default 1, change only w/ detault temp
    "n": 1,                       # num completions to generate
    "logprobs": 0,                # return log probs of n tokens (max 5)
    "echo": False,
}