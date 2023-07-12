import hyperpie as hp

assert hp.settings.use_openai_api is False

para = 'Our system extends the implementation and hyper-parameters from Lee2017EndtoendNC with the following adjustments. We use a 1 layer BiLSTM with 200-dimensional hidden layers. All the FFNNs have 2 hidden layers of 150 dimensions each. We use 0.4 variational dropout [15] for the LSTMs, 0.4 dropout for the FFNNs, and 0.5 dropout for the input embeddings. We model spans up to 8 words. For beam pruning, we use \\(\\lambda _{\text{C}}=0.3\\) for coreference resolution and \\(\\lambda _{\text{R}}=0.4\\) for relation extraction. For constructing the knowledge graph'  # noqa

completion_dict, form_cache = hp.llm.predict.openai_api(
    para,
    hp.llm.prompt_templates.text_e2e_fillin_twostep_1_alpaca_style_nointro.format(  # noqa
        text=para
    ),
    verbose=True
)

print(completion_dict['completion']['choices'][0]['text'])
