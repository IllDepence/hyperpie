import hyperpie


demo_text_positive_1 = """We divide our 438 annotated documents into training (70%), validation (30%) and test set (30%). The base document representation of our model is formed by SciBERT-base [4] and BiLSTM with 128-d hidden state. We use a dropout of 0.2 after BiLSTM embeddings. All feedforward networks are composed of two hidden layers, each of dimension 128 with gelu activation and with a dropout of 0.2 between layers. For additive attention layer in span representation, we collapse the token embeddings to scalars by passing through the feedforward layer with 128-d hidden state and performing a softmax. We train our model for 30 epochs using Adam optimizer with 1e-3 as learning rate for all non BERT weights and 2e-5 for BERT weights. We use early stopping with a patience value of 7 on the validation set using relation extraction F1 score. All our models were trained using 48Gb Quadro RTX 8000 GPUs. The multitask model takes approximately 3"""  # noqa: E501


def demo():
    completion = hyperpie.llm.predict.prompt_and_save(
        demo_text_positive_1,
        hyperpie.llm.prompt_templates.text_e2e.format(
            text=demo_text_positive_1
        ),
        hyperpie.llm.default_params
    )


if __name__ == '__main__':
    demo()
