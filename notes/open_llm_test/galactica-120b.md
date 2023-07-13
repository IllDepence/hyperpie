# facebook/galactica-120b

## Note

run w/ half precision


## Prompts

### Original

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
In the context of machine learning and related fields,  what (if any) are the entities (datasets, models, methods, loss functions, regularization techniques) mentioned in the LaTeX Input Text below? What (if any) are their parameters and values?

[LaTeX Input Text start]
Our system extends the implementation and hyper-parameters from Lee2017EndtoendNC with the following adjustments. We use a 1 layer BiLSTM with 200-dimensional hidden layers. All the FFNNs have 2 hidden layers of 150 dimensions each. We use 0.4 variational dropout [15] for the LSTMs, 0.4 dropout for the FFNNs, and 0.5 dropout for the input embeddings. We model spans up to 8 words. For beam pruning, we use \(\lambda _{\text{C}}=0.3\) for coreference resolution and \(\lambda _{\text{R}}=0.4\) for relation extraction. For constructing the knowledge graph
[LaTeX Input Text end]

Answer in the following YAML format.

Format:
---
text_contains_entities: true/false
entities:
  - entity<N>:
      id: e<N>
      name: "<entity name>"
      type: dataset/model/method/loss function/regularization technique
      has_parameters: true/false
      parameters:
        - parameter<M>:
            id: p<N.M>
            name: "<parameter name>"
            has_values: true/false
            values:
              - value<O>:
                  value_id: v<N.M.O>
                  value: "<parameter value>"
                  context: "<value context>"/null
                  context_id: c<N.M.O>/null
...

Only include entities that are of type dataset, model, method, loss function, or regularization technique. Do not output entities that are of another type. Do not include entities of type task, metric, library, software, or API.
Only produce output in the YAML format specified above. Output no additional text.

### Response:
```

(for default)

```
**Explanation**

The LaTeX input text is processed using the `allennlp.service.predictors.predictor_for_json` service.  The service processes the input text as follows:

1.  It converts the input text into a JSON object using the `json` predictor, which is configured to recognize `text_contains_entities` and `entities` fields.
2.  It passes the JSON object to the predictor `allennlp.service.predictors.predictor_for_json`, which is configured to recognize `text_contains_entities` and `entities` fields.
3.  It outputs the `text_contains_entities` and `entities` fields of the JSON object produced by the `allennlp.service.predictors.predictor_for_json` predictor.

The `json` predictor is configured to recognize the `text_contains_entities` and `entities` fields, and to include an `id`, `name`, and `type` field in the `entities` field for each entity it detects. The `allenn
```

(with deterministic preset)

```
[YAML Output start]
---
text_contains_entities: true
entities:
  - entity0:
      id: e0
      name: "Lee2017EndtoendNC"
      type: model
      has_parameters: true
      parameters:
        - parameter0:
            id: p0
            name: "hidden_size"
            has_values: true
            values:
              - value0:
                  value_id: v0
                  value: "200"
                  context: "hidden size of the BiLSTM"
                  context_id: c0
  - entity1:
      id: e1
      name: "Lee2017EndtoendNC"
      type: model
      has_parameters: true
      parameters:
        - parameter1:
            id: p1
            name: "ffnn_hidden_size"
            has_values: true
            values:
              - value1:
                  value_id: v1
                  value: "150"
                  context: "hidden size of the FFNN"
                  context_id: c1
  - entity2:
      id: e2
      name: "Lee2017EndtoendNC"
      type: model
      has_parameters: true
      parameters:
        - parameter2:
            id: p2
            name: "ffnn_depth"
            has_values: true
            values:
              - value2:
                  value_id: v2
                  value: "2"
                  context: "depth of the FFNN"
                  context_id: c2
  - entity3:
      id: e3
      name: "Lee2017EndtoendNC"
      type: model
      has_parameters: true
      parameters:
        - parameter3:
            id: p3
            name: "ffnn_dropout"
            has_values: true
            values:
              - value3:
                  value_id: v3
                  value: "0.4"
                  context: "dropout rate for the FFNN"
                  context_id: c3
  - entity4:
      id: e4
      name: "Lee2017EndtoendNC"
      type: model
      has_parameters: true
      parameters:
        - parameter4:
            id: p4
            name: "lstm_dropout"
            has_values: true
            values:
              - value4:
                  value_id: v4
                  value: "0.4"
                  context: "dropout rate for the BiLSTM"
                  context_id: c4
  - entity5:
      id: e5
      name: "Lee2017EndtoendNC"
      type: model
      has_parameters: true
      parameters:
        - parameter5:
            id: p5
            name: "span_width"
            has_values: true
            values:
              - value5:
                  value_id: v5
                  value: "8"
                  context: "maximum span width"
                  context_id: c5
[YAML Output end]






... (endless repetition of newlines)
```

### Alt1

(author specified simple prompt format)

```
Our system extends the implementation and hyper-parameters from Lee2017EndtoendNC with the following adjustments. We use a 1 layer BiLSTM with 200-dimensional hidden layers. All the FFNNs have 2 hidden layers of 150 dimensions each. We use 0.4 variational dropout [15] for the LSTMs, 0.4 dropout for the FFNNs, and 0.5 dropout for the input embeddings. We model spans up to 8 words. For beam pruning, we use \(\lambda _{\text{C}}=0.3\) for coreference resolution and \(\lambda _{\text{R}}=0.4\) for relation extraction. For constructing the knowledge graph

What scientific entities are mentioned in the text above?

```

---

```
protein, gene, protein complex, protein family, protein structure, protein function, protein interaction, protein structure, protein domain, protein function, protein structure, protein domain, protein function, protein structure, protein domain, protein function, protein structure, protein domain, protein function, protein structure, protein domain, protein function, protein structure [...]
```

### Alt2

(simple style original prompt)

```
In the context of machine learning and related fields, what (if any) are the entities (datasets, models, methods, loss functions, regularization techniques) mentioned in the LaTeX Input Text below? What (if any) are their parameters and values?

[LaTeX Input Text start]
Our system extends the implementation and hyper-parameters from Lee2017EndtoendNC with the following adjustments. We use a 1 layer BiLSTM with 200-dimensional hidden layers. All the FFNNs have 2 hidden layers of 150 dimensions each. We use 0.4 variational dropout [15] for the LSTMs, 0.4 dropout for the FFNNs, and 0.5 dropout for the input embeddings. We model spans up to 8 words. For beam pruning, we use \(\lambda _{\text{C}}=0.3\) for coreference resolution and \(\lambda _{\text{R}}=0.4\) for relation extraction. For constructing the knowledge graph
[LaTeX Input Text end]

Answer in the following YAML format.

Format:
---
- text_contains_entities: true/false
- entities (datasets, models, methods, loss functions, regularization techniques):
    - entity<N>:
        name: <entity name>
        type: <entity type>
        has_parameters: true/false
        parameters:
            - parameter<N>:
                name: <parameter name>
                value: <parameter value>/null
                context: <value context>/null
...

Only produce output in the YAML format specified above. Output no additional text.

Output:
```

---

```
---
text_contains_entities: true
entities (datasets, models, methods, loss functions, regularization techniques):
    - entity0:
        name: LSTM
        type: model
        has_parameters: true
        parameters:
            - parameter0:
                name: hidden_size
                value: 200
                context: hidden_size of the LSTM
    - entity1:
        name: FFNN
        type: model
        has_parameters: true
        parameters:
            - parameter0:
                name: hidden_size
                value: 150
                context: hidden_size of the FFNN
    - entity2:
        name: dropout
        type: regularization
        has_parameters: true
        parameters:
            - parameter0:
                name: value
                value: 0.4
                context: value of the dropout
    - entity3:
        name: variational_dropout
        type: regularization
        has_parameters: true
        parameters:
            - parameter0:
                name: value
                value: 0.4
                context: value of the variational dropout
    - entity4:
        name: span_length
        type: method
        has_parameters: false
    - entity5:
        name: beam_pruning
        type: method
        has_parameters: true
        parameters:
            - parameter0:
                name: lambda_C
                value: 0.3
                context: lambda_C for the coreference resolution
            - parameter1:
                name: lambda_R
                value: 0.4
                context: lambda_R for the relation extraction




(repeats newlines)
```

### Alt3

```
In the context of machine learning and related fields,  what (if any) are the scientific entities (datasets, models, methods, loss functions, regularization techniques) mentioned in the LaTeX Input Text below? What (if any) are their parameters and values?

LaTeX Input Text:
Our system extends the implementation and hyper-parameters from Lee2017EndtoendNC with the following adjustments. We use a 1 layer BiLSTM with 200-dimensional hidden layers. All the FFNNs have 2 hidden layers of 150 dimensions each. We use 0.4 variational dropout [15] for the LSTMs, 0.4 dropout for the FFNNs, and 0.5 dropout for the input embeddings. We model spans up to 8 words. For beam pruning, we use \(\lambda _{\text{C}}=0.3\) for coreference resolution and \(\lambda _{\text{R}}=0.4\) for relation extraction. For constructing the knowledge graph

Answer in the following YAML format.

Format:
---
text_contains_entities: true/false
entities:
  - entity<N>:
      id: e<N>
      name: "<entity name>"
      type: dataset/model/method/loss function/regularization technique
      has_parameters: true/false
      parameters:
        - parameter<M>:
            id: p<N.M>
            name: "<parameter name>"
            has_values: true/false
            values:
              - value<O>:
                  value_id: v<N.M.O>
                  value: "<parameter value>"
                  context: "<value context>"/null
                  context_id: c<N.M.O>/null
...

Only include entities that are of type dataset, model, method, loss function, or regularization technique. Do not output entities that are of another type. Do not include entities of type task, metric, library, software, or API.
Only produce output in the YAML format specified above. Output no additional text.
```

---

```



...
```

(only outputs newlines)


### Alt4

```
### Instruction:
In the context of machine learning and related fields,  what (if any) are the scientific entities (datasets, models, methods, loss functions, regularization techniques) mentioned in the LaTeX Input Text below? What (if any) are their parameters and values?

LaTeX Input Text:
Our system extends the implementation and hyper-parameters from Lee2017EndtoendNC with the following adjustments. We use a 1 layer BiLSTM with 200-dimensional hidden layers. All the FFNNs have 2 hidden layers of 150 dimensions each. We use 0.4 variational dropout [15] for the LSTMs, 0.4 dropout for the FFNNs, and 0.5 dropout for the input embeddings. We model spans up to 8 words. For beam pruning, we use \(\lambda _{\text{C}}=0.3\) for coreference resolution and \(\lambda _{\text{R}}=0.4\) for relation extraction. For constructing the knowledge graph

Answer in the following YAML format.

Format:
---
text_contains_entities: true/false
entities:
  - entity<N>:
      id: e<N>
      name: "<entity name>"
      type: dataset/model/method/loss function/regularization technique
      has_parameters: true/false
      parameters:
        - parameter<M>:
            id: p<N.M>
            name: "<parameter name>"
            has_values: true/false
            values:
              - value<O>:
                  value_id: v<N.M.O>
                  value: "<parameter value>"
                  context: "<value context>"/null
                  context_id: c<N.M.O>/null
...

Only include entities that are of type dataset, model, method, loss function, or regularization technique. Do not output entities that are of another type. Do not include entities of type task, metric, library, software, or API.
Only produce output in the YAML format specified above. Output no additional text.

### Response:
```

---

```
(three backticks)yaml
---
text_contains_entities: true
entities:
  - entity0:
      id: e0
      name: "CoNLL-2012"
      type: dataset
      has_parameters: true
      parameters:
        - parameter0:
            id: p0
            name: "train_file"
            has_values: true
            values:
              - value0:
                  value_id: v0
                  value: "data/conll-2012/train.txt"
                  context: "CoNLL-2012"
                  context_id: c0
  - entity1:
      id: e1
      name: "Lee2017EndtoendNC"
      type: model
      has_parameters: true
      parameters:
        - parameter0:
            id: p1
            name: "word_dim"
            has_values: true
            values:
              - value0:
                  value_id: v1
                  value: "300"
                  context: "Lee2017EndtoendNC"
                  context_id: c1
        - parameter1:
            id: p2
            name: "tag_dim"
            has_values: true
            values:
              - value0:
                  value_id: v2
                  value: "30"
                  context: "Lee2017EndtoendNC"
                  context_id: c2
        - parameter2:
            id: p3
            name: "lstm_dim"
            has_values: true
            values:
              - value0:
                  value_id: v3
                  value: "200"
                  context: "Lee2017EndtoendNC"
                  context_id: c3
        - parameter3:
            id: p4
            name: "mlp_dim"
            has_values: true
            values:
              - value0:
                  value_id: v4
                  value: "150"
                  context: "Lee2017EndtoendNC"
                  context_id: c4
        - parameter4:
            id: p5
            name: "train_epochs"
            has_values: true
            values:
              - value0:
                  value_id: v5
                  value: "100"
                  context: "Lee2017EndtoendNC"
                  context_id: c5
        - parameter5:
            id: p6
            name: "train_batch_size"
            has_values: true
            values:
              - value0:
                  value_id: v6
                  value: "10"
                  context: "Lee2017EndtoendNC"
                  context_id: c6
        - parameter6:
            id: p7
            name: "dropout"
            has_values: true
            values:
              - value0:
                  value_id: v7
                  value: "0.4"
                  context: "Lee2017EndtoendNC"
                  context_id: c7
        - parameter7:
            id: p8
            name: "word_dropout"
            has_values: true
            values:
              - value0:
                  value_id: v8
                  value: "0.05"
                  context: "Lee2017EndtoendNC"
                  context_id: c8
        - parameter8:
            id: p9
            name: "tag_dropout"
            has_values: true
            values:
              - value0:
                  value_id: v9
                  value: "0.4"
                  context: "Lee2017EndtoendNC"
                  context_id: c9
        - parameter9:
            id: p10
            name: "lstm_dropout"
            has_values: true
            values:
              - value0:
                  value_id: v10
                  value: "0.4"
                  context: "Lee2017...
```