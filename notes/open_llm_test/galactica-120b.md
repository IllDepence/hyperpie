# facebook/galactica-120b

## Note

when loading w/ half precision

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 44.00 MiB (GPU 1; 79.18 GiB total capacity; 77.90 GiB already allocated; 25.50 MiB free; 77.97 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

currently only usable in 8bit: `MODEL=facebook/galactica-120b MODEL_LOAD_IN_8BIT=true MODEL_TRUST_REMOTE_CODE=true PORT=10127 python -m basaran`


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
[LaTeX Output Text start]
We use a 1 layer BiLSTM with 200-dimensional hidden layers. All the FFNNs have 2 hidden layers of 150 dimensions each. We use 0.4 variational dropout [15] for the LSTMs, 0.4 dropout for the FFNNs, and 0.5 dropout for the input embeddings. We model spans up to 8 words. For beam pruning, we use \(\lambda _{\text{C}}=0.3\) for coreference resolution and \(\lambda _{\text{R}}=0.4\) for relation extraction. For constructing the knowledge graph
[LaTeX Output Text end]

### Context:

- You will be given a short text with LaTeX Input Text.
- The goal is to output a short text with LaTeX Output Text.
- You will be given a short text with LaTeX Input Text.
- The goal is to output a short text with LaTeX Output Text.

## Example:

- You will be given a short text with LaTeX
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
        - parameter0:
            id: p1
            name: "ffnn_hidden_size"
            has_values: true
            values:
              - value0:
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
        - parameter0:
            id: p2
            name: "ffnn_depth"
            has_values: true
            values:
              - value0:
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
        - parameter0:
            id: p3
            name: "ffnn_dropout"
            has_values: true
            values:
              - value0:
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
        - parameter0:
            id: p4
            name: "lstm_dropout"
            has_values: true
            values:
              - value0:
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
        - parameter0:
            id: p5
            name: "span_width"
            has_values: true
            values:
              - value0:
                  value_id: v5
                  value: "8"
                  context: "maximum span width"
                  context_id: c5
[YAML Output end]


### Sample Input 1

### Sample Output 1

### Sample Input 2

### Sample Output 2

### Sample Input 3

### Sample Output 3

### Sample Input 4

### Sample Output 4

### Sample Input 5

### Sample Output 5

### Sample Input 6

### Sample Output 6

### Sample Input 7

### Sample Output 
```

### Alt1

(author specified simple prompt format)

```
Our system extends the implementation and hyper-parameters from Lee2017EndtoendNC with the following adjustments. We use a 1 layer BiLSTM with 200-dimensional hidden layers. All the FFNNs have 2 hidden layers of 150 dimensions each. We use 0.4 variational dropout [15] for the LSTMs, 0.4 dropout for the FFNNs, and 0.5 dropout for the input embeddings. We model spans up to 8 words. For beam pruning, we use \(\lambda _{\text{C}}=0.3\) for coreference resolution and \(\lambda _{\text{R}}=0.4\) for relation extraction. For constructing the knowledge graph

What scientific entities are mentioned in the text above?

```

---

```
No chemical entities were mentioned

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

## References

[1] https://spacy.io/
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
  - entity1:
      id: e1
      name: "CoNLL-2012"
      type: dataset
      has_parameters: true
      parameters:
        - parameter1:
            id: p1
            name: "train_file"
            has_values: true
            values:
              - value1:
                  value_id: v1
                  value: "data/conll-2012/train.txt"
                  context: "The train file."
                  context_id: c1
  - entity2:
      id: e2
      name: "Lee2017EndtoendNC"
      type: model
      has_parameters: true
      parameters:
        - parameter1:
            id: p2
            name: "hidden_dim"
            has_values: true
            values:
              - value1:
                  value_id: v2
                  value: "200"
                  context: "The hidden dimension."
                  context_id: c2
        - parameter2:
            id: p3
            name: "ffnn_hidden_dim"
            has_values: true
            values:
              - value1:
                  value_id: v3
                  value: "150"
                  context: "The hidden dimension of the FFNN."
                  context_id: c3
        - parameter3:
            id: p4
            name: "ffnn_depth"
            has_values: true
            values:
              - value1:
                  value_id: v4
                  value: "2"
                  context: "The depth of the FFNN."
                  context_id: c4
        - parameter4:
            id: p5
            name: "word_embedding_dim"
            has_values: true
            values:
              - value1:
                  value_id: v5
                  value: "100"
                  context: "The dimension of the word embeddings."
                  context_id: c5
        - parameter5:
            id: p6
            name: "train_file"
            has_values: true
            values:
              - value1:
                  value_id: v6
                  value: "data/conll-2012/train.txt"
                  context: "The train file."
                  context_id: c6
        - parameter6:
            id: p7
            name: "eval_file"
            has_values: true
            values:
              - value1:
                  value_id: v7
                  value: "data/conll-2012/test.txt"
                  context: "The test file."
                  context_id:
```