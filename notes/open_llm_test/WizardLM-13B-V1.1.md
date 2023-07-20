# WizardLM/WizardLM-13B-V1.1

## Notes

* also tested with prompt format listed for 13B model at https://github.com/nlpxucan/WizardLM)
    * “A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: hello, who are you? ASSISTANT: ”
    * resulted in empty completions

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
---
text_contains_entities: true
entities:
- entity<1>:
  id: e1
  name: "Lee2017EndtoendNC"
  type: model
has_parameters: true
parameters:
- parameter<1>:
    id: p1
    name: "hyper-parameters"
    has_values: true
    values:
      - value<1.1>:
          value_id: v1.1
          value: "1 layer BiLSTM with 200-dimensional hidden layers"
          context: "1 layer BiLSTM"
          context_id: c1.1
      - value<1.2>:
          value_id: v1.2
          value: "FFNNs have 2 hidden layers of 150 dimensions each"
          context: "FFNNs"
          context_id: c1.2
- entity<2>:
  id: e2
  name: "1 layer BiLSTM"
  type: model
has_parameters: false
- entity<3>:
  id: e
```

(for deterministic preset w/ max output 2048)

```
text_contains_entities: true
entities:
- entity<1>:
  id: e1
  name: "1 layer BiLSTM"
  type: model
  has_parameters: false
- entity<2>:
  id: e2
  name: "200-dimensional hidden layers"
  type: model
  has_parameters: false
- entity<3>:
  id: e3
  name: "2 hidden layers of 150 dimensions each"
  type: model
  has_parameters: false
- entity<4>:
  id: e4
  name: "0.4 variational dropout"
  type: loss function
  has_parameters: false
- entity<5>:
  id: e5
  name: "0.4 dropout"
  type: loss function
  has_parameters: false
- entity<6>:
  id: e6
  name: "0.5 dropout for input embeddings"
  type: loss function
  has_parameters: false
- entity<7>:
  id: e7
  name: "model spans up to 8 words"
  type: method
  has_parameters: false
- entity<8>:
  id: e8
  name: "for beam pruning"
  type: method
  has_parameters: false
- entity<9>:
  id: e9
  name: "we use"
  type: method
  has_parameters: false
- entity<10>:
  id: e10
  name: "λ\_C=0.3"
  type: method
  has_parameters: false
- entity<11>:
  id: e11
  name: "λ\_R=0.4"
  type: method
  has_parameters: false
- entity<12>:
  id: e12
  name: "for coreference resolution"
  type: method
  has_parameters: false
- entity<13>:
  id: e13
  name: "for relation extraction"
  type: method
  has_parameters: false
- entity<14>:
  id: e14
  name: "constructing the knowledge graph"
  type: method
  has_parameters: false
```

### Var 1

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
{
"text_contains_entities": true,
"entities": {
"datasets": [],
"models": [
{
"name": "Lee2017EndtoendNC",
"type": "implementation",
"has_parameters": true,
"parameters": [
{
"name": "1 layer BiLSTM",
"value": "1",
"context": "A single layer of a bidirectional long short-term memory (BiLSTM) network with 200-dimensional hidden layers."
},
{
"name": "FFNNs",
"value": "FFNNs",
"context": "Feedforward neural networks (FFNNs)"
},
{
"name": "2 hidden layers",
"value": "2",
"context": "Each FFNN has 2 hidden layers, each with 150 dimensions."
},
{
"name": "0.4 variational dropout",
"value": "0.4",
"context": "A dropout rate of 0.4 for the LSTMs."
},
{
"name": "0.4 dropout",
"value": "0.4",
"context": "A dropout rate of 0.4 for the FFNNs."
},
{
"name": "0.5 dropout",
"value": "0.5",
"context": "A dropout rate of 0.5 for the input embeddings."
},
{
"name": "model spans",
"value": "8 words",
"context": "Model spans up to 8 words for constructing the knowledge graph."
}
],
"loss_functions": [],
"regularization_techniques": []
}
}
```

### Var 2

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
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

### Response:
```

---

```
---
- text_contains_entities: true
- entities:
    - entity1:
        name: BiLSTM
        type: model
        has_parameters: true
        parameters:
            - parameter1:
                name: hidden layers
                value: 1
                context: 200
    - entity2:
        name: FFNN
        type: model
        has_parameters: true
        parameters:
            - parameter2:
                name: hidden layers
                value: 2
                context: 150
    - entity3:
        name: LSTM
        type: model
        has_parameters: true
        parameters:
            - parameter3:
                name: hidden layers
                value: 1
                context: 200
    - entity4:
        name: dropout
        type: regularization technique
        has_parameters: true
        parameters:
            - parameter4:
                name: LSTM
                value: 0.4
                context: variational
    - entity5:
        name: dropout
        type: regularization technique
        has_parameters: true
        parameters:
            - parameter5:
                name: FFNN
                value: 0.4
                context: dropout
    - entity6:
        name: dropout
        type: regularization technique
        has_parameters: true
        parameters:
            - parameter6:
                name: embeddings
                value: 0.5
                context: input
    - entity7:
        name: spans
        type: method
        has_parameters: false
    - entity8:
        name: coreference resolution
        type: method
        has_parameters: false
    - entity9:
        name: relation extraction
        type: method
        has_parameters: false
    - entity10:
        name: beam pruning
        type: method
        has_parameters: true
        parameters:
            - parameter10:
                name: coreference resolution
                value: 0.3
                context: lambda
    - entity11:
        name: beam pruning
        type: method
        has_parameters: true
        parameters:
            - parameter11:
                name: relation extraction
                value: 0.4
                context: lambda
    - entity12:
        name: knowledge graph
        type: method
        has_parameters: false
...
```

### Var 3

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

Only include entities that are of type dataset, model, method, loss function, or regularization technique. Do not output entities that are of another type. Do not include entities of type task, metric, library, software, or API.
Only produce output in the YAML format specified above. Output no additional text.

### Response:
```

### Var 4

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: In the context of machine learning and related fields, what are the most important datasets, models, methods, loss functions, regularization techniques mentioned in the LaTeX Input Text below? What are their parameters and values?

[LaTeX Input Text start]
We divide our 438 annotated documents into training (70%), validation (30%) and test set (30%).
The base document representation of our model is formed by SciBERT-base [4] and BiLSTM with 128-d hidden state. We use a dropout of 0.2 after BiLSTM embeddings. All feedforward networks are composed of two hidden layers, each of dimension 128 with gelu activation and with a dropout of 0.2 between layers. For additive attention layer in span representation, we collapse the token embeddings to scalars by passing through the feedforward layer with 128-d hidden state and performing a softmax. We train our model for 30 epochs using Adam optimizer with 1e-3 as learning rate for all non BERT weights and 2e-5 for BERT weights. We use early stopping with a patience value of 7 on the validation set using relation extraction F1 score. All our models were trained using 48Gb Quadro RTX 8000 GPUs. The multitask model takes approximately 3 hrs to train.

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

Only produce output in the YAML format specified above. Output no additional text. Only include the most important datasets, models, methods, loss functions, regularization techniques mentioned in the LaTeX Input Text. Then end your response.

ASSISTANT: 
---
text_contains_entities:
```

```
 true
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

Datasets:
- e1: PubMed
- e2: CiteSeer

Models:
- e3: SciBERT-base
- e4: BiLSTM
- e5: Feedforward Networks

Methods:
- e6: Additive Attention
- e7: Span Representation

Loss Functions:
- e8: Cross-Entropy

Regularization Techniques:
- e9: Dropout
- e10: Early Stopping
```

### Var 5

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: In the context of machine learning and related fields, what are the most important entities (datasets, models, methods, loss functions, regularization techniques) mentioned in the LaTeX Input Text below? What are their parameters and values?

[LaTeX Input Text start]
We train our model for 30 epochs using Adam optimizer with 1e-3 as learning rate for all non BERT weights and 2e-5 for BERT weights. We use early stopping with a patience value of 7 on the validation set using relation extraction F1 score. All our models were trained using 48Gb Quadro RTX 8000 GPUs. The multitask model takes approximately 3 hrs to train.
[LaTeX Input Text end]

Answer in the following YAML format (everything in angle brackets (<example>) is a placeholder, everything divided by slashes (like/this) are options.

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

Only produce output in the YAML format specified above. Output no additional text. Only include the most important datasets, models, methods, loss functions, regularization techniques mentioned in the LaTeX Input Text.

ASSISTANT: 
---
text_contains_entities:
```

```
 true
entities:
- entity<1>:
  id: e1
  name: "epochs"
  type: dataset
  has_parameters: false
- entity<2>:
  id: e2
  name: "Adam optimizer"
  type: model
  has_parameters: false
- entity<3>:
  id: e3
  name: "learning rate"
  type: parameter
  has_parameters: true
  parameters:
  - parameter<1>:
      id: p1
      name: "non BERT weights"
      has_values: true
      values:
      - value<1>:
          value_id: v1
          value: "1e-3"
          context: "for all non BERT weights"
          context_id: c1
- entity<4>:
  id: e4
  name: "BERT weights"
  type: parameter
  has_parameters: true
  parameters:
  - parameter<2>:
      id: p2
      name: "validation set"
      has_values: true
      values:
      - value<2>:
          value_id: v2
          value: "7"
          context: "with a patience value of 7"
          context_id: c2
- entity<5>:
  id: e5
  name: "early stopping"
  type: method
  has_parameters: false
- entity<6>:
  id: e6
  name: "F1 score"
  type: loss function
  has_parameters: false
- entity<7>:
  id: e7
  name: "multitask model"
  type: model
  has_parameters: false
- entity<8>:
  id: e8
  name: "Quadro RTX 8000 GPUs"
  type: hardware
  has_parameters: false
- entity<9>:
  id: e9
  name: "3 hrs"
  type: parameter
  has_parameters: true
  parameters:
  - parameter<3>:
      id: p3
      name: "training time"
      has_values: true
      values:
```

### Var 6:

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: In the context of machine learning and related fields, what are the entities (allowed types: datasets, models, methods, loss functions, regularization techniques) mentioned in the LaTeX Input Text below? What are their parameters and values?

[LaTeX Input Text start]
We divide our 438 annotated documents into training (70%), validation (30%) and test set (30%).
The base document representation of our model is formed by SciBERT-base [4] and BiLSTM with 128-d hidden state. We use a dropout of 0.2 after BiLSTM embeddings. All feedforward networks are composed of two hidden layers, each of dimension 128 with gelu activation and with a dropout of 0.2 between layers. For additive attention layer in span representation, we collapse the token embeddings to scalars by passing through the feedforward layer with 128-d hidden state and performing a softmax.  We train our model for 30 epochs using Adam optimizer with 1e-3 as learning rate for all non BERT weights and 2e-5 for BERT weights. We use early stopping with a patience value of 7 on the validation set using relation extraction F1 score. All our models were trained using 48Gb Quadro RTX 8000 GPUs. The multitask model takes approximately 3 hrs to train.
[LaTeX Input Text end]

Answer in the following YAML format (everything in angle brackets (<example>) is a placeholder, everything divided by slashes (like/this) are options.

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

Only produce output in the YAML format specified above. Output no additional text. Only include entities (of type datasets, models, methods, loss functions, or regularization techniques), parameters and values that are mentioned in the LaTeX Input Text.

ASSISTANT: 
---
text_contains_entities:
```

```
 true
entities:
- entity<1>:
  id: e1
  name: "SciBERT-base"
  type: model
  has_parameters: true
  parameters:
  - parameter<1>:
      id: p1
      name: "Adam optimizer"
      has_values: true
      values:
      - value<1.1>:
          value_id: v1.1
          value: "1e-3"
          context: "all non BERT weights"
          context_id: c1.1
      - value<1.2>:
          value_id: v1.2
          value: "2e-5"
          context: "BERT weights"
          context_id: c1.2
- entity<2>:
  id: e2
  name: "BiLSTM"
  type: model
  has_parameters: true
  parameters:
  - parameter<2>:
      id: p2
      name: "dropout"
      has_values: true
      values:
      - value<2.1>:
          value_id: v2.1
          value: "0.2"
          context: "after BiLSTM embeddings"
          context_id: c2.1
- entity<3>:
  id: e3
  name: "Gelu activation"
  type: method
  has_parameters: false
- entity<4>:
  id: e4
  name: "softmax"
  type: method
  has_parameters: false
- entity<5>:
  id: e5
  name: "early stopping"
  type: method
  has_parameters: true
  parameters:
  - parameter<5.1>:
      id: p5.1
      name: "patience value"
      has_values: true
      values:
      - value<5.1.1>:
          value_id: v5.1.1
          value: "7"
          context: "on the validation set"
          context_id: c5.1.1
- entity<6>:
  id: e6
  name: "Quadro R
```