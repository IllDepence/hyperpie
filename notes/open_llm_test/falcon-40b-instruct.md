# tiiuae/falcon-40b-instruct

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
  - entity1:
      id: e1
      name: "Entity 1"
      type: dataset
      has_parameters: true
      parameters:
        - parameter1:
            id: p1
            name: "Parameter 1"
            has_values: true
            values:
              - value1:
                  value_id: v1
                  value: "Value 1"
                  context: "Context 1"
  - entity2:
      id: e2
      name: "Entity 2"
      type: model
      has_parameters: true
      parameters:
        - parameter2:
            id: p2
            name: "Parameter 2"
            has_values: false
  - entity3:
      id: e3
      name: "Entity 3"
      type: method
      has_parameters: false
      parameters: null
  - entity4:
      id: e4
      name: "Entity 4"
      type: loss function
      has_parameters: false
      parameters: null
  - entity5:
      id: e5
      name: "Entity 5"
      type: regularization technique
      has_parameters
```

(for deterministic preset w/ max output 2048)

```
text_contains_entities: true

entities:
  - entity1:
      id: e1
      name: "Dataset 1"
      type: dataset
      has_parameters: false
      parameters: []
  - entity2:
      id: e2
      name: "Model 1"
      type: model
      has_parameters: true
      parameters:
        - parameter1:
            id: p1
            name: "Learning rate"
            has_values: true
            values:
              - value1:
                  value_id: v1
                  value: "0.001"
                  context: "Initial learning rate"
                  context_id: c1/null
  - entity3:
      id: e3
      name: "Method 1"
      type: method
      has_parameters: false
      parameters: []
  - entity4:
      id: e4
      name: "Loss function 1"
      type: loss function
      has_parameters: false
      parameters: []
  - entity5:
      id: e5
      name: "Regularization technique 1"
      type: regularization technique
      has_parameters: false
      parameters: []


The entities mentioned in the LaTeX Input Text are:

- Dataset 1
- Model 1
- Method 1
- Loss function 1
- Regularization technique 1

All of them are of type dataset, model, method, loss function, or regularization technique. None of them are of type task, metric, library, software, or API.
```

### Var1

(user/assistant)

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: In the context of machine learning and related fields,  what (if any) are the entities (datasets, models, methods, loss functions, regularization techniques) mentioned in the LaTeX Input Text below? What (if any) are their parameters and values?

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

ASSISTANT: 
```

---

```
The entities mentioned in the LaTeX Input Text are:

- dataset: "Coref-TACL"
- model: "Coref-TACL"
- method: "Coref-TACL"
- loss function: "Cross-Entropy"
- regularization technique: "Dropout"

All of these entities have parameters and values. The parameters and values for each entity are as follows:

- dataset:
  - parameter: "Coref-TACL"
  - value: "Coref-TACL"
  - context: "Coref-TACL"
- model:
  - parameter: "Coref-TACL"
  - value: "Coref-TACL"
  - context: "Coref-TACL"
- method:
  - parameter: "Coref-TACL"
  - value: "Coref-TACL"
  - context: "Coref-TACL"
- loss function:
  - parameter: "Cross-Entropy"
  - value: "Cross-Entropy"
  - context: "Cross-Entropy"
- regularization technique:
  - parameter: "Dropout"
  - value: "Dropout"
  - context: "Dropout"

All of these entities have parameters and values, and all of the parameters and values are included in the output.
```

### Var2

(user/assistant simple)

```
User: In the context of machine learning and related fields,  what (if any) are the entities (datasets, models, methods, loss functions, regularization techniques) mentioned in the LaTeX Input Text below? What (if any) are their parameters and values?

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

Assistant: 
```

---

```
The entities mentioned in the LaTeX Input Text are:

- BiLSTM: a type of neural network architecture that uses two layers of LSTM (Long Short-Term Memory) units to process sequential data.
- FFNN: a type of neural network architecture that uses fully connected layers to process data.
- Variational dropout: a technique used to regularize neural networks by randomly dropping out some of the connections between layers.
- Dropout: a technique used to regularize neural networks by randomly dropping out some of the neurons in a layer.
- Input embeddings: a technique used to represent words as vectors of numbers, which can be used to train neural networks on text data.
- Coreference resolution: a task in natural language processing that involves identifying which words in a text refer to the same entity.
- Relation extraction: a task in natural language processing that involves identifying relationships between entities in a text.
- Knowledge graph: a collection of data that represents a domain of knowledge, including entities and their relationships.

The parameters and values of these entities are not explicitly mentioned in the LaTeX Input Text.
```

### Var 3

(simplified format)

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

```
---
- text_contains_entities: true
- entities:
  - entity1:
      name: "BiLSTM"
      type: "model"
      has_parameters: true
      parameters:
        - parameter1:
            name: "hidden_dim"
            value: 200
            context: "hidden layer"
        - parameter2:
            name: "dropout"
            value: 0.4
            context: "variational dropout"
        - parameter3:
            name: "num_layers"
            value: 1
            context: "layer"
        - parameter4:
            name: "num_units"
            value: 200
            context: "hidden unit"
        - parameter5:
            name: "num_units"
            value: 200
            context: "hidden unit"
        - parameter6:
            name: "num_units"
            value: 200
            context: "hidden unit"
        - parameter7:
            name: "num_units"
            value: 200
            context: "hidden unit"
        - parameter8:
            name: "num_units"
            value: 200
            context: "hidden unit"
        - parameter9:
            name: "num_units"
            value: 200
            context: "hidden unit"
        - parameter10:
            name: "num_units"
            value: 200
            context: "hidden unit"
        - parameter11:
            name: "num_units"
            value: 200
            context: "hidden unit"
        - parameter12:
            name: "num_units"
            value: 200
            context: "hidden unit"
        - parameter13:
            name: "num_units"
            value: 200
            context: "hidden unit"
        - parameter14:
            name: "num_units"
            value: 200
            context: "hidden unit"
        - parameter15:
            name: "num_units"
            value: 200
            context: "hidden unit"
        - parameter16:
            name:
...
```

(same prompt but w/ default params)

```
---
- text_contains_entities: true
- entities (datasets, models, methods, loss functions, regularization techniques):
  - entity1:
      name: "BiLSTM"
      type: "model"
      has_parameters: true
      parameters:
        - parameter1:
            name: "num_layers"
            value: "1"
            context: "num_layers"
        - parameter2:
            name: "hidden_dim"
            value: "200"
            context: "hidden_dim"
        - parameter3:
            name: "dropout"
            value: "0.4"
            context: "dropout"
        - parameter4:
            name: "num_spans"
            value: "8"
            context: "num_spans"
        - parameter5:
            name: "beam_pruning"
            value: "0.3"
            context: "beam_pruning"
        - parameter6:
            name: "coreference_resolution"
            value: "0.3"
            context: "coreference_resolution"
        - parameter7
```

### Var4

(simplified format, user/assistant prompt)

```
User: In the context of machine learning and related fields, what (if any) are the entities (datasets, models, methods, loss functions, regularization techniques) mentioned in the LaTeX Input Text below? What (if any) are their parameters and values?

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

Assistant: 
```

```
Entities mentioned in the LaTeX Input Text are:

Dataset:
- knowledge graph

Models:
- 1-layer BiLSTM with 200-dimensional hidden layers
- 2-layer FFNN with 150-dimensional hidden layers

Methods:
- variational dropout
- dropout
- beam pruning
- coreference resolution
- relation extraction

Loss functions:
- cross-entropy loss

Regularization techniques:
- dropout

The following are the parameters and their values:

Beam pruning:
- lambda_C = 0.3 for coreference resolution
- lambda_R = 0.4 for relation extraction

Dropout:
- 0.4 for LSTMs
- 0.4 for FFNNs
- 0.5 for input embeddings

Coreference resolution:
- lambda_C = 0.3

Relation extraction:
- lambda_R = 0.4
```
