# GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k

## Note

works better w/ “free form” promt than author supplied

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
  - dataset:
      id: d<1>
      name: "<dataset name>"
      has_parameters: true
      parameters:
        - parameter<M>:
            id: p<1.M>
            name: "batch size"
            has_values: true
            values:
              - value<O>:
                  value_id: v<1.M.O>
                  value: "<dataset value>"
                  context: "<value context>"/null
                  context_id: c<1.M.O>/null
  - model:
      id: m<1>
      name: "<model name>"
      has_parameters: true
      parameters:
        - parameter<M>:
            id: p<1.M>
            name: "learning rate"
            has_values: true
            values:
              - value<O>:
                  value_id: v<1.M.O>
                  value: "<model value>"
                  context
```

(for deterministic preset w/ max output 2048)

```
---
text_contains_entities: true
entities:
  - dataset:
      id: d<N>
      name: "<dataset name>"
      has_parameters: true
      parameters:
        - parameter<M>:
            id: p<N.M>
            name: "<dataset parameter>"
            has_values: true
            values:
              - value<O>:
                  value_id: v<N.M.O>
                  value: "<dataset value>"
                  context: "<dataset context>"/null
                  context_id: c<N.M.O>/null
  - model:
      id: m<N>
      name: "<model name>"
      has_parameters: true
      parameters:
        - parameter<M>:
            id: p<N.M>
            name: "<model parameter>"
            has_values: true
            values:
              - value<O>:
                  value_id: v<N.M.O>
                  value: "<model value>"
                  context: "<model context>"/null
                  context_id: c<N.M.O>/null
  - method:
      id: m<N>
      name: "<method name>"
      has_parameters: true
      parameters:
        - parameter<M>:
            id: p<N.M>
            name: "<method parameter>"
            has_values: true
            values:
              - value<O>:
                  value_id: v<N.M.O>
                  value: "<method value>"
                  context: "<method context>"/null
                  context_id: c<N.M.O>/null
  - loss function:
      id: l<N>
      name: "<loss function name>"
      has_parameters: true
      parameters:
        - parameter<M>:
            id: p<N.M>
            name: "<loss function parameter>"
            has_values: true
            values:
              - value<O>:
                  value_id: v<N.M.O>
                  value: "<loss function value>"
                  context: "<loss function context>"/null
                  context_id: c<N.M.O>/null
  - regularization technique:
      id: r<N>
      name: "<regularization technique name>"
      has_parameters: true
      parameters:
        - parameter<M>:
            id: p<N.M>
            name: "<regularization technique parameter>"
            has_values: true
            values:
              - value<O>:
                  value_id: v<N.M.O>
                  value: "<regularization technique value>"
                  context: "<regularization technique context>"/null
                  context_id: c<N.M.O>/null
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
- text_contains_entities: true
- entities (datasets, models, methods, loss functions, regularization techniques):
    - entity1:
        name: LSTM
        type: Model
        has_parameters: true
        parameters:
            - parameter1:
                name: dropout
                value: 0.4
                context: None
    - entity2:
        name: FFNN
        type: Model
        has_parameters: true
        parameters:
            - parameter2:
                name: dropout
                value: 0.4
                context: None
    - entity3:
        name: SpanPrediction
        type: Method
        has_parameters: true
        parameters:
            - parameter3:
                name: beam_pruning_lambda_c
                value: 0.3
                context: None
    - entity4:
        name: KnowledgeGraph
        type: Method
        has_parameters: true
        parameters:
            - parameter4:
                name: beam_pruning_lambda_r
                value: 0.4
                context: None
    - entity5:
        name: CoreferenceResolution
        type: Method
        has_parameters: true
        parameters:
            - parameter5:
                name: beam_pruning_lambda_c
                value: 0.3
                context: None
    - entity6:
        name: RelationExtraction
        type: Method
        has_parameters: true
        parameters:
            - parameter6:
                name: beam_pruning_lambda_r
                value: 0.4
                context: None
    - entity7:
        name: KnowledgeGraphConstructor
        type: Method
        has_parameters: true
        parameters:
            - parameter7:
                name: beam_pruning_lambda_c
                value: 0.3
                context: None
    - entity8:
        name: CoreferenceResolution
        type: Method
        has_parameters: true
        parameters:
            - parameter8:
                name: beam_pruning_lambda_r
                value: 0.4
                context: None
    - entity9:
        name: RelationExtraction
        type: Method
        has_parameters: true
        parameters:
            - parameter9:
                name: beam_pruning_lambda_c
                value: 0.3
                context: None
    - entity10:
        name: KnowledgeGraphConstructor
        type: Method
        has_parameters: true
        parameters:
            - parameter10:
                name: beam_pruning_lambda_r
                value: 0.4
                context: None
    - entity11:
        name: CoreferenceResolution
        type: Method
        has_parameters: true
        parameters:
            - parameter11:
                name: beam_pruning_lambda_c
                value: 0.3
                context: None
    - entity12:
        name: RelationExtraction
        type: Method
        has_parameters: true
        parameters:
            - parameter12:
                name: beam_pruning_lambda_r
                value: 0.4
                context: None
    - entity13:
        name: KnowledgeGraphConstructor
        type: Method
        has_parameters: true
        parameters:
            - parameter13:
                name: beam_pruning_lambda_c
                value: 0.3
                context: None
    - entity14:
        name: CoreferenceResolution
        type: Method
        has_parameters: true
        parameters:
            - parameter14:
                name: beam_pruning_lambda_r
                value: 0.4
                context: None
    - entity15:
        name: RelationExtraction
        type: Method
        has_parameters: true
        parameters:
            - parameter15:
                name: beam_pruning_lambda_c
                value: 0.3
                context: None
    - entity16:
        name: KnowledgeGraphConstructor
        type: Method
        has_parameters: true
        parameters:
            - parameter16:
                name: beam_pruning_lambda_r
                value: 0.4
                context: None
    - entity17:
        name: CoreferenceResolution
        type: Method
        has_parameters: true
        parameters:
            - parameter17:
                name: beam_pruning_lambda_c
                value: 0.3
                context: None
    - entity18:
        name: RelationExtraction
        type: Method
        has_parameters: true
        parameters:
            - parameter18:
                name: beam_pruning_lambda_r
                value: 0.4
                context: None
    - entity19:
        name: KnowledgeGraphConstructor
        type: Method
        has_parameters: true
        parameters:
            - parameter19:
                name: beam_pruning_lambda_c
                value: 0.3
                context: None
    - entity20:
        name: CoreferenceResolution
        type: Method
        has_parameters: true
        parameters:
            - parameter20:
                name: beam_pruning_lambda_r
                value: 0.4
                context: None
    - entity21:
        name: RelationExtraction
        type: Method
        has_parameters: true
        parameters:
            - parameter21:
                name: beam_pruning_lambda_c
                value: 0.3
                context: None
    - entity22:
        name: KnowledgeGraphConstructor
        type: Method
        has_parameters: true
        parameters:
            - parameter2
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
- text_contains_entities: true
- entities (datasets, models, methods, loss functions, regularization techniques):
    - entity1:
        name: LSTM
        type: NeuralNetwork
        has_parameters: true
        parameters:
            - parameter1:
                name: dropout
                value: 0.4
                context: dropout
    - entity2:
        name: FFNN
        type: NeuralNetwork
        has_parameters: true
        parameters:
            - parameter2:
                name: dropout
                value: 0.4
                context: dropout
    - entity3:
        name: BiLSTM
        type: NeuralNetwork
        has_parameters: true
        parameters:
            - parameter3:
                name: dropout
                value: 0.4
                context: dropout
    - entity4:
        name: CoreferenceResolution
        type: NeuralNetwork
        has_parameters: true
        parameters:
            - parameter4:
                name: lambda_c
                value: 0.3
                context: lambda_c
    - entity5:
        name: RelationExtraction
        type: NeuralNetwork
        has_parameters: true
        parameters:
            - parameter5:
                name: lambda_r
                value: 0.4
                context: lambda_r
    - entity6:
        name: KnowledgeGraph
        type: NeuralNetwork
        has_parameters: true
        parameters:
            - parameter6:
                name: max_span_length
                value: 8
                context: max_span_length
    - entity7:
        name: BeamPruning
        type: NeuralNetwork
        has_parameters: true
        parameters:
            - parameter7:
                name: lambda_c
                value: 0.3
                context: lambda_c
            - parameter8:
                name: lambda_r
                value: 0.4
                context: lambda_r
            - parameter9:
                name: beam_size
                value: 10
                context: beam_size
            - parameter10:
                name: max_iterations
                value: 10
                context: max_iterations
            - parameter11:
                name: max_nodes
                value: 100
                context: max_nodes
            - parameter12:
                name: max_edges
                value: 1000
                context: max_edges
            - parameter13:
                name: max_depth
                value: 10
                context: max_depth
            - parameter14:
                name: max_width
                value: 10
                context: max_width
            - parameter15:
                name: max_width_factor
                value: 1.5
                context: max_width_factor
            - parameter16:
                name: max_depth_factor
                value: 1.5
                context: max_depth_factor
            - parameter17:
                name: max_depth_factor_increase
                value: 0.5
                context: max_depth_factor_increase
            - parameter18:
                name: max_width_factor_increase
                value: 0.5
                context: max_width_factor_increase
            - parameter19:
                name: max_width_factor_decrease
                value: 0.5
                context: max_width_factor_decrease
            - parameter20:
                name: max_depth_factor_decrease
                value: 0.5
                context: max_depth_factor_decrease
            - parameter21:
                name: max_width_factor_increase
                value: 0.5
                context: max_width_factor_increase
            - parameter22:
                name: max_depth_factor_decrease
                value: 0.5
                context: max_depth_factor_decrease
            - parameter23:
                name: max_width_factor_increase
                value: 0.5
                context: max_width_factor_increase
            - parameter24:
                name: max_depth_factor_decrease
                value: 0.5
                context: max_depth_factor_decrease
            - parameter25:
                name: max_width_factor_increase
                value: 0.5
                context: max_width_factor_increase
            - parameter26:
                name: max_depth_factor_decrease
                value: 0.5
                context: max_depth_factor_decrease
            - parameter27:
                name: max_width_factor_increase
                value: 0.5
                context:
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

---

```
- text_contains_entities: true
- entities (datasets, models, methods, loss functions, regularization techniques):
    - dataset:
        name: <dataset name>
        has_parameters: true
        parameters:
            - parameter<N>:
                name: <parameter name>
                value: <parameter value>
                context: <value context>
...
    - model:
        name: <model name>
        has_parameters: true
        parameters:
            - parameter<N>:
                name: <parameter name>
                value: <parameter value>
                context: <value context>
...
    - method:
        name: <method name>
        has_parameters: true
        parameters:
            - parameter<N>:
                name: <parameter name>
                value: <parameter value>
                context: <value context>
...
    - loss_function:
        name: <loss function name>
        has_parameters: true
        parameters:
            - parameter<N>:
                name: <parameter name>
                value: <parameter value>
                context: <value context>
...
    - regularization_technique:
        name: <regularization technique name>
        has_parameters: true
        parameters:
            - parameter<N>:
                name: <parameter name>
                value: <parameter value>
                context: <value context>
...
Note: The entities mentioned in the LaTeX Input Text are not included in the output as they are not of type dataset, model, method, loss function, or regularization technique.
```