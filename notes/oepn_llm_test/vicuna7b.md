# eachadea/vicuna-7b-1.1

## Notes

Seems to not know when to stop

## Prompts

### Original (default params)

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
text\_contains\_entities: true
entities:
- entity1:
    name: BiLSTM
    type: neural network
    has\_parameters: true
    parameters:
        - parameter1:
            name: number of hidden layers
            value: 200
            context: model
- entity2:
    name: FFNN
    type: neural network
    has\_parameters: true
    parameters:
        - parameter2:
            name: number of hidden layers
            value: 150
            context: model
- entity3:
    name: dropout
    type: regularization technique
    has\_parameters: true
    parameters:
        - parameter3:
            name: dropout rate
            value: 0.4
            context: model
- entity4:
    name: coreference resolution
    type: task
    has\_parameters: true
    parameters:
        - parameter4:
            name: dropout rate
            value: 0.3
            context: model
- entity5:
    name: relation extraction
    type: task
    has
```

### same as above but w/ max output 2048

```
---
text\_contains\_entities: true
entities:
- entity1:
    name: 1 layer BiLSTM
    type: model
    has\_parameters: true
    parameters:
        - parameter1:
            name: number of hidden layers
            value: 200
            context: input
- entity2:
    name: 2 hidden layers FFNN
    type: model
    has\_parameters: true
    parameters:
        - parameter2:
            name: number of dimensions
            value: 150
            context: input
- entity3:
    name: dropout
    type: regularization technique
    has\_parameters: true
    parameters:
        - parameter3:
            name: dropout rate
            value: 0.4
            context: input
- entity4:
    name: coreference resolution
    type: task
    has\_parameters: true
    parameters:
        - parameter4:
            name: dropout rate
            value: 0.3
            context: input
- entity5:
    name: relation extraction
    type: task
    has\_parameters: true
    parameters:
        - parameter5:
            name: dropout rate
            value: 0.4
            context: input
- entity6:
    name: beam pruning
    type: method
    has\_parameters: true
    parameters:
        - parameter6:
            name: dropout rate
            value: 0.3
            context: input
- entity7:
    name: knowledge graph construction
    type: method
    has\_parameters: true
    parameters:
        - parameter7:
            name: dropout rate
            value: 0.4
            context: input
- entity8:
    name: span
    type: entity
    has\_parameters: false
    parameters: null
- entity9:
    name: input embedding
    type: entity
    has\_parameters: false
    parameters: null
- entity10:
    name: coreference resolution span
    type: entity
    has\_parameters: false
    parameters: null
- entity11:
    name: relation extraction span
    type: entity
    has\_parameters: false
    parameters: null
- entity12:
    name: relation type
    type: entity
    has\_parameters: false
    parameters: null
- entity13:
    name: relation name
    type: entity
    has\_parameters: false
    parameters: null
- entity14:
    name: relation source
    type: entity
    has\_parameters: false
    parameters: null
- entity15:
    name: relation target
    type: entity
    has\_parameters: false
    parameters: null
- entity16:
    name: relation type
    type: entity
    has\_parameters: false
    parameters: null
- entity17:
    name: relation name
    type: entity
    has\_parameters: false
    parameters: null
- entity18:
    name: relation source
    type: entity
    has\_parameters: false
    parameters: null
- entity19:
    name: relation target
    type: entity
    has\_parameters: false
    parameters: null
- entity20:
    name: knowledge graph
    type: entity
    has\_parameters: false
    parameters: null
- entity21:
    name: span
    type: entity
    has\_parameters: false
    parameters: null
- entity22:
    name: span
    type: entity
    has\_parameters: false
    parameters: null
- entity23:
    name: span
    type: entity
    has\_parameters: false
    parameters: null
[...]
- entity59:
    name: span
    type
```

### with Aplaca style prompt

```
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


(default params)

```
---
text_contains_entities: true
entities:
- entity1:
    name: 1 layer BiLSTM
    type: model
    has_parameters: true
    parameters:
        - parameter1: 200
- entity2:
    name: 2 hidden layers
    type: number of hidden layers
    has_parameters: true
    parameters:
        - parameter2: 150
- entity3:
    name: dropout
    type: regularization technique
    has_parameters: true
    parameters:
        - parameter3: 0.4
    context: Lee2017EndtoendNC
- entity4:
    name: coreference resolution
    type: task
    has_parameters: true
    parameters:
        - parameter4: 0.3
    context: Lee2017EndtoendNC
- entity5:
    name: relation extraction
    type: task
    has_parameters: true
    parameters:
        - parameter5: 0.4
    context: Lee2017EndtoendNC
- entity
```

(deterministic preset w/ 2048 max output)

```
---
text\_contains\_entities: true
entities:
- entity1:
    name: BiLSTM
    type: model
    has\_parameters: true
    parameters:
        - parameter1:
            name: hidden\_layer\_size
            value: 200
            context: hidden layers
- entity2:
    name: FFNN
    type: model
    has\_parameters: true
    parameters:
        - parameter2:
            name: hidden\_layer\_size
            value: 150
            context: hidden layers
- entity3:
    name: dropout
    type: regularization technique
    has\_parameters: true
    parameters:
        - parameter3:
            name: rate
            value: 0.4
            context: dropout rates
- entity4:
    name: span
    type: entity
    has\_parameters: false
- entity5:
    name: coreference\_resolution
    type: task
    has\_parameters: false
- entity6:
    name: relation\_extraction
    type: task
    has\_parameters: false
- entity7:
    name: variational\_dropout
    type: regularization technique
    has\_parameters: true
    parameters:
        - parameter7:
            name: rate
            value: 0.4
- entity8:
    name: beam\_pruning
    type: method
    has\_parameters: true
    parameters:
        - parameter8:
            name: dropout\_ratio
            value: 0.3
            context: dropout rates
        - parameter8:
            name: dropout\_ratio
            value: 0.4
            context: dropout rates
- entity9:
    name: knowledge\_graph
    type: method
    has\_parameters: false
- entity10:
    name: span\_length
    type: entity
    has\_parameters: false
- entity11:
    name: task
    type: entity
    has\_parameters: false
- entity12:
    name: dataset
    type: entity
    has\_parameters: false
- entity13:
    name: loss\_function
    type: entity
    has\_parameters: false
- entity14:
    name: regularization\_technique
    type: entity
    has\_parameters: false
- entity15:
    name: input\_embedding
    type: entity
    has\_parameters: false
- entity16:
    name: beam\_width
    type: entity
    has\_parameters: false
- entity17:
    name: knowledge\_graph\_context
    type: entity
    has\_parameters: false
- entity18:
    name: span\_boundary
    type: entity
    has\_parameters: false
- entity19:
    name: relation\_type
    type: entity
    has\_parameters: false
- entity20:
    name: relation\_predicate
    type: entity
    has\_parameters: false
- entity21:
    name: relation\_head
    type: entity
    has\_parameters: false
- entity22:
    name: relation\_predicate\_context
    type: entity
    has\_parameters: false
- entity23:
    name: relation\_type\_context
    type: entity
    has\_parameters: false
- entity24:
    name: relation\_predicate\_context
    type: entity
    has\_parameters: false
- entity25:
    name: relation\_head\_context
    type: entity
    has\_parameters: false
- entity26:
    name: relation\_predicate
```

### complex output format

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

---
text\_contains\_entities: true
entities:
- entity1:
id: e1
name: "1 layer BiLSTM"
type: model
has\_parameters: true
parameters:
- parameter1:
id: p1
name: "hidden layers"
has\_values: true
values:
- value1:
value\_id: v1
value: "200"
context: "hidden layers"
context\_id: c1
- parameter2:
id: p2
name: "FFNNs"
has\_values: true
values:
- value2:
value\_id: v2
value: "2"
context: "FFNNs"
context\_id: c2
- parameter3:
id: p3
name: "variational dropout"
has\_values: true
values:
- value3:
value\_id: v3
value: "0.4"
context: "variational dropout"
context\_id: c3
- parameter4:
id: p4
name: "dropout"
has\_values: true
values:
- value4:
value\_id: v4
value: "0.4"
context: "dropout"
context\_id: c4
- parameter5:
id: p5
name: "input embeddings"
has\_values: true
values:
- value5:
value\_id: v5
value: "0.5"
context: "input embeddings"
context\_id: c5
- parameter6:
id: p6
name: "beam pruning"
has\_values: true
values:
- value6:
value\_id: v6
value: "0.3"
context: "beam pruning"
context\_id: c6
- parameter7:
id: p7
name: "knowledge graph construction"
has\_values: true
values:
- value7:
value\_id: v7
value: "0.4"
context: "knowledge graph construction"
context\_id: c7
- parameter8:
id: p8
name: "coreference resolution"
has\_values: true
values:
- value8:
value\_id: v8
value: "0
...
```