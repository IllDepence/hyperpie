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
