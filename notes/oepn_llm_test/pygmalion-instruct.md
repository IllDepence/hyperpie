# AlpinDale/pygmalion-instruct

## Notes

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


(default params)
```
---
text_contains_entities: true
entities:
- entity1:
  name: "Lee2017EndtoendNC"
  type: "implementation"
  has_parameters: false
- entity2:
  name: "BiLSTM"
  type: "model"
  has_parameters: true
- entity3:
  name: "FFNN"
  type: "model"
  has_parameters: true
- entity4:
  name: "dropout"
  type: "regularization technique"
  has_parameters: false
- entity5:
  name: "coreference resolution"
  type: "method"
  has_parameters: true
- entity6:
  name: "relation extraction"
  type: "method"
  has_parameters: true
- entity7:
  name: "beam pruning"
  type: "method"
  has_parameters: true
- entity8:
  name: "knowledge graph construction"
  type: "method"
  has_parameters: true
- entity9:
```

---

(deterministic preset w/ max len 2048)
```
---
text_contains_entities: true
entities:
- entity1:
  name: "Lee2017EndtoendNC"
  type: "implementation"
  has_parameters: true
  parameters:
    - parameter1:
      name: "1 layer BiLSTM"
      value: "200-dimensional hidden layers"
      context: "context of the implementation"
- entity2:
  name: "FFNN"
  type: "method"
  has_parameters: true
  parameters:
    - parameter2:
      name: "2 hidden layers of 150 dimensions each"
      value: "150-dimensional hidden layers"
      context: "context of the method"
- entity3:
  name: "variational dropout"
  type: "regularization technique"
  has_parameters: true
  parameters:
    - parameter3:
      name: "0.4 variational dropout"
      value: "0.4 dropout for the LSTMs"
      context: "context of the regularization technique"
- entity4:
  name: "dropout"
  type: "regularization technique"
  has_parameters: true
  parameters:
    - parameter4:
      name: "0.4 dropout"
      value: "0.4 dropout for the FFNNs"
      context: "context of the regularization technique"
- entity5:
  name: "input embeddings"
  type: "method"
  has_parameters: true
  parameters:
    - parameter5:
      name: "150-dimensional hidden layers"
      value: "150-dimensional hidden layers"
      context: "context of the method"
- entity6:
  name: "beam pruning"
  type: "method"
  has_parameters: true
  parameters:
    - parameter6:
      name: "lambda C"
      value: "0.3"
      context: "context of the method"
- entity7:
  name: "relation extraction"
  type: "method"
  has_parameters: true
  parameters:
    - parameter7:
      name: "lambda R"
      value: "0.4"
      context: "context of the method"
- entity8:
  name: "constructing the knowledge graph"
  type: "method"
  has_parameters: true
  parameters:
    - parameter8:
      name: "lambda C"
      value: "0.3"
      context: "context of the method"
- entity9:
  name: "spans up to 8 words"
  type: "parameter"
  has_parameters: true
  parameters:
    - parameter9:
      name: "spans up to 8 words"
      value: "spans up to 8 words"
      context: "context of the parameter"
- entity10:
  name: "coreference resolution"
  type: "parameter"
  has_parameters: true
  parameters:
    - parameter10:
      name: "lambda C"
      value: "0.3"
      context: "context of the parameter"
- entity11:
  name: "relation extraction"
  type: "parameter"
  has_parameters: true
  parameters:
    - parameter11:
      name: "lambda R"
      value: "0.4"
      context: "context of the parameter"
- entity12:
  name: "constructing the knowledge graph"
  type: "parameter"
  has_parameters: true
  parameters:
    - parameter12:
      name: "lambda C"
      value: "0.3"
      context: "context of the parameter"
- entity13:
  name: "spans up to 8 words"
  type: "parameter"
  has_parameters: true
  parameters:
    - parameter13:
      name: "spans up to 8 words"
      value: "spans up to 8 words"
      context: "context of the parameter"
- entity14:
  name: "coreference resolution"
  type: "parameter"
  has_parameters: true
  parameters:
    - parameter14:
      name: "lambda C"
      value: "0.3"
      context: "context of the parameter"
- entity15:
  name: "relation extraction"
  type: "parameter"
  has_parameters: true
  parameters:
    - parameter15:
      name: "lambda R"
      value: "0.4"
      context: "context of the parameter"
- entity16:
  name: "constructing the knowledge graph"
  type: "parameter"
  has_parameters: true
  parameters:
    - parameter16:
      name: "lambda C"
      value: "0.3"
      context: "context of the parameter"
- entity17:
  name: "spans up to 8 words"
  type: "parameter"
  has_parameters: true
  parameters:
    - parameter17:
      name: "spans up to 8 words"
      value: "spans up to 8 words"
      context: "context of the parameter"
- entity18:
  name: "coreference resolution"
  type: "parameter"
  has_parameters: true
  parameters:
    - parameter18:
      name: "lambda C"
      value: "0.3"
      context: "context of the parameter"
- entity19:
  name: "relation extraction"
  type: "parameter"
  has_parameters: true
  parameters:
    - parameter19:
      name: "lambda R"
      value: "0.4"
      context: "context of the parameter"
- entity20:
  name: "constructing the knowledge graph"
  type: "parameter"
  has_parameters: true
  parameters:
    - parameter20:
      name: "lambda C"
      value: "0.3"
      context: "context of the parameter"
- entity21:
  name: "spans up to 8 words"
  type: "parameter"
  has_parameters: true
  parameters:
    - parameter21:
      name: "spans up to 8 words"
      value: "spans up to 8 words"
      context: "context of the parameter"
- entity22:
  name: "coreference resolution"
  type: "parameter"
  has_parameters: true
  parameters:
    - parameter22:
      name: "lambda C"
      value: "0.3"
      context: "context of the parameter"
- entity23:
  name: "relation extraction"
  type: "parameter"
  has_parameters: true
  parameters:
    - parameter23:
      name: "lambda R"
      value: "0.4"
      context: "context of the parameter"
- entity24:
  name: "constructing the knowledge graph"
  type: "parameter"
  has_parameters: true
  parameters:
    - parameter24:
      name: "lambda C"
      value: "0.3"
      context: "context of the parameter"
- entity25:
  name: "spans up to 8 words"
  type: "parameter"
  has_parameters: true
  parameters:
    - parameter25:
      name: "spans up to 8 words"
      value: "spans up to 8 words"
      context: "context of the parameter"
- entity26:
  name: "coreference resolution"
  type: "method"
  has_parameters: true
  parameters:
    - parameter26:
      name  //<-- cuda out of memory error at this point
```

### Alt 1

Adding “Limit our output to the most significant entities.” to the prompt does not help overly long output.