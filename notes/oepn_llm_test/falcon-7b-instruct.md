# tiiuae/falcon-7b-instruct

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

(default settings)

```
---
- text_contains_entities: true/false
- entities (datasets, models, methods, loss functions, regularization techniques):
    - entity1:
        name: entity1
        type: entity1
        has_parameters: true
        parameters:
            - parameter1: entity2
            - parameter2: entity1
            context: entity1/entity2
    - entity2:
        name: entity2
        type: entity2
        has_parameters: true
        parameters:
            - parameter1: entity3
            - parameter2: entity2
            context: entity2/entity3
    - entity3:
        name: entity3
        type: entity3
        has_parameters: true
        parameters:
            - parameter1: entity4
            - parameter2: entity3
            context: entity3/entity4
            ...A tutorial on how to use the <code>nltk</code> library in Python to generate natural language understanding applications is available at https://nbviewer.jupyter.org/~<a href="https://nbviewer.jupyter.org/~/examples/tutorial.ipynb" rel="noreferrer">https://nbview
```

(deterministic preset w/ 2048 max output)

```
The entities mentioned in the LaTeX Input Text are:

- `text_contains_entities` (true)
- `entities` (datasets, models, methods, loss functions, regularization techniques)
- `text_contains_entities` (true)
- `entities` (datasets, models, methods, loss functions, regularization techniques)
- `text_contains_entities` (true)
- `entities` (datasets, models, methods, loss functions, regularization techniques)
- `text_contains_entities` (true)
- `entities` (datasets, models, methods, loss functions, regularization techniques)
- `text_contains_entities` (true)
- `entities` (datasets, models, methods, loss functions, regularization techniques)
- `text_contains_entities` (true)
- `entities` (datasets, models, methods, loss functions, regularization techniques)
- `text_contains_entities` (true)
- `entities` (datasets, models, methods, loss functions, regularization techniques)
- `text_contains_entities` (true)
- `entities` (datasets, models, methods, loss functions, regularization techniques
...
```

### Var 1

prepend “Instruction:” → same as above

### Var 2

surround format block with [YAML format start/end] → same as above

### Var 3

```
[LaTeX Input Text start]
Our system extends the implementation and hyper-parameters from Lee2017EndtoendNC with the following adjustments. We use a 1 layer BiLSTM with 200-dimensional hidden layers. All the FFNNs have 2 hidden layers of 150 dimensions each. We use 0.4 variational dropout [15] for the LSTMs, 0.4 dropout for the FFNNs, and 0.5 dropout for the input embeddings. We model spans up to 8 words. For beam pruning, we use \(\lambda _{\text{C}}=0.3\) for coreference resolution and \(\lambda _{\text{R}}=0.4\) for relation extraction. For constructing the knowledge graph
[LaTeX Input Text end]

Entities (datasets, models, methods, loss functions, regularization techniques) and their parameters mentioned in the LaTeX Input Text above:
```

```
- Datasets: \textbf{Dataset 1}, \textbf{Dataset 2}, \textbf{Dataset 3}, \textbf{Dataset 4}, \textbf{Dataset 5}, \textbf{Dataset 6}, \textbf{Dataset 7}, \textbf{Dataset 8}, \textbf{Dataset 9}, ...
```
