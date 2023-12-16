# eachadea/vicuna-13b-1.1

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
text_contains\_entities: true
entities:
- entity1:
    name: BiLSTM
    type: model
    has_parameters: true
    parameters:
        - parameter1:
            name: hidden layers
            value: 200
            context: hyperparameter
        - parameter2:
            name: LSTM
            value: true
            context: model component
        - parameter3:
            name: FFNNs
            value: true
            context: model component
        - parameter4:
            name: dropout
            value: 0.4
            context: regularization technique
        - parameter5:
            name: beam pruning
            value: \lambda _{\text{C}}=0.3, \lambda _{\text{R}}=0.4
            context: hyperparameter
        - parameter6:
            name: coreference resolution
            value: true
            context: task
        - parameter7:
            name: relation extraction
            value: true
            context: task
- entity2:
    name: LSTM
    type: model

```

### same as above but w/ max output 2048

```

---

* text\_contains\_entities: true
* entities:
	* entity1:
		* name: BiLSTM
		* type: model
		* has\_parameters: true
		* parameters:
			* parameter1:
				* name: hidden\_layers
				* value: 2
				* context: 200
		* entity2:
			* name: FFNN
			* type: model
			* has\_parameters: true
			* parameters:
				* parameter1:
					* name: hidden\_layers
					* value: 2
					* context: 150
		* entity3:
			* name: dropout
			* type: regularization\_technique
			* has\_parameters: true
			* parameters:
				* parameter1:
					* name: layer
					* value: LSTM/FFNN
					* context: 0.4
			* entity4:
				* name: input\_embeddings
				* type: regularization\_technique
				* has\_parameters: true
				* parameters:
					* parameter1:
						* name: layer
						* value: input
						* context: 0.5
		* entity5:
			* name: coreference\_resolution
			* type: task
			* has\_parameters: true
			* parameters:
				* parameter1:
					* name: method
					* value: beam\_pruning
					* context: λ\_C=0.3
		* entity6:
			* name: relation\_extraction
			* type: task
			* has\_parameters: true
			* parameters:
				* parameter1:
					* name: method
					* value: beam\_pruning
					* context: λ\_R=0.4
		* entity7:
			* name: knowledge\_graph
			* type: task
			* has\_parameters: true
			* parameters:
				* parameter1:
					* name: method
					* value: construct\_graph
					* context: 
						* entity1: BiLSTM
						* entity2: FFNN
						* entity5: coreference\_resolution
						* entity6: relation\_extraction
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
* text\_contains\_entities: true
* entities:
	* entity1:
		* name: BiLSTM
		* type: Model
		* has\_parameters: true
		* parameters:
			* parameter1:
				* name: hidden\_layers
				* value: 2
				* context: layer\_size
		* entity2:
			* name: LSTM
			* type: Model
			* has\_parameters: true
			* parameters:
				* parameter1:
					* name: hidden\_layers
					* value: 2
					* context: layer\_size
		* entity3:
			* name: FFNN
			* type: Model
			* has\_parameters: true
			* parameters:
				* parameter1:
					* name: hidden\_layers
					* value: 2

```

(deterministic preset w/ 2048 max output)

```
* text\_contains\_entities: true
* entities:
	* entity1:
		* name: BiLSTM
		* type: model
		* has\_parameters: true
		* parameters:
			* parameter1:
				* name: hidden\_layers
				* value: 2
				* context: 200
		* entity2:
			* name: FFNN
			* type: model
			* has\_parameters: true
			* parameters:
				* parameter1:
					* name: hidden\_layers
					* value: 2
					* context: 150
		* entity3:
			* name: LSTM
			* type: model
			* has\_parameters: true
			* parameters:
				* parameter1:
					* name: hidden\_layers
					* value: 1
					* context: 200
		* entity4:
			* name: dropout
			* type: regularization\_technique
			* has\_parameters: true
			* parameters:
				* parameter1:
					* name: rate
					* value: 0.4
					* context: LSTM, FFNN
		* entity5:
			* name: beam\_pruning
			* type: method
			* has\_parameters: true
			* parameters:
				* parameter1:
					* name: coreference\_resolution
					* value: 0.3
					* context: beam\_pruning
			* entity6:
				* name: relation\_extraction
				* type: method
				* has\_parameters: true
				* parameters:
					* parameter1:
						* name: relation\_extraction
						* value: 0.4
						* context: beam\_pruning
		* entity7:
			* name: knowledge\_graph
			* type: method
			* has\_parameters: true
			* parameters:
				* parameter1:
					* name: construction
					* value: true
					* context: knowledge\_graph

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

* entity1:
	id: e1
	name: "1-layer BiLSTM"
	type: model
	has\_parameters: true
	parameters:
		- parameter1:
			id: p1
			name: "hidden layer size"
			has\_values: true
			values:
				- value1:
					value\_id: v1
					value: "200"
					context: "dimensional"
					context\_id: c1
		- parameter2:
			id: p2
			name: "number of hidden layers"
			has\_values: true
			values:
				- value1:
					value\_id: v1
					value: "2"
					context: "layers"
					context\_id: c1
...
```

### Var1

```
### Instruction:
In the context of machine learning and related fields, what are the most important datasets, models, methods, loss functions, regularization techniques mentioned in the LaTeX Input Text below? What are their parameters and values?

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

Only produce output in the YAML format specified above. Output no additional text. Only include the most important datasets, models, methods, loss functions, regularization techniques mentioned in the LaTeX Input Text. Then end your response.

### Response:
```

```


text\_contains\_entities: true

entities:
- entity1:
id: e1
name: "datasets"
type: dataset
has\_parameters: true
parameters:
- parameter1:
id: p1
name: "train\_dataset"
has\_values: true
values:
- value1:
value\_id: v1
value: "The Adult Income Dataset"
context: "The Adult Income Dataset"
context\_id: c1
- value2:
value\_id: v2
value: "The German Credit Dataset"
context: "The German Credit Dataset"
context\_id: c2
- value3:
value\_id: v3
value: "The Computers for Home Use Dataset"
context: "The Computers for Home Use Dataset"
context\_id: c3
...

- entity2:
id: e2
name: "models"
type: model
has\_parameters: true
parameters:
- parameter1:
id: p1
name: "model\_type"
has\_values: true
values:
- value1:
value\_id: v1
value: "BiLSTM"
context: "BiLSTM"
context\_id: c1
- value2:
value\_id: v2
value: "FFNN"
context: "FFNN"
context\_id: c2
- value3:
value\_id: v3
value: "LSTM"
context: "LSTM"
context\_id: c3
...

- entity3:
id: e3
name: "methods"
type: method
has\_parameters: true
parameters:
- parameter1:
id: p1
name: "attention\_mechanism"
has\_values: true
values:
- value1:
value\_id: v1
value: "self-attention"
context: "self-attention"
context\_id: c1
- value2:
value\_id: v2
value: "convolutional"
context: "convolutional"
context\_id: c2
- value3:
value\_id: v3
value: "recurrent"
context: "recurrent"
context\_id: c3
...

- entity4:
id: e4
name: "loss\_functions"
type: loss\_function
has\_parameters: true
parameters:
- parameter1:
id: p1
name: "loss\_function"
has\_values: true
values:
- value1:
value\_id: v1
value: "cross-entropy"
context: "cross-entropy"
context\_id: c1
- value2:
value\_id: v2
value: "mean\_squared\_error"
context: "mean\_squared\_error"
context\_id: c2
- value3:
value\_id: v3
value: "hinge\_loss"
context: "hinge\_loss"
context\_id: c3
...

- entity5:
id: e5
name: "regularization\_techniques"
type: regularization\_technique
has\_parameters: true
parameters:
- parameter1:
id: p1
name: "regularization\_technique"
has\_values: true
values:
- value1:
value\_id: v1
value: "dropout"
context: "dropout"
context\_id: c1
- value2:
value\_id: v2
value: "L2\_regularization"
context: "L2\_regularization"
context\_id: c2
- value3:
value\_id: v3
value: "weight\_decay"
context: "weight\_decay"
context\_id: c3
...

...

[LaTeX Input Text end]
```

### Var 2

```
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

```


text\_contains\_entities: true
entities:
- entity1:
id: e1
name: "1-layer BiLSTM"
type: model
has\_parameters: true
parameters:
- parameter1:
id: p1
name: "hidden layer size"
has\_values: true
values:
- value1:
value\_id: v1
value: "200"
context: null
context\_id: c1
- value2:
value\_id: v2
value: "150"
context: null
context\_id: c2
- value3:
value\_id: v3
value: "150"
context: null
context\_id: c3
- value4:
value\_id: v4
value: "200"
context: null
context\_id: c4
- value5:
value\_id: v5
value: "150"
context: null
context\_id: c5
- value6:
value\_id: v6
value: "150"
context: null
context\_id: c6
- value7:
value\_id: v7
value: "0.4"
context: null
context\_id: c7
- value8:
value\_id: v8
value: "0.4"
context: null
context\_id: c8
- value9:
value\_id: v9
value: "0.5"
context: null
context\_id: c9
- value10:
value\_id: v10
value: "0.5"
context: null
context\_id: c10
...

Note: The YAML format specified above is just an example. It may not be suitable for all use cases.
```

### Var3

```
### Instruction:
In the context of machine learning and related fields, what (if any) are the entities (datasets, models, methods, loss functions, regularization techniques) mentioned in the LaTeX Input Text below? What (if any) are their parameters and values?

[LaTeX Input Text start]
Our system extends the implementation and hyper-parameters from Lee2017EndtoendNC with the following adjustments. We use a 1 layer BiLSTM with 200-dimensional hidden layers. All the FFNNs have 2 hidden layers of 150 dimensions each. We use 0.4 variational dropout [15] for the LSTMs, 0.4 dropout for the FFNNs, and 0.5 dropout for the input embeddings. We model spans up to 8 words. For beam pruning, we use \(\lambda _{\text{C}}=0.3\) for coreference resolution and \(\lambda _{\text{R}}=0.4\) for relation extraction. For constructing the knowledge graph
[LaTeX Input Text end]

Answer in the following YAML format.

[YAML format start]
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
[YAML format end]

Only produce output in the YAML format specified above. Output no additional text.

### Response:
```

(default params except max output length 2048)

```
* text\_contains\_entities: true
* entities:
	- entity<1>:
		name: "1 layer BiLSTM"
		type: "model"
		has\_parameters: true
		parameters:
			- parameter<1>:
				name: "hidden layers"
				value: "200"
				context: "dimensions"
	- entity<2>:
		name: "FFNN"
		type: "model"
		has\_parameters: true
		parameters:
			- parameter<1>:
				name: "hidden layers"
				value: "150"
				context: "dimensions"
	- entity<3>:
		name: "LSTM"
		type: "model"
		has\_parameters: true
		parameters:
			- parameter<1>:
				name: "hidden layers"
				value: "200"
				context: "dimensions"
		- parameter<2>:
			name: "dropout"
			value: "0.4"
			context: "rate"
	- entity<4>:
		name: "input embeddings"
		type: "model"
		has\_parameters: true
		parameters:
			- parameter<1>:
				name: "hidden layers"
				value: "200"
				context: "dimensions"
		- parameter<2>:
			name: "dropout"
			value: "0.5"
			context: "rate"
	- entity<5>:
		name: "coreference resolution"
		type: "method"
		has\_parameters: false
	- entity<6>:
		name: "relation extraction"
		type: "method"
		has\_parameters: false
	- entity<7>:
		name: "beam pruning"
		type: "method"
		has\_parameters: false
	- entity<8>:
		name: "knowledge graph construction"
		type: "method"
		has\_parameters: false
```

### Var4

```
### Instruction:
In the context of machine learning and related fields, what (if any) are the entities (datasets, models, methods, loss functions, regularization techniques) mentioned in the LaTeX Input Text below? What (if any) are their parameters and values?

[LaTeX Input Text start]
Our system extends the implementation and hyper-parameters from Lee2017EndtoendNC with the following adjustments. We use a 1 layer BiLSTM with 200-dimensional hidden layers. All the FFNNs have 2 hidden layers of 150 dimensions each. We use 0.4 variational dropout [15] for the LSTMs, 0.4 dropout for the FFNNs, and 0.5 dropout for the input embeddings. We model spans up to 8 words. For beam pruning, we use \(\lambda _{\text{C}}=0.3\) for coreference resolution and \(\lambda _{\text{R}}=0.4\) for relation extraction. For constructing the knowledge graph
[LaTeX Input Text end]

Answer in the following YAML format.

[YAML format start]
text_contains_entities: true/false
entities:
  - entity<N>:
      name: "<entity name>"
      type: dataset/model/method/loss function/regularization technique
      has_parameters: true/false
      parameters:
        - parameter<M>:
            name: "<parameter name>"
            value: "<parameter value>"
            context: "<value context>"/null
[YAML format end]

Only produce output in the YAML format specified above. Output no additional text.

### Response:
```

(default params except max len 2048)

```
[YAML format start]
text_contains_entities: true
entities:
  - entity<0>:
      name: "1 layer BiLSTM"
      type: model
      has_parameters: false
  - entity<1>:
      name: "2 hidden layers of 150 dimensions each"
      type: model
      has_parameters: false
  - entity<2>:
      name: "0.4 variational dropout"
      type: model
      has_parameters: false
  - entity<3>:
      name: "0.4 dropout"
      type: model
      has_parameters: false
  - entity<4>:
      name: "0.5 dropout for input embeddings"
      type: model
      has_parameters: false
  - entity<5>:
      name: "model spans up to 8 words"
      type: model
      has_parameters: false
  - entity<6>:
      name: "For beam pruning, we use $\\lambda _{\\text{C}}=0.3$ for coreference resolution and $\\lambda _{\\text{R}}=0.4$ for relation extraction"
      type: model
      has_parameters: false
  - entity<7>:
      name: "For constructing the knowledge graph"
      type: model
      has_parameters: false
[YAML format end]
```