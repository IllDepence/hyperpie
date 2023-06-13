# First shot at LLM zero shot evaluation

## Prompt

```
In the context of machine learning and related fields, what (if any) are the entities (datasets, models, methods, loss functions, regularization techniques) mentioned in the LaTeX Input Text below? What (if any) are their parameters and values?

[LaTeX Input Text start]
{text}
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

with `{text}` replaced by the paragraph text.

<details><summary>numbers based on paragraph text before newline fix in here</summary>

(raw data in `completion_cache_llm_zero_shot_230602.json.xz`)

## Parameters

```
gpt_default_params = {
    "model": "text-davinci-003",
    "max_tokens": 512,
    "temperature": 0.0,
    "top_p": 1,
    "n": 1,
    "logprobs": 0,
    "echo": False,
} 
```

## Eval (“full info sets”)

**Filter:**  
Both ground truth and predictions ran through `hyperpie.data.filter_annots.require_apv`, i.e. only “full info sets” (`a<p<v[<c]`) are considered.

**Partial overlap: False**

| Method       | TP | FP | FN | Precision (P) | Recall (R) | F1 Score |
|--------------|----|----|----|---------------|------------|----------|
| ER           | 30 | 408| 111| 0.068         | 0.213      | 0.104    |
| ER + Clf     | 26 | 408| 115| 0.060         | 0.184      | 0.090    |
| Co-ref resol.| 4  | 494| 34 | 0.008         | 0.105      | 0.015    |
| Rel. extr.   | 12 | 397| 104| 0.029         | 0.103      | 0.046    |

**Partial overlap: True**

| Method       | TP | FP | FN | Precision (P) | Recall (R) | F1 Score |
|--------------|----|----|----|---------------|------------|----------|
| ER           | 85 | 345| 56 | 0.198         | 0.603      | 0.298    |
| ER + Clf     | 72 | 345| 69 | 0.173         | 0.511      | 0.258    |
| Co-ref resol.| 16 | 482| 22 | 0.032         | 0.421      | 0.060    |
| Rel. extr.   | 39 | 372| 77 | 0.095         | 0.336      | 0.148    |


## Eval (class specific)

### ER / ER + Clf

**False positives (exact match)**

* a: 94
* c: 48
* p: 142
* v: 124

**False positives (partial overlap)**

* a: 79
* c: 44
* p: 123
* v: 99

**False negatives (exact match)**

* a: 29
* c: 12
* p: 37
* v: 33

**False negatives (partial overlap)**

* a: 10
* c: 10
* p: 20
* v: 16

## Eval (“reqire parent entity”)

**Filter:**  
Both ground truth and predictions ran through `hyperpie.data.filter_annots.require_parents`, i.e. only parameters with artifacts, values with parameters, contexts with values.

**Partial overlap: False**

| Method       | TP | FP  | FN | Precision (P) | Recall (R) | F1 Score |
|--------------|----|-----|----|---------------|------------|----------|
| ER           | 374| 1214| 888| 0.236         | 0.296      | 0.262    |
| ER + Clf     | 363| 1214| 899| 0.230         | 0.288      | 0.256    |
| Co-ref resol.| 242| 1316| 954| 0.155         | 0.202      | 0.176    |
| Rel. extr.   |  12|  586| 119| 0.020         | 0.092      | 0.033    |

**Partial overlap: True**

| Method       | TP | FP  | FN | Precision (P) | Recall (R) | F1 Score |
|--------------|----|-----|----|---------------|------------|----------|
| ER           | 625|  962| 637| 0.394         | 0.495      | 0.439    |
| ER + Clf     | 595|  962| 667| 0.382         | 0.471      | 0.422    |
| Co-ref resol.| 338| 1214| 858| 0.218         | 0.283      | 0.246    |
| Rel. extr.   |  39|  561| 92 | 0.065         | 0.298      | 0.107    |


**False positives (exact match)**

* a: 714
* c: 48
* p: 330
* v: 122

**False positives (partial overlap)**

* a: 525
* c: 41
* p: 301
* v: 95

**False negatives (exact match)**

* a: 785
* c: 12
* p: 58
* v: 33

**False negatives (partial overlap)**

* a: 574
* c: 10
* p: 37
* v: 16

# Changed Parameters

* same prompt
* parameter `max_tokens` set from 512 to 2048

</details>

## Parameters

```
gpt_default_params = {
    "model": "text-davinci-003",
    "max_tokens": 2048,
    "temperature": 0.0,
    "top_p": 1,
    "n": 1,
    "logprobs": 0,
    "echo": False,
} 
```

## Eval (“reqire parent entity”)

**Partial overlap: False**

| Method       | TP | FP | FN | Precision (P) | Recall (R) | F1 Score |
|--------------|----|----|----|---------------|------------|----------|
| ER           | 523| 744| 739| 0.413         | 0.414      | 0.414    |
| ER + Clf     | 507| 744| 755| 0.405         | 0.402      | 0.404    |
| Co-ref resol.| 350| 704| 846| 0.332         | 0.293      | 0.311    |
| Rel. extr.   | 8  | 202| 123| 0.038         | 0.061      | 0.047    |

**Partial overlap: True**

| Method       | TP | FP | FN | Precision (P) | Recall (R) | F1 Score |
|--------------|----|----|----|---------------|------------|----------|
| ER           | 641| 634| 621| 0.503         | 0.508      | 0.505    |
| ER + Clf     | 619| 634| 643| 0.494         | 0.490      | 0.492    |
| Co-ref resol.| 390| 660| 806| 0.371         | 0.326      | 0.347    |
| Rel. extr.   | 15 | 194| 116| 0.072         | 0.115      | 0.088    |


**False positives (exact match)**

* a: 553
* p: 191

**False positives (partial overlap)**

* a: 455
* p: 179

**False negatives (exact match)**

* a: 633
* c: 13
* p: 48
* v: 45

**False negatives (partial overlap)**

* a: 527
* c: 13
* p: 38
* v: 43
