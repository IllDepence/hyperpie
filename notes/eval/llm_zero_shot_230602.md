# First shot at LLM zero shot evaluation

(raw data in `completion_cache_llm_zero_shot_230602.json.xz`)

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
