# LLM zero shot evaluation with two-stage prompt

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
| ER           | 496| 1071| 766| 0.317         | 0.393      | 0.351    |
| ER + Clf     | 485| 1071| 777| 0.312         | 0.384      | 0.344    |
| Co-ref resol.| 294| 1022| 902| 0.223         | 0.246      | 0.234    |
| Rel. extr.   | 2  | 235| 129| 0.008         | 0.015      | 0.011    |

**Partial overlap: True**

| Method       | TP | FP | FN | Precision (P) | Recall (R) | F1 Score |
|--------------|----|----|----|---------------|------------|----------|
| ER           | 672| 891| 590| 0.430         | 0.532      | 0.476    |
| ER + Clf     | 642| 891| 620| 0.419         | 0.509      | 0.459    |
| Co-ref resol.| 386| 954| 810| 0.288         | 0.323      | 0.304    |
| Rel. extr.   | 14 | 220| 117| 0.060         | 0.107      | 0.077    |


**False positives (exact match)**

* a: 772
* c: 13
* p: 163
* v: 123

**False positives (partial overlap)**

* a: 629
* c: 11
* p: 153
* v: 98

**False negatives (exact match)**

* a: 649
* c: 13
* p: 62
* v: 42

**False negatives (partial overlap)**

* a: 495
* c: 13
* p: 52
* v: 30



## Eval (“reqire parent entity”) — string-matched surface forms

(instead of using second prompt for annotating entity surface forms in text, use simple string matching (same as in single prompt setup))

**Partial overlap: False**

| Method       | TP | FP | FN | Precision (P) | Recall (R) | F1 Score |
|--------------|----|----|----|---------------|------------|----------|
| ER           | 548| 1251| 714| 0.305         | 0.434      | 0.358    |
| ER + Clf     | 535| 1251| 727| 0.300         | 0.424      | 0.351    |
| Co-ref resol.| 356| 2602| 840| 0.120         | 0.298      | 0.171    |
| Rel. extr.   | 19 | 479| 112| 0.038         | 0.145      | 0.060    |

**Partial overlap: True**

| Method       | TP | FP | FN | Precision (P) | Recall (R) | F1 Score |
|--------------|----|----|----|---------------|------------|----------|
| ER           | 679| 962| 583| 0.414         | 0.538      | 0.468    |
| ER + Clf     | 657| 962| 605| 0.406         | 0.521      | 0.456    |
| Co-ref resol.| 396| 2558| 800| 0.134         | 0.331      | 0.191    |
| Rel. extr.   | 38 | 466| 93 | 0.075         | 0.290      | 0.120    |


**False positives (exact match)**

* a: 561
* c: 79
* p: 227
* v: 384

**False positives (partial overlap)**

* a: 461
* c: 78
* p: 215
* v: 208

**False negatives (exact match)**

* a: 629
* c: 8
* p: 46
* v: 31

**False negatives (partial overlap)**

* a: 521
* c: 8
* p: 37
* v: 17



## Eval (“reqire parent entity”) — string-matched surface forms IGNORING P&C

1. instead of using second prompt for annotating entity surface forms in text, use simple string matching (same as in single prompt setup)
2. use YAML conversion code for single prompt, leading to values and contexts always being ignored (see 0 false positives below)

**TODO: look into class specific F1 scores**

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
| Rel. extr.   | 16 | 194| 115| 0.076         | 0.122      | 0.094    |


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
