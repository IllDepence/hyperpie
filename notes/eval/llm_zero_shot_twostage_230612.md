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
| ER           | 536| 1024| 726| 0.344         | 0.425      | 0.380    |
| ER + Clf     | 526| 1024| 736| 0.339         | 0.417      | 0.374    |
| Co-ref resol.| 356| 1756| 840| 0.169         | 0.298      | 0.215    |
| Rel. extr.   | 16 | 261| 115| 0.058         | 0.122      | 0.078    |

**Partial overlap: True**

| Method       | TP | FP | FN | Precision (P) | Recall (R) | F1 Score |
|--------------|----|----|----|---------------|------------|----------|
| ER           | 661| 818| 601| 0.447         | 0.524      | 0.482    |
| ER + Clf     | 644| 818| 618| 0.440         | 0.510      | 0.473    |
| Co-ref resol.| 394| 1714| 802| 0.187         | 0.329      | 0.238    |
| Rel. extr.   | 31 | 248| 100| 0.111         | 0.237      | 0.151    |


**False positives (exact match)**

* a: 561
* c: 46
* p: 184
* v: 233

**False positives (partial overlap)**

* a: 461
* c: 45
* p: 178
* v: 134

**False negatives (exact match)**

* a: 631
* c: 11
* p: 48
* v: 36

**False negatives (partial overlap)**

* a: 525
* c: 11
* p: 43
* v: 22


## Eval JSON (“reqire parent entity”) — string-matched surface forms

(JSON format output)

(same as above: instead of using second prompt for annotating entity surface forms in text, use simple string matching (same as in single prompt setup))

**Partial overlap: False**

| Method       | TP | FP | FN | Precision (P) | Recall (R) | F1 Score |
|--------------|----|----|----|---------------|------------|----------|
| ER           | 547| 1371| 715| 0.285         | 0.433      | 0.344    |
| ER + Clf     | 538| 1371| 724| 0.282         | 0.426      | 0.339    |
| Co-ref resol.| 354| 93460| 842| 0.004         | 0.296      | 0.007    |
| Rel. extr.   | 14 | 237| 117| 0.056         | 0.107      | 0.073    |

**Partial overlap: True**

| Method       | TP | FP | FN | Precision (P) | Recall (R) | F1 Score |
|--------------|----|----|----|---------------|------------|----------|
| ER           | 672| 1213| 590| 0.356         | 0.532      | 0.427    |
| ER + Clf     | 650| 1213| 612| 0.349         | 0.515      | 0.416    |
| Co-ref resol.| 388| 93430| 808| 0.004         | 0.324      | 0.008    |
| Rel. extr.   | 26 | 228| 105| 0.102         | 0.198      | 0.135    |


**False positives (exact match)**

* a: 579
* c: 459
* p: 206
* v: 127

**False positives (partial overlap)**

* a: 479
* c: 425
* p: 203
* v: 106

**False negatives (exact match)**

* a: 617
* c: 13
* p: 49
* v: 36

**False negatives (partial overlap)**

* a: 511
* c: 11
* p: 45
* v: 23


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
