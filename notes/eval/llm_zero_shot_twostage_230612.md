# LLM zero shot evaluation with two-stage prompt

(preliminary numbers; post-processing of LLM output not final yet)

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
| ER           | 367| 1200| 895| 0.234         | 0.291      | 0.259    |
| ER + Clf     | 358| 1200| 904| 0.230         | 0.284      | 0.254    |
| Co-ref resol.| 224| 1092| 972| 0.170         | 0.187      | 0.178    |
| Rel. extr.   | 2  | 235| 129| 0.008         | 0.015      | 0.011    |

**Partial overlap: True**

| Method       | TP | FP | FN | Precision (P) | Recall (R) | F1 Score |
|--------------|----|----|----|---------------|------------|----------|
| ER           | 648| 911| 614| 0.416         | 0.513      | 0.459    |
| ER + Clf     | 616| 911| 646| 0.403         | 0.488      | 0.442    |
| Co-ref resol.| 378| 942| 818| 0.286         | 0.316      | 0.300    |
| Rel. extr.   | 6  | 224| 125| 0.026         | 0.046      | 0.033    |


**False positives (exact match)**

* a: 897
* c: 13
* p: 165
* v: 125

**False positives (partial overlap)**

* a: 650
* c: 11
* p: 152
* v: 98

**False negatives (exact match)**

* a: 776
* c: 13
* p: 64
* v: 42

**False negatives (partial overlap)**

* a: 519
* c: 13
* p: 51
* v: 3
