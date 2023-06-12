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
| ER           | 283| 853| 979| 0.249         | 0.224      | 0.236    |
| ER + Clf     | 274| 853| 988| 0.243         | 0.217      | 0.229    |
| Co-ref resol.| 0  | 0  | 1196| 1.000         | 0.000      | 0.000    |
| Rel. extr.   | 1  | 236| 130| 0.004         | 0.008      | 0.005    |

**Partial overlap: True**

| Method       | TP | FP | FN | Precision (P) | Recall (R) | F1 Score |
|--------------|----|----|----|---------------|------------|----------|
| ER           | 456| 682| 806| 0.401         | 0.361      | 0.380    |
| ER + Clf     | 424| 682| 838| 0.383         | 0.336      | 0.358    |
| Co-ref resol.| 4  | 0  | 1192| 1.000         | 0.003      | 0.007    |
| Rel. extr.   | 6  | 234| 125| 0.025         | 0.046      | 0.032    |


**False positives (exact match)**

* a: 628
* c: 11
* p: 114
* v: 100

**False positives (partial overlap)**

* a: 492
* c: 9
* p: 103
* v: 78

**False negatives (exact match)**

* a: 859
* c: 13
* p: 64
* v: 43

**False negatives (partial overlap)**

* a: 709
* c: 12
* p: 55
* v: 3
