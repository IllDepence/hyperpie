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
