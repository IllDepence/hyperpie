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
| ER           | 387| 1180| 875| 0.247         | 0.307      | 0.274    |
| ER + Clf     | 378| 1180| 884| 0.243         | 0.300      | 0.268    |
| Co-ref resol.| 204| 1112| 992| 0.155         | 0.171      | 0.162    |
| Rel. extr.   | 2  | 235| 129| 0.008         | 0.015      | 0.011    |

**Partial overlap: True**

| Method       | TP | FP | FN | Precision (P) | Recall (R) | F1 Score |
|--------------|----|----|----|---------------|------------|----------|
| ER           | 597| 971| 665| 0.381         | 0.473      | 0.422    |
| ER + Clf     | 561| 971| 701| 0.366         | 0.445      | 0.402    |
| Co-ref resol.| 302| 1028| 894| 0.227         | 0.253      | 0.239    |
| Rel. extr.   | 6  | 233| 125| 0.025         | 0.046      | 0.032    |


**False positives (exact match)**

* a: 876
* c: 13
* p: 167
* v: 124

**False positives (partial overlap)**

* a: 704
* c: 11
* p: 156
* v: 100

**False negatives (exact match)**

* a: 755
* c: 13
* p: 64
* v: 43

**False negatives (partial overlap)**

* a: 570
* c: 12
* p: 54
* v: 29
