* full\_annot.json: full set of text segments annotated in pilot study
* jkr\_221110.json: set of text segments annotated by two annotators (annotator jkr)
* tsa\_221110.json: set of text segments annotated by two annotators (annotator tsa)

```
$ python3 iaa.py
105 out of 133 entities (0.789) match exactly
88 out of 132 relations (0.667) match exactly
Cohen’s kappa based on character level entity class: 0.867
Cohen’s kappa based on character level entity class and relationtarget span: 0.737
```
