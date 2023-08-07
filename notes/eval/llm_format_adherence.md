# LLM format adherence eval

### text-davinci-0003

##### Paragraph eval

Paragraphs total: 444

| Category               | Count |
| ---------------------- | ----- |
| No YAML found          |     0 |
| Empty YAML             |     0 |
| Garbage around YAML    |     0 |
| YAML parse fail        |     0 |
| Coarse Structure error |     0 |

##### Entity eval

| Criterion      | Valid | Invalid |
| -------------- | ----- | ------- |
| Entity in text |   858 |     121 |
| Entity type    |   905 |      74 |
| Artifact ID    |   979 |       0 |
| Parameter ID   |   164 |      53 |
| Value ID       |   178 |      53 |
| Context ID     |    48 |     183 |


### Vicuna 13b v1.3

Paragraphs total: 444

##### Paragraph eval

| Category               | Count | JSON eval: | Count |
| ---------------------- | ----- | ---------- | ----- |
| No YAML found          |     0 |            |   107 |
| Empty YAML             |     0 |            |     0 |
| Garbage around YAML    |    28 |            |   188 |
| YAML parse fail        |    36 |            |   270 |
| Coarse Structure error |     0 |            |     0 |

##### Entity eval

| Criterion      | Valid | Invalid | JSON eval: | Valid | Invalid |
| -------------- | ----- | ------- | ---------- | ----- | ------- |
| Entity in text |  2362 |    1005 |            |   495 |     161 |
| Entity type    |  2114 |    1253 |            |   442 |     214 |
| Artifact ID    |  3351 |      16 |            |   610 |      46 |
| Parameter ID   |     0 |     758 |            |     0 |     270 |
| Value ID       |     0 |     741 |            |     0 |     264 |
| Context ID     |     0 |     719 |            |     0 |     264 |


### GALACTICA 120b

Paragraphs total: 444

##### Paragraph eval

| Category               | Count |
| ---------------------- | ----- |
| No YAML found          |     0 |
| Empty YAML             |    88 |
| Garbage around YAML    |   161 |
| YAML parse fail        |    56 |
| Coarse Structure error |     5 |

##### Entity eval

| Criterion      | Valid | Invalid |
| -------------- | ----- | ------- |
| Entity in text |   435 |     850 |
| Entity type    |  1151 |     134 |
| Artifact ID    |  1285 |       0 |
| Parameter ID   |     0 |    1068 |
| Value ID       |     0 |    1052 |
| Context ID     |     0 |    1003 |


### WizardLM 13b v1.1

Paragraphs total: 444

##### Paragraph eval

| Category               | Count |
| ---------------------- | ----- |
| No YAML found          |     0 |
| Empty YAML             |     0 |
| Garbage around YAML    |    42 |
| YAML parse fail        |    62 |
| Coarse Structure error |     0 |

##### Entity eval

| Criterion      | Valid | Invalid |
| -------------- | ----- | ------- |
| Entity in text |  2089 |    1070 |
| Entity type    |  2788 |     371 |
| Artifact ID    |  2951 |     208 |
| Parameter ID   |     0 |     929 |
| Value ID       |     0 |     879 |
| Context ID     |     0 |     818 |


### Falcon 40b instruct

Paragraphs total: 444

##### Paragraph eval

| Category               | Count |
| ---------------------- | ----- |
| No YAML found          |     0 |
| Empty YAML             |     0 |
| Garbage around YAML    |   283 |
| YAML parse fail        |   269 |
| Coarse Structure error |     0 |

##### Entity eval

| Criterion      | Valid | Invalid |
| -------------- | ----- | ------- |
| Entity in text |   592 |     813 |
| Entity type    |  1105 |     300 |
| Artifact ID    |  1405 |       0 |
| Parameter ID   |     0 |     128 |
| Value ID       |     0 |     183 |
| Context ID     |     0 |      53 |
