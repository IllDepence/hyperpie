# LLM format adherence eval

### text-davinci-0003

##### Paragraph eval

Paragraphs total: 444

| Category               | Count | JSON eval: | Count |
| ---------------------- | ----- | ---------- | ----- |
| No YAML found          |     0 |            |     0 |
| Empty YAML             |     0 |            |     0 |
| Garbage around YAML    |     0 |            |     0 |
| YAML parse fail        |     0 |            |    2¹ |
| Coarse Structure error |     0 |            |     0 |

¹ one b/c of token limit, other one legitimate JSON format error  
  (`{'k': 'v', {'k': 'v'}, ...}` → dict in place of dict key)

##### Entity eval

| Criterion      | Valid | Invalid | JSON eval: | Valid | Invalid |
| -------------- | ----- | ------- | ---------- | ----- | ------- |
| Entity in text |   858 |     121 |            |   890 |      96 |
| Entity type    |   905 |      74 |            |   922 |      64 |
| Artifact ID    |   979 |       0 |            |   986 |       0 |
| Parameter ID   |   164 |      53 |            |   139 |      56 |
| Value ID       |   178 |      53 |            |   112 |      39 |
| Context ID     |    48 |     183 |            |    86 |      65 |


### Vicuna 13b v1.3

Paragraphs total: 444

##### Paragraph eval

| Category               | Count | JSON eval: | Count |
| ---------------------- | ----- | ---------- | ----- |
| No YAML found          |     0 |            |     0 |
| Empty YAML             |     0 |            |     0 |
| Garbage around YAML    |    28 |            |   205 |
| YAML parse fail        |    36 |            |  188¹ |
| Coarse Structure error |     0 |            |    82 |

¹ 166 of those b/c format template w/ things like `true/false` was just copied

##### Entity eval

| Criterion      | Valid | Invalid | JSON eval: | Valid | Invalid |
| -------------- | ----- | ------- | ---------- | ----- | ------- |
| Entity in text |  2362 |    1005 |            |   578 |     263 |
| Entity type    |  2114 |    1253 |            |   514 |     327 |
| Artifact ID    |  3351 |      16 |            |   630 |     211 |
| Parameter ID   |     0 |     758 |            |     0 |     408 |
| Value ID       |     0 |     741 |            |     0 |     404 |
| Context ID     |     0 |     719 |            |     0 |     404 |


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

| Category               | Count | JSON eval: | Count |
| ---------------------- | ----- | ---------- | ----- |
| No YAML found          |     0 |            |     0 |
| Empty YAML             |     0 |            |     0 |
| Garbage around YAML    |    42 |            |     6 |
| YAML parse fail        |    62 |            |  220¹ |
| Coarse Structure error |     0 |            |     9 |

¹ 164 of those b/c format template w/ things like `true/false` was just copied

##### Entity eval

| Criterion      | Valid | Invalid | JSON eval: | Valid | Invalid |
| -------------- | ----- | ------- | ---------- | ----- | ------- |
| Entity in text |  2089 |    1070 |            |   254 |     368 |
| Entity type    |  2788 |     371 |            |   511 |     111 |
| Artifact ID    |  2951 |     208 |            |   540 |      82 |
| Parameter ID   |     0 |     929 |            |     4 |     516 |
| Value ID       |     0 |     879 |            |     0 |     821 |
| Context ID     |     0 |     818 |            |     0 |     821 |


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
