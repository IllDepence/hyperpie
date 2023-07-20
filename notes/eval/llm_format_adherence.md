# LLM format adherence eval

### text-davinci-0003

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

| Category               | Count |
| ---------------------- | ----- |
| No YAML found          |     0 |
| Empty YAML             |     0 |
| Garbage around YAML    |    28 |
| YAML parse fail        |     0 |
| Coarse Structure error |     0|

##### Entity eval

| Criterion      | Valid | Invalid |
| -------------- | ----- | ------- |
| Entity in text |  2362 |    1005 |
| Entity type    |  2114 |    1253 |
| Artifact ID    |  3351 |      16 |
| Parameter ID   |     0 |     758 |
| Value ID       |     0 |     741 |
| Context ID     |     0 |     719 |


### GALACTICA 120b

(still running)

Paragraphs total: 322

##### Paragraph eval

| Category               | Count |
| ---------------------- | ----- |
| No YAML found          |     0 |
| Empty YAML             |    63 |
| Garbage around YAML    |   116 |
| YAML parse fail        |     0 |
| Coarse Structure error |     5 |

##### Entity eval

| Criterion      | Valid | Invalid |
| -------------- | ----- | ------- |
| Entity in text |   322 |     612 |
| Entity type    |   837 |      97 |
| Artifact ID    |   934 |       0 |
| Parameter ID   |     0 |     771 |
| Value ID       |     0 |     759 |
| Context ID     |     0 |     725 |
