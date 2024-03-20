# HyperPIE

Hyperparameter Information Extraction from Scientific Publications

![](doc/img/schema_visual.png)

This repository contains the source code, data, and documentation for the ECIR paper “HyperPIE: Hyperparameter Information Extraction from Scientific Publications”.

▶ [paper](https://doi.org/10.1007/978-3-031-56060-6_17) ([author copy on arXiv](https://doi.org/10.48550/arXiv.2312.10638))

## Contents

* **Documentation**
    * [Annotation guidelines](doc/annotation_guidelines.md)
    * [Dataset format](data/preprocessed_data_format.md)
    * [Prompt example zero-shot](doc/prompt_examples.md#Zero-shot)
    * [Prompt example few-shot](doc/prompt_examples.md#Few-shot)
* **Data**
    * Manually annotated data
        * [data](data/#README)
        * [format](data/preprocessed_data_format.md)
    * LLM completions
        * `code/hyperpie/llm/completion_cache.json.xz`
        * `code/hyperpie/llm/completion_cache_few_shot.json.xz`
* **Code**
    * [Fine-tuned models](code/PL-Marker/)
    * [LLMs](code/hyperpie/)

## Cite as

```
@inproceedings{Saier2024HyperPIE,
  title         = {{HyperPIE: Hyperparameter Information Extraction from Scientific Publications}},
  author        = {Saier, Tarek and Ohta, Mayumi and Asakura, Takuto and F{\"a}rber, Michael},
  booktitle     = {Advances in Information Retrieval},
  series        = {Lecture Notes in Computer Science},
  volume        = {14609},
  pages         = {254--269},
  year          = {2024},
  month         = mar,
  doi           = {10.1007/978-3-031-56060-6_17},
  publisher     = {Springer Nature Switzerland},
  isbn          = {978-3-031-56060-6}
}
```
