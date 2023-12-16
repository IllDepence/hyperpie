**Tags**: `#baseline-model`, `#data-to-use`, `#ner-re-pipeline`, `#join-ner-re`, `#llm`

# Similar Data / Model

### Some Paper

* Some notes


# Annotation Schema / Data Modeling

### Ontology for Informatics Research Artifacts (10.1007/978-3-030-80418-3\_23)

* TODO

### Extracting Information about Research Resources from Scholarly Papers (10.1007/978-3-031-21756-2\_35)

* TODO

# LLMs

### Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond (arXiv:2304.13712)

- referece for
    - LLMs are preferrable if thereis limited or no annotated training data
    - LLMs preferrable if there is a strong reliance on background knowledge
    - LLM tendency to closed-sourcing
    - API-based research could become predominant for academic community 
- Fig 1 nice lineage illustration from word2vec to BERT, T5, and GTP-4
- Section on LLMs for NER. Verdict: only a low level intermmediate task at which smaller dedicated models outperform LLMs, but task itself might be irrelevant if LLM can perform the high level task without the need of the intermmediate step
- Sec 5.1 nice little summary of ressources used for training LLMs
- repo: https://github.com/Mooler0410/LLMsPracticalGuide

### Structured information extraction from complex scientific text with fine-tuned large language models (arXiv:2212.05238)
- task: joined NER and RE
    - link dopants to host material
    - catalog metal-organic frameworks
    - general chem/phrase/... IE
- model: fine-tuned GPT3 (davinci 175B)
- in: single sentence/multiple sentences
- out: natural language / JSON

- challenge: in application domain (matsci) simple binary relations apparently are not enough

- during fine-tuning:
    - prompt-completion pairs with a consistent output format schema
    - use cross-entropy loss
    - use model pre-trained on 100 samples for annotation suggestions (see 2.5 x speed increase in annotation)

- eval
    - two types of F1 measures (exact/loose)
    - custom metrics (parsability, Jaro-Winkler similarity, etc. averaged)
    - also do manual eval (or define a kind of manual eval?)

- note that performance is limited by how comprehensively and consistently the annotation schema can be defined
    -> reducing ambiguity in the annotation schema is seen as a potential source of performance improvement
- note that class definitions drifted as annotators found more edge cases during the annotation process

### Large language models are few-shot clinical information extractors (EMLP 2022)- several types of clinical IE (but not joint NER+RE)

- justify need for new data by the fact that LLMs (here GPT3) might have been trained on existing benchmark data
- very extensive appendix with experiment information
- note that LLM output can be simultaneously qualitatively impressive but strictly speaking incorrect on the token level (trivial syntax mistakes etc)
- note tendency of LLM to output non-trivial answer even if there is none
    - suggest chaining promts, first asking if there is X contained in text, and if so give list
