# Zero-shot

```
### Instruction:
In the context of machine learning and related fields, what are the most important datasets, models, methods, loss functions, regularization techniques mentioned in the LaTeX Input Text below? What are their parameters and values?

[LaTeX Input Text start]
The architecture of the teachers is the backbone language model and a top classification layer for token-level classification. Specifically, we adopt RoBERTa and DistilRoBERTa as backbone for teacher 1 and teacher 2.
The corresponding student has the same architecture as their teacher. The max training epoch is 50 for all datasets. The training batch size is 16 for CoNLL03, Webpage, and Twitter and 32 for OntoNotes 5.0. The learning rate is set to 1e-5 for CoNLL03 and Webpage, and 2e-5 for OntoNotes 5.0 and Twitter.
For the pretraining stage with noisy labels, we separately train 1, 2, 12, and 6 epochs for CoNLL03, OntoNotes 5.0, Webpage, and Twitter datasets. For adaptive teacher learning, the confidence threshold \(\sigma _1\)  is 0.9 for all datasets.
In the fine-grained student ensemble, \(m\)  are 0.995, 0.995, 0.99, 0.995 and \(\sigma _2\)  is set to 0.8, 0.995, 0.8, and 0.75 for dataset CoNLL03, OntoNotes 5.0, Webpage, and Twitter, respectively.
[LaTeX Input Text end]

Answer in the following YAML format.

Format:
---
text_contains_entities: true/false
entities:
  - entity<N>:
      id: e<N>
      name: "<entity name>"
      type: dataset/model/method/loss function/regularization technique
      has_parameters: true/false
      parameters:
        - parameter<M>:
            id: p<N.M>
            name: "<parameter name>"
            has_values: true/false
            values:
              - value<O>:
                  value_id: v<N.M.O>
                  value: "<parameter value>"
                  context: "<value context>"/null
                  context_id: c<N.M.O>/null
...

Only produce output in the YAML format specified above. Output no additional text. Only include the most important datasets, models, methods, loss functions, regularization techniques mentioned in the LaTeX Input Text. Then end your response.

### Response:
---
text_contains_entities:
```


### Few-shot

```
### Instruction:
In the context of machine learning and related fields, what are the most important datasets, models, methods, loss functions, regularization techniques mentioned in the LaTeX Input Text below? What are their parameters and values?

Answer in the following YAML format.

Format:
---
has_entities: true/false
entities:
  - entity<N>:
      id: a<N>
      name: "<entity name>"
      type: dataset/model/method/loss function/regularization technique
      has_parameters: true/false
      parameters:
        - parameter<M>:
            id: p<N.M>
            name: "<parameter name>"
            has_values: true/false
            values:
              - value<O>:
                  value_id: v<N.M.O>
                  value: "<parameter value>"
                  context: "<value context>"/null
                  context_id: c<N.M.O>/null
...

Here are several examples.

### Example 1:

[LaTeX Input Text start]
Natural Language Processing (NLP) and Computer Vision (CV) are among the most important Machine Learning fields. The last breakthroughs resulted in models that share similar concepts, such as representation learning or transfer learning [0], [1]. It is sufficient to mention ResNet [2] or DenseNet [3] in the CV and BERT [4] or RoBERTa [5] from the NLP field. These models are preliminarily pre-trained on the massive amount of general data and can be fine-tuned to suit the downstream task such as classification or entailment. In the field of CV, there is a very lively discussion about using Metric Learning methods for fine-tuning pre-trained models. Metric Learning tries to shape the embedding space so that similar data are close to each other while dissimilar are far from each other [6]. 
[LaTeX Input Text end]

### Response 1:
---
has_entities: true
entities:
- entity1:
    id: a1
    name: "ResNet"
    has_parameters: false
- entity2:
    id: a2
    name: "DenseNet"
    has_parameters: false
- entity3:
    id: a3
    name: "BERT"
    has_parameters: false
- entity4:
    id: a4
    name: "RoBERTa"
    has_parameters: false
...

### Example 2:

[LaTeX Input Text start]
Our system extends the implementation and hyper-parameters from Lee2017EndtoendNC with the following adjustments. We use a 1 layer BiLSTM with 200-dimensional hidden layers. All the FFNNs have 2 hidden layers of 150 dimensions each. We use 0.4 variational dropout [15] for the LSTMs, 0.4 dropout for the FFNNs, and 0.5 dropout for the input embeddings. We model spans up to 8 words. For beam pruning, we use \(\lambda _{\text{C}}=0.3\)  for coreference resolution and \(\lambda _{\text{R}}=0.4\)  for relation extraction. For constructing the knowledge graph, we use the following heuristics to normalize the entity phrases. We replace all acronyms with their corresponding full name and normalize all the plural terms with their singular counterparts. {TABLE}
[LaTeX Input Text end]

### Response 2:
---
has_entities: true
entities:
- entity1:
    id: a1
    name: "system"
    has_parameters: true
    parameters:
    - parameter9:
        id: p9
        name: "dropout"
        has_values: true
        values:
        - value7:
            value_id: v7
            value: "0.5"
            context: "input embeddings"
            context_id: c1
        - value6:
            value_id: v6
            value: "0.4"
            context: null
            context_id: null
- entity9:
    id: a9
    name: "BiLSTM"
    has_parameters: true
    parameters:
    - parameter7:
        id: p7
        name: "dimensional hidden layers"
        has_values: true
        values:
        - value2:
            value_id: v2
            value: "200"
            context: null
            context_id: null
    - parameter10:
        id: p10
        name: "layer"
        has_values: true
        values:
        - value1:
            value_id: v1
            value: "1"
            context: null
            context_id: null
- entity7:
    id: a7
    name: "FFNNs"
    has_parameters: true
    parameters:
    - parameter6:
        id: p6
        name: "hidden layers"
        has_values: true
        values:
        - value3:
            value_id: v3
            value: "2"
            context: null
            context_id: null
    - parameter7:
        id: p7
        name: "dimensional hidden layers"
        has_values: true
        values:
        - value4:
            value_id: v4
            value: "150"
            context: null
            context_id: null
    - parameter9:
        id: p9
        name: "dropout"
        has_values: true
        values:
        - value7:
            value_id: v7
            value: "0.5"
            context: "input embeddings"
            context_id: c1
        - value6:
            value_id: v6
            value: "0.4"
            context: null
            context_id: null
- entity14:
    id: a14
    name: "LSTMs"
    has_parameters: true
    parameters:
    - parameter8:
        id: p8
        name: "variational dropout"
        has_values: true
        values:
        - value5:
            value_id: v5
            value: "0.4"
            context: null
            context_id: null
- entity11:
    id: a11
    name: "beam pruning"
    has_parameters: true
    parameters:
    - parameter5:
        id: p5
        name: "\lambda _{\text{R}}"
        has_values: true
        values:
        - value10:
            value_id: v10
            value: "0.4"
            context: "relation extraction"
            context_id: c3
    - parameter3:
        id: p3
        name: "\lambda _{\text{C}}"
        has_values: true
        values:
        - value9:
            value_id: v9
            value: "0.3"
            context: "coreference resolution"
            context_id: c2
- entity3:
    id: a3
    name: "knowledge graph"
    has_parameters: false
...

### Example 3:

[LaTeX Input Text start]
Author affiliations appear in different variations (e .g., “MIT” and “Massachusetts Institute of Technology”) in JATS XML files. We normalize author affiliations using the Research Organization Registry (ROR) [20]. The ROR is a community-led registry of identifiers for research organizations.https://ror.org/about/, last accessed on 2021-12-14 It provides an API, which allows to retrieve, search, and filter the organizations indexed in the ROR. We identify the corresponding ROR entity for each author affiliation string using this API.https://github.com/ror-community/ror-api, last accessed on 2021-12-14 Although strings of some author affiliations are marked up with “institution” and “country,”An example: <institution>University of Minnesota</institution>, Minneapolis, MN 55455, <country>USA</country> we used full strings of author affiliation as queries for consistency. The ROR API returns a list of organizations that are matched to a query sorted by confidence scores. We use the returned organization with the highest confidence score and if the field “chosen” (i.e., binary indicator of whether the score is high enough to consider the organization correctly matched) is true. For the 260,231 authors, there are 335,188 affiliations. Among them, 273,804 affiliations (81.69%) can be linked to a ROR entity. In our analysis, we consider citation bias on the institution and country level. Thus, we use the name and country information of the organizations. 
[LaTeX Input Text end]

### Response 3:
---
has_entities: false
...

### Example 4:

[LaTeX Input Text start]
So far, from the E2E model, we have the probability of a token being a named entity, and only if we could have the same information from the LM, we can condition it using the modified prefix beam search decoding. This can be achieved, if we replace all the individual person tokens with a \(<\) person\(>\)  token as shown in Figure REF  and train an LM (we call it semantic LM or S-LM) on the updated corpus again. The S-LM would learn the probability of the next token being a \(<\) PER\(>\)  instead of the exact names. Therefore, at the test time, we now have the named entity tag information from the E2E model and the S-LM. At the test time, now we can condition the E2E model output with the S-LM to rank the top n-paths using the modified prefix beam search. The modified "prefix beam search" decoding is such that, it assigns a higher probability to the output of E2E model if the same can be inferred from the S-LM or otherwise. Furthermore, while scoring the top beams, we can condition the named entities on actual person names from a pre-defined dictionary of names. 
[LaTeX Input Text end]

### Response 4:
---
has_entities: true
entities:
- entity2:
    id: a2
    name: "E2E model"
    has_parameters: false
- entity14:
    id: a14
    name: "prefix beam search decoding"
    has_parameters: false
...

### Example 5:

[LaTeX Input Text start]
We examine citation bias in bioRxiv, a preprint server in the field of biology, because preprints deposited to bioRxiv provide sufficient information regarding author affiliations. We analyze citations of more than 36,000 preprints deposited between November 2013 (i.e., the launch of bioRxiv) and June 2019 and their publisher versions. We use the COCI (OpenCitations Index of Crossref open DOI-to-DOI references) [13] as citation data. To measure citation inequality, we calculate the Gini coefficients \(G\) , following previous studies [24], [8]. In our analysis, we can confirm a citation bias, especially for preprints at different affiliation levels (i.e., institutions, countries), as we find that preprints have twice the citation inequality as the publisher versions (e.g., \(G=0.23\)  for preprints and \(G=0.12\)  for publisher versions at the institution level). Furthermore, we observe larger citation inequalities for preprints than those for publisher versions in different journal types that are mega-journalsMega-journals are journals that solely focus on scientific trustworthiness [2] in the process of peer-review, compared to other journals., disciplinary journals, and prestigious journals (e.g., Nature and Science). 
[LaTeX Input Text end]

### Response 5:
...
has_entities: true
entities:
- entity3:
    id: a3
    name: "COCI (OpenCitations Index of Crossref open DOI-to-DOI references)"
    has_parameters: false
...

Only produce output in the YAML format specified above. Output no additional text. Only include the most important datasets, models, methods, loss functions, regularization techniques mentioned in the LaTeX Input Text. Then end your response.

[LaTeX Input Text start]
We use the sentence in Fig. REF , “BERT, formally published at NAACL-HLT 2019, leads to a significant change of NLP” as an example of using CRPSE for citation recommendation. First, using scispaCy, entities BERT, NAACL-HLT and NLP are detected. Through the checking in the constructed published scientific entity-papers mapping dataset, only BERT is included in this dataset. Therefore, only the entity BERT will be passed to the next step for recommendation. When \(K\)  is set to be 5 which means 5 papers are required to be recommended and the cooccurrence count-based sorting criterion is used, the recommended papers for the entity BERT in order are “BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding” [15], “Deep Contextualized Word Representations” [47], “Attention Is All You Need” [56], “Improving Language Understanding by Generative Pre-Training” [48] and “XLNet: Generalized Autoregressive Pretraining for Language Understanding” [65]. Among these papers, the top one is the source paper proposing the entity BERT and the other four papers are also related to the entity BERT. 
[LaTeX Input Text end]

### Response:
```
